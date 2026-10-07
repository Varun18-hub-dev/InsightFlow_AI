import json
import time
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import run_graph
from app.api.deps import get_current_user, get_user_by_token
from app.core.config import settings
from app.core.prompt_manager import prompt_manager
from app.db.base import get_db
from app.db.models import Conversation, Message, User
from app.db.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationDetailResponse,
    ConversationResponse,
    ConversationUpdate,
    MessageResponse,
)
from app.llm.factory import LLMProviderFactory
from app.services.redis_service import get_redis_service

router = APIRouter()
logger = structlog.get_logger()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    redis = get_redis_service()
    user_id = str(current_user.id)

    # Rate limit
    if not await redis.rate_limit_check(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    # Cache check
    cache_key = redis.generate_cache_key(request.query, user_id)
    cached = await redis.get_cached_response(cache_key)
    if cached:
        logger.info("cache_hit", user_id=user_id)
        return ChatResponse(**cached)

    # Get or create conversation
    conv_id = request.conversation_id
    if conv_id:
        stmt = select(Conversation).where(
            Conversation.id == conv_id,
            Conversation.user_id == current_user.id
        )
        result = await db.execute(stmt)
        conv = result.scalar_one_or_none()
        if not conv:
            conv_id = None

    if not conv_id:
        conv = Conversation(user_id=current_user.id, title=request.query[:50])
        db.add(conv)
        await db.flush()
        conv_id = str(conv.id)

    # Save user message
    user_msg = Message(
        conversation_id=conv_id,
        user_id=current_user.id,
        role="user",
        content=request.query,
    )
    db.add(user_msg)
    await db.flush()

    start = time.time()
    logger.info("chat_started", user_id=user_id, query_len=len(request.query), conv_id=conv_id)

    # Run LangGraph orchestration
    state = await run_graph(
        query=request.query,
        user_id=user_id,
        db=db,
        conversation_id=conv_id,
        document_ids=request.document_ids or [],
    )

    answer = state.get("answer") or "I was unable to generate a response."
    sources = state.get("sources") or []
    confidence = float(state.get("confidence") or 0.0)

    # Save assistant message
    asst_msg = Message(
        conversation_id=conv_id,
        user_id=current_user.id,
        role="assistant",
        content=answer,
        sources=sources,
        metadata=state.get("metadata") or {},
    )
    db.add(asst_msg)
    await db.commit()

    response = ChatResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        conversation_id=conv_id,
        message_id=str(asst_msg.id),
    )

    # Cache
    await redis.set_cached_response(cache_key, response.model_dump())

    logger.info(
        "chat_complete",
        latency=round(time.time() - start, 3),
        sources=len(sources),
        intent=state.get("intent"),
    )
    return response


async def _run_chat_stream(query: str, conversation_id: str | None, current_user: User, db: AsyncSession):
    user_id = str(current_user.id)

    logger.info(
        "stream_started",
        user_id=user_id,
        query_len=len(query),
        conversation_id=conversation_id,
    )

    # Get or create conversation with user_id authorization
    conv_id = conversation_id
    if conv_id:
        stmt = select(Conversation).where(
            Conversation.id == conv_id,
            Conversation.user_id == current_user.id,
        )
        result = await db.execute(stmt)
        conv = result.scalar_one_or_none()
        if not conv:
            conv_id = None

    if not conv_id:
        conv = Conversation(user_id=current_user.id, title=query[:50])
        db.add(conv)
        await db.flush()
        conv_id = str(conv.id)

    # Save user message
    user_msg = Message(
        conversation_id=conv_id,
        user_id=current_user.id,
        role="user",
        content=query,
    )
    db.add(user_msg)
    await db.flush()
    await db.commit()

    async def generate():
        full_answer = ""
        sources = []
        message_id = str(uuid.uuid4())
        gen_start_time = time.time()

        try:
            from app.reranking.factory import get_reranker
            from app.retrieval.hybrid_retriever import HybridRetriever

            # Retrieval stage
            retrieval_start = time.time()
            logger.info("retrieval_started", user_id=user_id, query_len=len(query))
            retriever = HybridRetriever(db)
            retrieved = await retriever.retrieve(
                query=query,
                filters={"user_id": user_id},
                final_top_k=settings.TOP_K_RETRIEVAL,
            )
            retrieval_elapsed = round(time.time() - retrieval_start, 3)
            logger.info(
                "retrieval_completed",
                user_id=user_id,
                retrieved_count=len(retrieved),
                elapsed_seconds=retrieval_elapsed,
            )

            # Reranking stage
            rerank_start = time.time()
            logger.info("reranking_started", user_id=user_id, candidate_count=len(retrieved))
            reranker = get_reranker()
            reranked = await reranker.rerank(query, retrieved, top_k=settings.TOP_K_RERANK)
            rerank_elapsed = round(time.time() - rerank_start, 3)
            logger.info(
                "reranking_completed",
                user_id=user_id,
                reranked_count=len(reranked),
                elapsed_seconds=rerank_elapsed,
            )

            context_parts = []
            for chunk in reranked:
                meta = chunk.metadata
                fname = meta.get("filename", "doc")
                page = meta.get("page_number", "?")
                context_parts.append(f"[Source: {fname}, Page {page}]\n{chunk.content}")
                sources.append({
                    "document": fname,
                    "page": page,
                    "chunk_id": chunk.id,
                    "score": round(chunk.score, 4),
                    "snippet": chunk.content.strip() if chunk.content else "",
                })

            context = "\n\n---\n\n".join(context_parts) if context_parts else "No context available."
            system_prompt = prompt_manager.get_prompt("rag", "system")
            answer_prompt = prompt_manager.render("rag", "answer", context=context, question=query)

            logger.info(
                "prompt_loaded",
                user_id=user_id,
                system_prompt_len=len(system_prompt),
                answer_prompt_len=len(answer_prompt),
                context_chunks=len(context_parts),
            )

            provider = LLMProviderFactory.get_provider()
            model_name = provider.get_model_name() if hasattr(provider, "get_model_name") else "unknown"

            llm_stream_start = time.time()
            logger.info(
                "llm_stream_started",
                user_id=user_id,
                model=model_name,
                system_prompt_len=len(system_prompt),
                answer_prompt_len=len(answer_prompt),
            )

            first_token_logged = False
            token_count = 0

            async for token in provider.chat_stream([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": answer_prompt},
            ]):
                token_count += 1
                if not first_token_logged:
                    first_token_time = round(time.time() - llm_stream_start, 3)
                    logger.info(
                        "llm_first_token",
                        user_id=user_id,
                        model=model_name,
                        time_to_first_token_seconds=first_token_time,
                    )
                    first_token_logged = True

                full_answer += token
                token_payload = json.dumps({"type": "token", "content": token})
                logger.info(
                    "sse_token_yield",
                    event_type="token",
                    token_index=token_count,
                    content_length=len(token),
                )
                yield f"data: {token_payload}\n\n"

            llm_stream_elapsed = round(time.time() - llm_stream_start, 3)
            logger.info(
                "llm_stream_completed",
                user_id=user_id,
                model=model_name,
                total_tokens=token_count,
                answer_len=len(full_answer),
                elapsed_seconds=llm_stream_elapsed,
            )

            transparency_meta = {
                "intent": "Document QA",
                "retrieved_count": len(retrieved),
                "reranked_count": len(reranked),
                "context_characters": sum(len(c) for c in context_parts),
                "model": "Gemini Flash Lite",
                "sources_count": len(sources),
                "response_time_seconds": round(time.time() - gen_start_time, 2),
                "generation_time_seconds": llm_stream_elapsed,
            }
            sources_payload = json.dumps({
                "type": "sources",
                "sources": sources,
                "message_id": message_id,
                "conversation_id": conv_id,
                "metadata": transparency_meta,
            })
            logger.info(
                "sse_sources_yield",
                event_type="sources",
                sources_count=len(sources),
                content_length=len(sources_payload),
            )
            yield f"data: {sources_payload}\n\n"

            done_payload = json.dumps({"type": "done", "conversation_id": conv_id})
            logger.info(
                "sse_done_yield",
                event_type="done",
                content_length=len(done_payload),
            )
            yield f"data: {done_payload}\n\n"

        except Exception as e:
            gen_elapsed = round(time.time() - gen_start_time, 3)
            err_str = str(e)
            if "503" in err_str or "high demand" in err_str.lower():
                user_msg = "The AI model is temporarily experiencing high demand. Please retry in a few moments."
            elif "timeout" in err_str.lower():
                user_msg = "The AI request timed out. Please retry in a few moments."
            else:
                user_msg = "An error occurred during streaming."

            logger.error(
                "stream_error",
                user_id=user_id,
                exception_class=type(e).__name__,
                sanitized_message=err_str,
                elapsed_seconds=gen_elapsed,
                error=err_str,
                error_type=type(e).__name__,
            )
            error_payload = json.dumps({"type": "error", "content": user_msg, "details": err_str})
            logger.info(
                "sse_error_yield",
                event_type="error",
                content_length=len(error_payload),
            )
            yield f"data: {error_payload}\n\n"
        finally:
            try:
                # Reload DB session to avoid stale state
                from app.db.base import AsyncSessionLocal
                async with AsyncSessionLocal() as save_db:
                    asst_msg = Message(
                        id=message_id,
                        conversation_id=conv_id,
                        user_id=current_user.id,
                        role="assistant",
                        content=full_answer,
                        sources=sources,
                        msg_metadata=transparency_meta if "transparency_meta" in locals() else None,
                    )
                    save_db.add(asst_msg)
                    await save_db.commit()
            except Exception as e:
                logger.error("stream_save_failed", error=str(e))

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/stream")
async def chat_stream_post(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Server-Sent Events streaming chat endpoint (POST for programmatic clients)."""
    return await _run_chat_stream(request.query, request.conversation_id, current_user, db)


@router.get("/chat/stream")
async def chat_stream_get(
    query: str,
    token: str = Query(...),
    conversation_id: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Server-Sent Events streaming chat endpoint (GET for browser EventSource)."""
    user = await get_user_by_token(token, db)
    return await _run_chat_stream(query, conversation_id, user, db)


@router.get("/chat/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all conversations for the current authenticated user."""
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
    )
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/chat/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get conversation details and full message history."""
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID") from None

    stmt = select(Conversation).where(
        Conversation.id == conv_uuid,
        Conversation.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg_stmt = (
        select(Message)
        .where(Message.conversation_id == conv_uuid)
        .order_by(Message.created_at.asc())
    )
    msg_result = await db.execute(msg_stmt)
    messages = msg_result.scalars().all()

    return ConversationDetailResponse(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[
            MessageResponse(
                id=m.id,
                conversation_id=m.conversation_id,
                role=m.role,
                content=m.content,
                sources=m.sources or [],
                metadata=m.msg_metadata or {},
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


@router.patch("/chat/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    update_in: ConversationUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Rename/update conversation title."""
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID") from None

    stmt = select(Conversation).where(
        Conversation.id == conv_uuid,
        Conversation.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv.title = update_in.title
    await db.commit()
    await db.refresh(conv)
    return conv


@router.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete conversation and all its messages."""
    try:
        conv_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID") from None

    stmt = select(Conversation).where(
        Conversation.id == conv_uuid,
        Conversation.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    from sqlalchemy import delete
    await db.execute(delete(Message).where(Message.conversation_id == conv_uuid))
    await db.delete(conv)
    await db.commit()
    return {"status": "deleted", "id": conversation_id}

