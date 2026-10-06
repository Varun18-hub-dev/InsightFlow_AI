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
from app.db.schemas import ChatRequest, ChatResponse
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

        try:
            from app.reranking.factory import get_reranker
            from app.retrieval.hybrid_retriever import HybridRetriever

            retriever = HybridRetriever(db)
            retrieved = await retriever.retrieve(
                query=query,
                filters={"user_id": user_id},
                final_top_k=settings.TOP_K_RETRIEVAL,
            )
            reranker = get_reranker()
            reranked = await reranker.rerank(query, retrieved, top_k=settings.TOP_K_RERANK)

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
                })

            context = "\n\n---\n\n".join(context_parts) if context_parts else "No context available."
            system_prompt = prompt_manager.get_prompt("rag", "system")
            answer_prompt = prompt_manager.render("rag", "answer", context=context, question=query)

            provider = LLMProviderFactory.get_provider()
            async for token in provider.chat_stream([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": answer_prompt},
            ]):
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'message_id': message_id})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'content': 'An error occurred during streaming'})}\n\n"
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
                    )
                    save_db.add(asst_msg)
                    await save_db.commit()
            except Exception as e:
                logger.error("stream_save_failed", error=str(e))

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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

