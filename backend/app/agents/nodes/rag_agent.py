"""
RAG Agent node: hybrid retrieval → reranking → LLM → citations.
"""
import time

import structlog

from app.agents.state import AgentState
from app.core.config import settings
from app.core.prompt_manager import prompt_manager
from app.llm.factory import LLMProviderFactory
from app.reranking.factory import get_reranker
from app.retrieval.hybrid_retriever import HybridRetriever

logger = structlog.get_logger()

def _build_context(chunks) -> tuple[str, list[dict]]:
    context_parts = []
    sources = []
    seen_ids = set()

    for chunk in chunks:
        if chunk.id in seen_ids:
            continue
        seen_ids.add(chunk.id)

        meta = chunk.metadata
        doc_name = meta.get("filename", meta.get("source", "Unknown Document"))
        page = meta.get("page_number", "N/A")

        context_parts.append(
            f"[Source: {doc_name}, Page {page}]\n{chunk.content}"
        )
        sources.append({
            "document": doc_name,
            "page": page,
            "chunk_id": chunk.id,
            "score": round(chunk.score, 4),
            "document_id": meta.get("document_id", "")
        })

    return "\n\n---\n\n".join(context_parts), sources

async def run_rag_agent(state: AgentState) -> AgentState:
    start_time = time.time()
    db = state.get("db")
    user_id = state["user_id"]
    query = state["query"]

    try:
        # 1. Hybrid retrieval
        retriever = HybridRetriever(db)
        filters = {"user_id": user_id}
        if state.get("document_ids"):
            filters["document_id"] = state["document_ids"][0]

        retrieved = await retriever.retrieve(
            query=query,
            top_k_semantic=settings.TOP_K_RETRIEVAL,
            top_k_keyword=settings.TOP_K_RETRIEVAL,
            filters=filters,
            final_top_k=settings.TOP_K_RETRIEVAL
        )

        # 2. Reranking
        reranker = get_reranker()
        reranked = await reranker.rerank(query, retrieved, top_k=settings.TOP_K_RERANK)

        if not reranked:
            return {
                **state,
                "retrieved_chunks": retrieved,
                "reranked_chunks": [],
                "answer": "I could not find relevant information in your documents to answer this question.",
                "sources": [],
                "confidence": 0.0,
                "metadata": {"latency": time.time() - start_time}
            }

        # 3. Build context
        context, sources = _build_context(reranked)

        # 4. Load prompts
        system_prompt = prompt_manager.get_prompt("rag", "system")
        answer_prompt = prompt_manager.render("rag", "answer", context=context, question=query)

        # 5. Call LLM
        provider = LLMProviderFactory.get_provider()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": answer_prompt}
        ]

        answer = await provider.chat(messages)
        latency = time.time() - start_time

        # 6. Confidence based on top chunk score
        confidence = min(reranked[0].score if reranked else 0.0, 1.0)

        logger.info("rag_agent_complete", latency=latency, sources=len(sources), chunks=len(reranked))

        return {
            **state,
            "retrieved_chunks": retrieved,
            "reranked_chunks": reranked,
            "context": context,
            "answer": answer,
            "sources": sources,
            "confidence": round(confidence, 4),
            "metadata": {
                "latency": round(latency, 3),
                "retrieved_count": len(retrieved),
                "reranked_count": len(reranked),
                "model": provider.get_model_name()
            }
        }
    except Exception as e:
        logger.error("rag_agent_failed", error=str(e))
        return {
            **state,
            "answer": "An error occurred while processing your question.",
            "sources": [],
            "confidence": 0.0,
            "error": str(e),
            "metadata": {"latency": time.time() - start_time}
        }
