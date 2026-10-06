import time

import structlog
from sqlalchemy import select

from app.agents.state import AgentState
from app.core.prompt_manager import prompt_manager
from app.db.models import DocumentChunk
from app.llm.factory import LLMProviderFactory

logger = structlog.get_logger()

MAX_CONTEXT_CHARS = 12000

async def run_summary_agent(state: AgentState) -> AgentState:
    start_time = time.time()
    db = state.get("db")
    doc_ids = state.get("document_ids", [])

    if not doc_ids:
        return {**state, "answer": "No document specified for summarization.", "sources": []}

    try:
        doc_id = doc_ids[0]

        # Load chunks from DB
        stmt = select(DocumentChunk).where(
            DocumentChunk.document_id == doc_id
        ).order_by(DocumentChunk.chunk_index).limit(30)

        result = await db.execute(stmt)
        chunks = result.scalars().all()

        if not chunks:
            return {**state, "answer": "No content found for this document.", "sources": []}

        content = "\n\n".join(c.content for c in chunks)
        if len(content) > MAX_CONTEXT_CHARS:
            content = content[:MAX_CONTEXT_CHARS] + "\n...[truncated]"

        filename = chunks[0].chunk_metadata.get("filename", "document") if chunks[0].chunk_metadata else "document"

        provider = LLMProviderFactory.get_provider()
        summary_prompt = prompt_manager.render("summarization", "summary", content=content)

        answer = await provider.chat([{"role": "user", "content": summary_prompt}])

        return {
            **state,
            "answer": answer,
            "sources": [{"document": filename, "document_id": doc_id}],
            "confidence": 0.9,
            "metadata": {"latency": time.time() - start_time, "chunk_count": len(chunks)}
        }
    except Exception as e:
        logger.error("summary_agent_failed", error=str(e))
        return {**state, "answer": "Failed to generate document summary.", "sources": [], "error": str(e)}
