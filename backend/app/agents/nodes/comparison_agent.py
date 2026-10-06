import time

import structlog
from sqlalchemy import select

from app.agents.state import AgentState
from app.core.prompt_manager import prompt_manager
from app.db.models import Document
from app.llm.factory import LLMProviderFactory
from app.retrieval.semantic_retriever import SemanticRetriever

logger = structlog.get_logger()

MAX_CONTEXT_PER_DOC = 6000

async def run_comparison_agent(state: AgentState) -> AgentState:
    start_time = time.time()
    db = state.get("db")
    doc_ids = state.get("document_ids", [])
    query = state.get("query", "Compare these documents")

    if len(doc_ids) < 2:
        return {**state, "answer": "Please provide two documents to compare.", "sources": []}

    try:
        retriever = SemanticRetriever()

        # Get doc names
        doc_names = []
        for doc_id in doc_ids[:2]:
            stmt = select(Document).where(Document.id == doc_id)
            result = await db.execute(stmt)
            doc = result.scalar_one_or_none()
            doc_names.append(doc.filename if doc else f"Document {doc_id[:8]}")

        # Retrieve relevant sections per document
        contents = []
        for doc_id in doc_ids[:2]:
            chunks = await retriever.retrieve(query, top_k=10, filters={"document_id": doc_id})
            content = "\n\n".join(c.content for c in chunks)
            if len(content) > MAX_CONTEXT_PER_DOC:
                content = content[:MAX_CONTEXT_PER_DOC] + "\n...[truncated]"
            contents.append(content)

        provider = LLMProviderFactory.get_provider()
        comparison_prompt = prompt_manager.render(
            "comparison", "comparison",
            doc_a_name=doc_names[0],
            content_a=contents[0] or "No content found",
            doc_b_name=doc_names[1],
            content_b=contents[1] or "No content found",
            query=query
        )

        answer = await provider.chat([{"role": "user", "content": comparison_prompt}])

        sources = [
            {"document": doc_names[0], "document_id": doc_ids[0]},
            {"document": doc_names[1], "document_id": doc_ids[1]},
        ]

        return {
            **state,
            "answer": answer,
            "sources": sources,
            "confidence": 0.85,
            "metadata": {"latency": time.time() - start_time}
        }
    except Exception as e:
        logger.error("comparison_agent_failed", error=str(e))
        return {**state, "answer": "Failed to compare documents.", "sources": [], "error": str(e)}
