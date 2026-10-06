from typing import Any, TypedDict

from app.retrieval.base_retriever import RetrievedChunk


class AgentState(TypedDict, total=False):
    query: str
    user_id: str
    conversation_id: str | None
    intent: str | None
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    context: str
    messages: list[dict]
    answer: str | None
    sources: list[dict]
    confidence: float
    error: str | None
    metadata: dict
    document_ids: list[str]
    sql_query: str | None
    sql_result: list[dict] | None
    db: Any  # AsyncSession — passed through graph
