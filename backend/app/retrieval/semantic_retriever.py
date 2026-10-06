
import structlog

from app.llm.embedding_factory import EmbeddingProviderFactory
from app.services.pinecone_service import get_vector_store

from .base_retriever import BaseRetriever, RetrievedChunk

logger = structlog.get_logger()

class SemanticRetriever(BaseRetriever):
    def __init__(self):
        self.embedding_provider = None
        self.vector_store = None

    async def _ensure_initialized(self):
        if self.embedding_provider is None:
            self.embedding_provider = EmbeddingProviderFactory.get_provider()
        if self.vector_store is None:
            self.vector_store = get_vector_store()

    async def retrieve(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[RetrievedChunk]:
        await self._ensure_initialized()
        try:
            query_vector = await self.embedding_provider.embed_query(query)
            results = await self.vector_store.search(query_vector, top_k=top_k, filter=filters)
            chunks = []
            for i, r in enumerate(results):
                chunks.append(RetrievedChunk(
                    id=r["id"],
                    content=r["metadata"].get("content", ""),
                    score=r["score"],
                    metadata=r["metadata"],
                    rank=i
                ))
            return chunks
        except Exception as e:
            logger.error("semantic_retrieval_failed", error=str(e))
            return []
