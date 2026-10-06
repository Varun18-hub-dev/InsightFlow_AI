
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .base_retriever import BaseRetriever, RetrievedChunk

logger = structlog.get_logger()

class KeywordRetriever(BaseRetriever):
    def __init__(self, db: AsyncSession):
        self.db = db

    async def retrieve(self, query: str, top_k: int = 20, filters: dict | None = None) -> list[RetrievedChunk]:
        user_id = filters.get("user_id") if filters else None
        try:
            query_sql = """
                SELECT
                    id::text,
                    content,
                    ts_rank(to_tsvector('english', content), plainto_tsquery('english', :query)) AS score,
                    chunk_metadata,
                    document_id::text,
                    chunk_index
                FROM document_chunks
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query)
            """
            params = {"query": query, "top_k": top_k}
            if user_id:
                query_sql += " AND user_id::text = :user_id"
                params["user_id"] = str(user_id)
            query_sql += " ORDER BY score DESC LIMIT :top_k"

            result = await self.db.execute(text(query_sql), params)
            rows = result.fetchall()
            chunks = []
            for i, row in enumerate(rows):
                metadata = dict(row.chunk_metadata or {})
                metadata["document_id"] = row.document_id
                chunks.append(RetrievedChunk(
                    id=row.id,
                    content=row.content,
                    score=float(row.score),
                    metadata=metadata,
                    rank=i
                ))
            return chunks
        except Exception as e:
            logger.error("keyword_retrieval_failed", error=str(e))
            try:
                await self.db.rollback()
            except Exception:
                pass
            return []
