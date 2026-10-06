import asyncio

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from .base_retriever import RetrievedChunk
from .keyword_retriever import KeywordRetriever
from .rrf_fusion import rrf_fusion
from .semantic_retriever import SemanticRetriever

logger = structlog.get_logger()

class HybridRetriever:
    def __init__(self, db: AsyncSession):
        self.semantic = SemanticRetriever()
        self.keyword = KeywordRetriever(db)

    async def retrieve(
        self,
        query: str,
        top_k_semantic: int = 20,
        top_k_keyword: int = 20,
        filters: dict | None = None,
        final_top_k: int = 20
    ) -> list[RetrievedChunk]:
        """Run semantic and keyword retrieval concurrently, fuse with RRF."""
        try:
            semantic_results, keyword_results = await asyncio.gather(
                self.semantic.retrieve(query, top_k=top_k_semantic, filters=filters),
                self.keyword.retrieve(query, top_k=top_k_keyword, filters=filters),
                return_exceptions=True
            )

            lists = []
            if isinstance(semantic_results, list):
                lists.append(semantic_results)
            else:
                logger.warning("semantic_retrieval_error", error=str(semantic_results))
                lists.append([])

            if isinstance(keyword_results, list):
                lists.append(keyword_results)
            else:
                logger.warning("keyword_retrieval_error", error=str(keyword_results))
                lists.append([])

            fused = rrf_fusion(lists)
            return fused[:final_top_k]
        except Exception as e:
            logger.error("hybrid_retrieval_failed", error=str(e))
            return []
