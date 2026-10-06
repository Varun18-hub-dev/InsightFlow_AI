"""
Cross-encoder reranker using sentence-transformers.
Falls back to LightweightReranker if model unavailable.
"""
import asyncio
from functools import lru_cache

import structlog

from app.retrieval.base_retriever import RetrievedChunk

from .base_reranker import BaseReranker
from .lightweight_reranker import LightweightReranker

logger = structlog.get_logger()

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

@lru_cache(maxsize=1)
def _load_cross_encoder():
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(DEFAULT_MODEL)
        logger.info("cross_encoder_loaded", model=DEFAULT_MODEL)
        return model
    except Exception as e:
        logger.warning("cross_encoder_load_failed", error=str(e))
        return None

class CrossEncoderReranker(BaseReranker):
    def __init__(self):
        self._fallback = LightweightReranker()

    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        model = _load_cross_encoder()
        if model is None:
            logger.warning("falling_back_to_lightweight_reranker")
            return await self._fallback.rerank(query, chunks, top_k)

        try:
            pairs = [(query, chunk.content) for chunk in chunks]
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(None, lambda: model.predict(pairs))

            scored = []
            for chunk, score in zip(chunks, scores, strict=False):
                scored.append(RetrievedChunk(
                    id=chunk.id,
                    content=chunk.content,
                    score=float(score),
                    metadata=chunk.metadata,
                    rank=chunk.rank
                ))
            scored.sort(key=lambda x: x.score, reverse=True)
            for i, c in enumerate(scored):
                c.rank = i
            return scored[:top_k]
        except Exception as e:
            logger.error("cross_encoder_rerank_failed", error=str(e))
            return await self._fallback.rerank(query, chunks, top_k)
