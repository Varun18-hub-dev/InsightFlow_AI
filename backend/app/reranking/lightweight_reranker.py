"""
Lightweight reranker — no model loading required, runs on CPU.
Scores based on: retrieval score, lexical overlap, metadata relevance.
"""
import re

from app.retrieval.base_retriever import RetrievedChunk

from .base_reranker import BaseReranker


def _lexical_overlap(query: str, text: str) -> float:
    query_words = set(re.findall(r'\w+', query.lower()))
    text_words = set(re.findall(r'\w+', text.lower()))
    if not query_words:
        return 0.0
    return len(query_words & text_words) / len(query_words)

def _normalize_scores(chunks: list[RetrievedChunk]) -> dict[str, float]:
    if not chunks:
        return {}
    scores = [c.score for c in chunks]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return {c.id: 1.0 for c in chunks}
    return {c.id: (c.score - min_s) / (max_s - min_s) for c in chunks}

class LightweightReranker(BaseReranker):
    RETRIEVAL_WEIGHT = 0.4
    LEXICAL_WEIGHT = 0.35
    METADATA_WEIGHT = 0.25

    async def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []

        normalized = _normalize_scores(chunks)
        query_lower = query.lower()

        scored = []
        for chunk in chunks:
            retrieval_score = normalized.get(chunk.id, 0.0)
            lexical_score = _lexical_overlap(query, chunk.content)

            # Metadata relevance: boost if source filename contains query keywords
            meta_score = 0.0
            source = chunk.metadata.get("source", chunk.metadata.get("filename", ""))
            query_words = set(re.findall(r'\w+', query_lower))
            if any(w in source.lower() for w in query_words if len(w) > 3):
                meta_score = 1.0

            combined = (
                self.RETRIEVAL_WEIGHT * retrieval_score +
                self.LEXICAL_WEIGHT * lexical_score +
                self.METADATA_WEIGHT * meta_score
            )

            scored.append(RetrievedChunk(
                id=chunk.id,
                content=chunk.content,
                score=combined,
                metadata=chunk.metadata,
                rank=chunk.rank
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        for i, c in enumerate(scored):
            c.rank = i
        return scored[:top_k]
