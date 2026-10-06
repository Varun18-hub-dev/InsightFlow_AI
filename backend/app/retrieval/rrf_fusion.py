"""
Reciprocal Rank Fusion (RRF) implementation.
Pure function - no I/O, fully unit testable.
"""
from .base_retriever import RetrievedChunk


def rrf_fusion(
    result_lists: list[list[RetrievedChunk]],
    k: int = 60
) -> list[RetrievedChunk]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Score formula: sum over lists of 1 / (k + rank)
    where rank is 1-indexed position in each list.

    Args:
        result_lists: List of ranked result lists to fuse
        k: Constant to prevent high impact of top ranks (default 60)

    Returns:
        Combined, deduplicated list sorted by RRF score (descending)
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list, start=1):
            rrf_score = 1.0 / (k + rank)
            if chunk.id in scores:
                scores[chunk.id] += rrf_score
            else:
                scores[chunk.id] = rrf_score
                chunk_map[chunk.id] = chunk

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused = []
    for rank, chunk_id in enumerate(sorted_ids):
        chunk = chunk_map[chunk_id]
        fused.append(RetrievedChunk(
            id=chunk.id,
            content=chunk.content,
            score=scores[chunk_id],
            metadata=chunk.metadata,
            rank=rank
        ))
    return fused
