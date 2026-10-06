"""Unit tests for LightweightReranker."""
import pytest
from app.retrieval.base_retriever import RetrievedChunk
from app.reranking.lightweight_reranker import LightweightReranker


def make_chunk(id: str, content: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(id=id, content=content, score=score, metadata={"filename": "doc.pdf"})


@pytest.mark.asyncio
async def test_returns_top_k():
    reranker = LightweightReranker()
    chunks = [make_chunk(str(i), f"content {i}") for i in range(10)]
    result = await reranker.rerank("test query", chunks, top_k=3)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_scores_in_range():
    reranker = LightweightReranker()
    chunks = [make_chunk(str(i), "some content about the topic") for i in range(5)]
    result = await reranker.rerank("content topic", chunks, top_k=5)
    for c in result:
        assert 0.0 <= c.score <= 1.0


@pytest.mark.asyncio
async def test_empty_input_returns_empty():
    reranker = LightweightReranker()
    result = await reranker.rerank("query", [], top_k=5)
    assert result == []


@pytest.mark.asyncio
async def test_relevant_chunk_ranks_higher():
    reranker = LightweightReranker()
    relevant = make_chunk("rel", "annual revenue financial report 2024 earnings", score=0.8)
    irrelevant = make_chunk("irr", "the weather today is sunny and pleasant", score=0.3)
    result = await reranker.rerank("annual revenue report", [irrelevant, relevant], top_k=2)
    assert result[0].id == "rel"


@pytest.mark.asyncio
async def test_top_k_larger_than_input():
    reranker = LightweightReranker()
    chunks = [make_chunk(str(i), "text") for i in range(3)]
    result = await reranker.rerank("query", chunks, top_k=10)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_rank_assigned_sequentially():
    reranker = LightweightReranker()
    chunks = [make_chunk(str(i), "content") for i in range(5)]
    result = await reranker.rerank("query", chunks, top_k=5)
    assert [c.rank for c in result] == list(range(5))
