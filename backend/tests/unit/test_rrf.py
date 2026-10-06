"""Unit tests for RRF fusion algorithm."""
import pytest
from app.retrieval.base_retriever import RetrievedChunk
from app.retrieval.rrf_fusion import rrf_fusion


def make_chunk(id: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(id=id, content=f"content_{id}", score=score, metadata={})


def test_single_list_preserves_order():
    chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
    result = rrf_fusion([chunks])
    assert [c.id for c in result] == ["a", "b", "c"]


def test_empty_input():
    assert rrf_fusion([]) == []


def test_empty_list_in_input():
    assert rrf_fusion([[]]) == []


def test_deduplication_boosts_shared_item():
    """Item appearing in both lists should rank higher than items in only one."""
    list1 = [make_chunk("a"), make_chunk("b")]
    list2 = [make_chunk("a"), make_chunk("c")]
    result = rrf_fusion([list1, list2])
    ids = [c.id for c in result]
    # "a" appears in both lists → should rank first
    assert ids[0] == "a"


def test_rrf_score_formula():
    """Score for a single item at rank 1 should be 1/(k+1)."""
    chunks = [make_chunk("x")]
    result = rrf_fusion([chunks], k=60)
    expected = 1.0 / (60 + 1)
    assert abs(result[0].score - expected) < 1e-9


def test_all_ids_present():
    """All unique IDs from input lists should appear in output."""
    list1 = [make_chunk("a"), make_chunk("b")]
    list2 = [make_chunk("c"), make_chunk("d")]
    result = rrf_fusion([list1, list2])
    ids = {c.id for c in result}
    assert ids == {"a", "b", "c", "d"}


def test_rank_is_assigned():
    """Output chunks should have rank 0, 1, 2, ..."""
    chunks = [make_chunk(str(i)) for i in range(5)]
    result = rrf_fusion([chunks])
    assert [c.rank for c in result] == list(range(5))
