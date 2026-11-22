"""Tests for the offline benchmark metrics and data plumbing (no models, no network)."""
import json
import math

import pytest

from benchmarks import datasets, metrics, retrievers


def test_recall_mrr_ndcg_on_known_rankings():
    ranked = ["d3", "d1", "d7"]
    rel = {"d1"}
    assert metrics.recall_at_k(ranked, rel, 1) == 0.0
    assert metrics.recall_at_k(ranked, rel, 2) == 1.0
    assert metrics.mrr_at_k(ranked, rel) == 0.5
    assert metrics.ndcg_at_k(ranked, rel) == pytest.approx(1 / math.log2(3))
    assert metrics.ndcg_at_k(["d1"], rel) == 1.0


def test_recall_refuses_queries_without_ground_truth():
    # The old app metric returned 1.0 here, which inflated averages.
    with pytest.raises(ValueError):
        metrics.recall_at_k(["d1"], set(), 5)


def test_answer_in_context_uses_squad_normalisation():
    assert metrics.answer_in_context(["It was built in 1889 by Eiffel."], ["1889"]) == 1.0
    assert metrics.answer_in_context(["The Denver Broncos won."], ["Denver Broncos"]) == 1.0
    assert metrics.answer_in_context(["Nothing relevant."], ["Paris"]) == 0.0
    assert math.isnan(metrics.answer_in_context(["x"], []))


def test_bootstrap_ci_brackets_the_mean():
    vals = [1.0] * 70 + [0.0] * 30
    mean, lo, hi = metrics.bootstrap_ci(vals)
    assert mean == pytest.approx(0.7)
    assert lo < 0.7 < hi
    assert hi - lo < 0.25


def test_paired_bootstrap_detects_a_real_difference_and_not_a_fake_one():
    a = [0.0] * 50 + [1.0] * 50
    better = [1.0] * 90 + [0.0] * 10
    assert metrics.paired_bootstrap(a, better)["p_value"] < 0.01
    same = metrics.paired_bootstrap(a, list(a))
    assert same["delta"] == 0.0 and same["p_value"] == 1.0


def test_load_squad_samples_one_question_per_paragraph(tmp_path):
    data = {"data": [{"title": "Eiffel_Tower", "paragraphs": [
        {"context": "The tower was finished in 1889.", "qas": [
            {"id": "q1", "question": "When was it finished?", "answers": [{"text": "1889"}]},
            {"id": "q2", "question": "What was finished?", "answers": [{"text": "The tower"}]}]},
        {"context": "It is in Paris.", "qas": [
            {"id": "q3", "question": "Where is it?", "answers": [{"text": "Paris"}]}]},
    ]}]}
    p = tmp_path / "squad.json"
    p.write_text(json.dumps(data))
    corpus, queries = datasets.load_squad(p, n_queries=None)
    assert len(corpus) == 2 and len(queries) == 2
    assert {q["group"] for q in queries} == {"Eiffel Tower"}
    for q in queries:
        assert len(q["relevant"]) == 1 and next(iter(q["relevant"])) in corpus


def test_chunks_collapse_to_unique_documents_in_rank_order():
    corpus = {"a": {"title": "A", "text": "alpha " * 400, "group": "A"},
              "b": {"title": "B", "text": "beta gamma", "group": "B"}}
    chunks = retrievers.build_chunks(corpus)
    assert len([c for c in chunks if c.metadata["doc_id"] == "a"]) > 1
    order = [chunks[1], chunks[0], chunks[-1]]
    assert retrievers.collapse_to_docs(order) == ["a", "b"]


def test_bm25_finds_the_obvious_document():
    corpus = {"x": {"title": "X", "text": "The Eiffel Tower is in Paris.", "group": "X"},
              "y": {"title": "Y", "text": "Photosynthesis happens in chloroplasts.", "group": "Y"}}
    idx = retrievers.BM25Index(retrievers.build_chunks(corpus))
    assert retrievers.collapse_to_docs(idx.search("where is the eiffel tower", 2))[0] == "x"
