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


def test_refusal_detector_matches_app_prompt_phrases_and_not_answers():
    from benchmarks.run_abstention_benchmark import is_refusal, strip_citations

    assert is_refusal("The requested information could not be found in the provided documents.")
    assert is_refusal("I could not find sufficient information in the provided documents.")
    assert is_refusal("The context does not mention who presided over the assembly.")
    assert not is_refusal("Denver Broncos (Source: Super Bowl 50, Page 1)")
    assert not is_refusal("It was finished in 1889.")
    assert strip_citations("Denver Broncos (Source: Super Bowl 50, Page 1)") == "Denver Broncos"


def test_auroc_extremes():
    from benchmarks.run_abstention_benchmark import auroc

    assert auroc([2, 3], [0, 1]) == 1.0
    assert auroc([1, 1], [1, 1]) == 0.5
    assert auroc([0], [1]) == 0.0


def test_load_squad2_balances_answerable_and_unanswerable(tmp_path):
    from benchmarks.run_abstention_benchmark import load_squad2

    data = {"data": [{"title": "T", "paragraphs": [{"context": "The tower was built in 1889.", "qas": [
        {"id": "a", "question": "When was it built?", "answers": [{"text": "1889"}], "is_impossible": False},
        {"id": "u", "question": "When was it demolished?", "answers": [], "is_impossible": True}]}]}]}
    p = tmp_path / "s2.json"
    p.write_text(json.dumps(data))
    corpus, queries = load_squad2(p, None, 0)
    assert len(corpus) == 1
    assert sorted(q["unanswerable"] for q in queries) == [False, True]


def test_strict_prompt_marker_counts_as_refusal_and_prompt_set_exists():
    from pathlib import Path

    from app.core.prompt_manager import PromptManager
    from benchmarks.run_abstention_benchmark import is_refusal

    assert is_refusal("NOT FOUND: the context gives 1777, not 1774.")
    assert is_refusal("  not found: no current chairman is named.")
    pm = PromptManager(Path(__file__).resolve().parents[2] / "benchmarks" / "prompts")
    rendered = pm.render("rag_strict", "answer", context="CTX", question="Q?")
    assert "CTX" in rendered and "Q?" in rendered and "NOT FOUND:" in rendered
    assert pm.get_prompt("rag_strict", "system")
