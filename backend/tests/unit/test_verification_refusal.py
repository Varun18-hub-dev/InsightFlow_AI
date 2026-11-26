"""The grounding check must treat the answer prompt's "NOT FOUND:" marker as a refusal."""
from app.agents.nodes.verification import run_verification


def test_not_found_marker_lowers_confidence():
    out = run_verification({"answer": "NOT FOUND: the context gives 1777, not 1774.", "sources": ["s"], "confidence": 0.9})
    assert out["confidence"] == 0.2


def test_normal_answer_keeps_confidence():
    out = run_verification({"answer": "It was finished in 1889 (Source: Eiffel, Page 1).", "sources": ["s"], "confidence": 0.9})
    assert out["confidence"] == 0.9
