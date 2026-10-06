"""Unit tests for rule-based intent classifier."""
import pytest
from app.agents.nodes.intent_classifier import _rule_based_classify


def test_compare_keyword():
    assert _rule_based_classify("compare document A and document B") == "DOCUMENT_COMPARISON"


def test_vs_keyword():
    assert _rule_based_classify("Q4 report vs Q3 report differences") == "DOCUMENT_COMPARISON"


def test_difference_keyword():
    assert _rule_based_classify("what is the difference between these contracts") == "DOCUMENT_COMPARISON"


def test_summarize_keyword():
    assert _rule_based_classify("can you summarize this document for me") == "DOCUMENT_SUMMARY"


def test_summary_keyword():
    assert _rule_based_classify("give me a summary of the annual report") == "DOCUMENT_SUMMARY"


def test_how_many_keyword():
    assert _rule_based_classify("how many documents did I upload this month") == "DATABASE_QUERY"


def test_count_keyword():
    assert _rule_based_classify("count my uploaded files") == "DATABASE_QUERY"


def test_list_all_keyword():
    assert _rule_based_classify("list all documents uploaded this week") == "DATABASE_QUERY"


def test_general_question_returns_none():
    result = _rule_based_classify("what is machine learning")
    assert result is None


def test_document_question_returns_none():
    result = _rule_based_classify("what does the contract say about payment terms")
    assert result is None  # Falls through to LLM


def test_case_insensitive_comparison():
    assert _rule_based_classify("COMPARE the two documents") == "DOCUMENT_COMPARISON"
