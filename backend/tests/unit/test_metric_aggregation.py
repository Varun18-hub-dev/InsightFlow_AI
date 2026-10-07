"""Regression tests for MetricResult type boundary, aggregation, and JSON serialization.

Verifies that MetricResult objects are properly extracted, averaged, and serialized
without raising: TypeError: unsupported operand type(s) for +: 'int' and 'MetricResult'.
"""

import json

from app.evaluation.eval_service import (
    EvaluationMetricsSuite,
    compute_average,
    extract_metric_value,
)
from app.evaluation.metrics.base import MetricResult
from app.evaluation.metrics.generation_metrics import (
    AnswerRelevanceMetric,
    FaithfulnessMetric,
)
from app.evaluation.metrics.retrieval_metrics import (
    HitRate,
    MeanReciprocalRank,
    RecallAtK,
)


def test_metric_result_arithmetic_and_float_conversion():
    """Verify MetricResult can participate in arithmetic with integers and floats."""
    m1 = MetricResult(name="Recall@5", value=0.8, description="Recall")
    m2 = MetricResult(name="Recall@5", value=0.6, description="Recall")

    # sum() in Python starts with integer 0: 0 + m1 + m2
    total = sum([m1, m2])
    assert round(total, 4) == 1.4

    # Float conversion
    assert float(m1) == 0.8
    assert int(MetricResult("Hit", 1.0, "Hit")) == 1

    # extract_metric_value
    assert extract_metric_value(m1) == 0.8
    assert extract_metric_value(0.75) == 0.75


def test_compute_average_with_metric_results():
    """Verify compute_average calculates the mean of MetricResult objects without TypeError."""
    m1 = MetricResult(name="Recall", value=1.0, description="desc")
    m2 = MetricResult(name="Recall", value=0.5, description="desc")
    m3 = MetricResult(name="Recall", value=0.0, description="desc")

    avg = compute_average([m1, m2, m3])
    assert avg == 0.5


def test_real_metric_computation_and_aggregation():
    """Verify real metrics return MetricResult and aggregate into pure numeric floats."""
    recall_fn = RecallAtK()
    hit_fn = HitRate()
    mrr_fn = MeanReciprocalRank()
    faith_fn = FaithfulnessMetric()
    rel_fn = AnswerRelevanceMetric()

    # Question 1: perfect hit
    r1 = recall_fn.compute(["doc1.pdf", "doc2.pdf"], ["doc1.pdf"], k=5)
    h1 = hit_fn.compute(["doc1.pdf", "doc2.pdf"], ["doc1.pdf"], k=5)
    m1 = mrr_fn.compute(["doc1.pdf", "doc2.pdf"], ["doc1.pdf"])
    f1 = faith_fn.compute("The revenue is $5M in Q4.", "In Q4, the company generated revenue of $5M.")
    a1 = rel_fn.compute("What was the Q4 revenue?", "The revenue was $5M in Q4.")

    assert isinstance(r1, MetricResult)
    assert isinstance(h1, MetricResult)
    assert isinstance(m1, MetricResult)
    assert isinstance(f1, MetricResult)
    assert isinstance(a1, MetricResult)

    # Question 2: miss
    r2 = recall_fn.compute(["other.pdf"], ["doc1.pdf"], k=5)
    h2 = hit_fn.compute(["other.pdf"], ["doc1.pdf"], k=5)
    m2 = mrr_fn.compute(["other.pdf"], ["doc1.pdf"])
    f2 = faith_fn.compute("Unrelated answer.", "Context discusses cloud architecture.")
    a2 = rel_fn.compute("What was the Q4 revenue?", "Cloud servers were configured.")

    # Aggregation using compute_average
    avg_recall = compute_average([r1, r2])
    avg_hit = compute_average([h1, h2])
    avg_mrr = compute_average([m1, m2])
    avg_faith = compute_average([f1, f2])
    avg_rel = compute_average([a1, a2])

    assert isinstance(avg_recall, float)
    assert avg_recall == 0.5
    assert avg_hit == 0.5
    assert avg_mrr == 0.5
    assert 0.0 <= avg_faith <= 1.0
    assert 0.0 <= avg_rel <= 1.0


def test_evaluation_metrics_suite_end_to_end():
    """Verify EvaluationMetricsSuite evaluates QA items and produces JSON-serializable results."""
    suite = EvaluationMetricsSuite()

    q1_scores, q1_details = suite.evaluate_qa_item(
        question="What is the refund policy?",
        expected_sources=["policy.pdf"],
        retrieved_sources=["policy.pdf", "faq.pdf"],
        answer="Refunds are processed within 30 days according to the policy.",
        context="The refund policy states all refunds are processed within 30 days.",
    )

    q2_scores, q2_details = suite.evaluate_qa_item(
        question="What is the billing cycle?",
        expected_sources=["billing.pdf"],
        retrieved_sources=["faq.pdf"],
        answer="I could not find sufficient information in the provided documents.",
        context="FAQ covers password resets.",
    )

    # All individual scores must be pure floats
    for k, val in q1_scores.items():
        assert isinstance(val, float), f"{k} was not float: {type(val)}"

    # Batch aggregation
    aggregated = suite.aggregate_batch([q1_scores, q2_scores], total_latency_seconds=4.52)

    assert isinstance(aggregated["recall_at_5"], float)
    assert isinstance(aggregated["faithfulness"], float)
    assert isinstance(aggregated["answer_relevance"], float)
    assert aggregated["questions_evaluated"] == 2
    assert aggregated["total_latency_seconds"] == 4.52

    # Verify JSON serializability for database persistence
    payload = {
        "metrics": aggregated,
        "per_question": [
            {"question": "Q1", "metrics": q1_scores, "details": q1_details},
            {"question": "Q2", "metrics": q2_scores, "details": q2_details},
        ],
    }
    serialized = json.dumps(payload)
    assert "recall_at_5" in serialized
    deserialized = json.loads(serialized)
    assert deserialized["metrics"]["questions_evaluated"] == 2
