"""Canonical evaluation service for calculating, extracting, and aggregating RAG metrics.

Ensures strict type safety between MetricResult objects, numeric averages, and JSON-serializable outputs.
"""

from typing import Any

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


def extract_metric_value(item: Any) -> float:
    """Extract numeric score from MetricResult, float, or int."""
    if isinstance(item, MetricResult):
        return float(item.value)
    if isinstance(item, (int, float)):
        return float(item)
    try:
        return float(item)
    except (TypeError, ValueError):
        return 0.0


def compute_average(items: list[Any]) -> float:
    """Safely compute average of MetricResult objects or numeric floats."""
    if not items:
        return 0.0
    values = [extract_metric_value(x) for x in items]
    return round(sum(values) / len(values), 4)


class EvaluationMetricsSuite:
    """Suite of standard RAG evaluation metrics."""

    def __init__(self):
        self.recall_metric = RecallAtK()
        self.hit_rate_metric = HitRate()
        self.mrr_metric = MeanReciprocalRank()
        self.faithfulness_metric = FaithfulnessMetric()
        self.relevance_metric = AnswerRelevanceMetric()

    def evaluate_qa_item(
        self,
        question: str,
        expected_sources: list[str],
        retrieved_sources: list[str],
        answer: str,
        context: str,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Compute all retrieval and generation metrics for a single question.

        Returns:
            numeric_scores: dictionary of metric names to pure float values.
            detailed_results: dictionary of metric names to serialized MetricResult dicts.
        """
        # Retrieval metrics
        recall_res = self.recall_metric.compute(retrieved_sources, expected_sources, k=5)
        hit_res = self.hit_rate_metric.compute(retrieved_sources, expected_sources, k=5)
        mrr_res = self.mrr_metric.compute(retrieved_sources, expected_sources, k=5)

        # Generation metrics
        faith_res = self.faithfulness_metric.compute(answer, context)
        rel_res = self.relevance_metric.compute(question, answer)

        numeric_scores = {
            "recall_at_5": float(recall_res.value),
            "hit_rate_at_5": float(hit_res.value),
            "mrr": float(mrr_res.value),
            "faithfulness": float(faith_res.value),
            "answer_relevance": float(rel_res.value),
        }

        detailed_results = {
            "recall_at_5": recall_res.to_dict(),
            "hit_rate_at_5": hit_res.to_dict(),
            "mrr": mrr_res.to_dict(),
            "faithfulness": faith_res.to_dict(),
            "answer_relevance": rel_res.to_dict(),
        }

        return numeric_scores, detailed_results

    def aggregate_batch(
        self,
        batch_scores: list[dict[str, float]],
        total_latency_seconds: float,
    ) -> dict[str, Any]:
        """Aggregate a batch of question-level numeric scores into overall benchmark metrics."""
        recalls = [s["recall_at_5"] for s in batch_scores if "recall_at_5" in s]
        hits = [s["hit_rate_at_5"] for s in batch_scores if "hit_rate_at_5" in s]
        mrrs = [s["mrr"] for s in batch_scores if "mrr" in s]
        faiths = [s["faithfulness"] for s in batch_scores if "faithfulness" in s]
        rels = [s["answer_relevance"] for s in batch_scores if "answer_relevance" in s]

        return {
            "recall_at_5": compute_average(recalls),
            "hit_rate_at_5": compute_average(hits),
            "mrr": compute_average(mrrs),
            "faithfulness": compute_average(faiths),
            "answer_relevance": compute_average(rels),
            "questions_evaluated": len(batch_scores),
            "total_latency_seconds": round(total_latency_seconds, 2),
        }
