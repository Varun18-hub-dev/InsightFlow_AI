"""Integration tests for the complete evaluation and experiment execution pipelines.

Tests that _run_evaluation_task and _run_experiment_task execute end-to-end
with real dataset, real metrics, and proper status transitions to COMPLETED.
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.evaluations import _run_evaluation_task
from app.api.experiments import _run_experiment_task
from app.db.models import EvaluationRun, ExperimentRun
from app.retrieval.base_retriever import RetrievedChunk


@pytest.mark.asyncio
async def test_run_evaluation_task_end_to_end():
    """Verify _run_evaluation_task runs real dataset questions, computes metrics, and sets COMPLETED."""
    run_id = uuid.uuid4()
    user_id = uuid.uuid4()

    fake_run = EvaluationRun(
        id=run_id,
        user_id=user_id,
        name="test_benchmark_e2e",
        status="pending",
        config={"chunk_size": 800},
        results=None,
        started_at=datetime.utcnow(),
    )

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = fake_run
    mock_db.execute.return_value = mock_result

    # Mock AsyncSessionLocal context manager in evaluations module
    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    fake_graph_state = {
        "answer": "InsightFlow AI is an enterprise RAG knowledge assistant.",
        "sources": [{"document": "sample_eval.json", "page": 1}],
        "context": "InsightFlow AI is an enterprise RAG knowledge assistant providing search.",
        "confidence": 0.92,
    }

    with (
        patch("app.api.evaluations.AsyncSessionLocal", mock_session_factory),
        patch("app.agents.graph.run_graph", new=AsyncMock(return_value=fake_graph_state)),
    ):
        await _run_evaluation_task(str(run_id), str(user_id))

    # Verify run completed successfully without errors
    assert fake_run.status == "completed", f"Run failed with: {fake_run.results}"
    assert fake_run.results is not None
    assert "metrics" in fake_run.results
    assert "per_question" in fake_run.results

    metrics = fake_run.results["metrics"]
    assert isinstance(metrics["recall_at_5"], float)
    assert isinstance(metrics["hit_rate_at_5"], float)
    assert isinstance(metrics["mrr"], float)
    assert isinstance(metrics["faithfulness"], float)
    assert isinstance(metrics["answer_relevance"], float)
    assert metrics["questions_evaluated"] > 0
    assert metrics["total_latency_seconds"] >= 0.0

    # Ensure no TypeError or exception in results
    assert "error" not in fake_run.results


@pytest.mark.asyncio
async def test_run_experiment_task_end_to_end():
    """Verify _run_experiment_task executes with custom hyperparameters and sets COMPLETED."""
    run_id = uuid.uuid4()
    user_id = uuid.uuid4()

    fake_run = ExperimentRun(
        id=run_id,
        user_id=user_id,
        name="test_experiment_e2e",
        status="pending",
        config={
            "chunk_size": 800,
            "chunk_overlap": 150,
            "top_k_retrieval": 10,
            "top_k_rerank": 5,
            "reranker_type": "lightweight",
            "model": "gemini-flash-lite-latest",
        },
        results=None,
        error_message=None,
        created_at=datetime.utcnow(),
    )

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = fake_run
    mock_db.execute.return_value = mock_result

    # Mock AsyncSessionLocal in experiments module
    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_db

    # Mock retriever chunks
    sample_retrieved = [
        RetrievedChunk(id="c1", content="Financial reports for Q4 show steady growth.", score=0.88, metadata={"filename": "annual_report.pdf", "page_number": 2}),
    ]

    mock_retriever = AsyncMock()
    mock_retriever.retrieve = AsyncMock(return_value=sample_retrieved)

    mock_provider = AsyncMock()
    mock_provider.chat = AsyncMock(return_value="The revenue increased during Q4 based on the annual report.")

    with (
        patch("app.api.experiments.AsyncSessionLocal", mock_session_factory),
        patch("app.api.experiments.HybridRetriever", return_value=mock_retriever),
        patch("app.api.experiments.LLMProviderFactory.get_provider", return_value=mock_provider),
    ):
        await _run_experiment_task(str(run_id), str(user_id))

    assert fake_run.status == "completed", f"Experiment failed with error: {fake_run.error_message}"
    assert fake_run.error_message is None
    assert fake_run.results is not None

    metrics = fake_run.results["metrics"]
    assert isinstance(metrics["recall_at_5"], float)
    assert isinstance(metrics["faithfulness"], float)
    assert isinstance(metrics["answer_relevance"], float)
    assert metrics["questions_evaluated"] > 0
    assert fake_run.completed_at is not None
