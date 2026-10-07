"""Unit tests for PostgreSQL ExperimentRun model, schemas, and API endpoints."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from app.api.evaluations import _find_eval_dataset_path
from app.api.experiments import (
    delete_experiment,
    get_experiment,
    list_experiments,
)
from app.db.models import ExperimentRun, User
from app.db.schemas import ExperimentRunRequest, ExperimentRunResponse


def test_eval_dataset_path_resolves():
    """Verify _find_eval_dataset_path finds the bundled sample_eval.json."""
    dataset_path = _find_eval_dataset_path()
    assert dataset_path.exists(), f"Evaluation dataset not found at {dataset_path}"
    assert dataset_path.name == "sample_eval.json"


def test_experiment_run_schema():
    """Verify ExperimentRunRequest and ExperimentRunResponse schema validation."""
    req = ExperimentRunRequest(
        name="hybrid_rrf_test",
        chunk_size=600,
        chunk_overlap=100,
        top_k_retrieval=15,
        top_k_rerank=5,
        reranker_type="lightweight",
    )
    assert req.name == "hybrid_rrf_test"
    assert req.chunk_size == 600
    assert req.reranker_type == "lightweight"

    run_id = uuid.uuid4()
    resp = ExperimentRunResponse(
        id=run_id,
        name=req.name or "test_run",
        status="completed",
        config=req.model_dump(),
        results={"metrics": {"recall_at_5": 0.85, "hit_rate": 1.0}},
        created_at=datetime.utcnow(),
    )
    assert resp.id == run_id
    assert resp.status == "completed"
    assert resp.results["metrics"]["recall_at_5"] == 0.85


@pytest.mark.asyncio
async def test_list_experiment_runs():
    """Verify list_experiments queries runs for the authenticated user."""
    user_id = uuid.uuid4()
    user = User(id=user_id, email="tester@example.com")

    run1 = ExperimentRun(
        id=uuid.uuid4(),
        user_id=user_id,
        name="run_1",
        status="completed",
        config={"chunk_size": 800},
        results={"metrics": {"recall_at_5": 0.9}},
        created_at=datetime.utcnow(),
    )

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [run1]
    mock_db.execute.return_value = mock_result

    runs = await list_experiments(db=mock_db, current_user=user)
    assert len(runs) == 1
    assert runs[0].name == "run_1"
    assert runs[0].status == "completed"


@pytest.mark.asyncio
async def test_get_experiment_run_found():
    """Verify get_experiment retrieves the single experiment run."""
    user_id = uuid.uuid4()
    run_id = uuid.uuid4()
    user = User(id=user_id, email="tester@example.com")

    run = ExperimentRun(
        id=run_id,
        user_id=user_id,
        name="run_detail",
        status="completed",
        config={"chunk_size": 500},
        results={"metrics": {"recall_at_5": 0.8}},
        created_at=datetime.utcnow(),
    )

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = run
    mock_db.execute.return_value = mock_result

    data = await get_experiment(experiment_id=str(run_id), db=mock_db, current_user=user)
    assert data.id == run_id
    assert data.name == "run_detail"


@pytest.mark.asyncio
async def test_get_experiment_run_not_found():
    """Verify get_experiment raises 404 when not found."""
    user_id = uuid.uuid4()
    run_id = uuid.uuid4()
    user = User(id=user_id, email="tester@example.com")

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    with pytest.raises(HTTPException) as exc_info:
        await get_experiment(experiment_id=str(run_id), db=mock_db, current_user=user)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_experiment_run():
    """Verify delete_experiment removes the record."""
    user_id = uuid.uuid4()
    run_id = uuid.uuid4()
    user = User(id=user_id, email="tester@example.com")

    run = ExperimentRun(
        id=run_id,
        user_id=user_id,
        name="run_delete",
        status="completed",
        config={},
        created_at=datetime.utcnow(),
    )

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = run
    mock_db.execute.return_value = mock_result

    res = await delete_experiment(experiment_id=str(run_id), db=mock_db, current_user=user)
    assert res == {"message": "Experiment run deleted"}
    mock_db.delete.assert_called_once_with(run)
    mock_db.commit.assert_awaited_once()
