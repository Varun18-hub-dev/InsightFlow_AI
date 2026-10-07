import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import structlog
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.base import AsyncSessionLocal, get_db
from app.db.models import EvaluationRun, User
from app.db.schemas import EvaluationRunResponse
from app.services.mlflow_service import get_mlflow_service

logger = structlog.get_logger()
router = APIRouter()


def _find_eval_dataset_path() -> Path:
    env_path = os.getenv("EVAL_DATASET_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    curr = Path(__file__).resolve()
    candidates = [
        curr.parent.parent / "evaluation" / "datasets" / "sample_eval.json",
        Path("/app/app/evaluation/datasets/sample_eval.json"),
        Path("/app/evaluation/datasets/sample_eval.json"),
        Path.cwd() / "app" / "evaluation" / "datasets" / "sample_eval.json",
        Path.cwd() / "evaluation" / "datasets" / "sample_eval.json",
        Path.cwd().parent / "evaluation" / "datasets" / "sample_eval.json",
    ]
    for n in (1, 2, 3, 4):
        if len(curr.parents) > n:
            candidates.append(curr.parents[n] / "evaluation" / "datasets" / "sample_eval.json")
            candidates.append(curr.parents[n] / "app" / "evaluation" / "datasets" / "sample_eval.json")
    for c in candidates:
        if c.exists():
            return c
    return curr.parent.parent / "evaluation" / "datasets" / "sample_eval.json"


EVAL_DATASET_PATH = _find_eval_dataset_path()


@router.get("/evaluations", response_model=list[EvaluationRunResponse])
async def list_evaluations(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all evaluation benchmark runs for the current authenticated user."""
    stmt = (
        select(EvaluationRun)
        .where(EvaluationRun.user_id == current_user.id)
        .order_by(EvaluationRun.started_at.desc())
    )
    result = await db.execute(stmt)
    runs = result.scalars().all()
    return runs


@router.get("/evaluations/{eval_id}", response_model=EvaluationRunResponse)
async def get_evaluation(
    eval_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get single evaluation run."""
    try:
        e_uuid = uuid.UUID(eval_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid evaluation ID") from None

    stmt = select(EvaluationRun).where(
        EvaluationRun.id == e_uuid,
        EvaluationRun.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run


async def _run_evaluation_task(run_id: str, user_id: str):
    """Background evaluation task."""
    run_uuid = uuid.UUID(run_id)
    user_uuid = uuid.UUID(user_id)

    async with AsyncSessionLocal() as db:
        stmt = select(EvaluationRun).where(
            EvaluationRun.id == run_uuid,
            EvaluationRun.user_id == user_uuid,
        )
        result = await db.execute(stmt)
        run = result.scalar_one_or_none()
        if not run:
            return

        run.status = "running"
        await db.commit()

        try:
            dataset_path = _find_eval_dataset_path()
            if not dataset_path.exists():
                run.status = "failed"
                run.results = {"error": f"Evaluation dataset not found at {dataset_path}"}
                run.completed_at = datetime.utcnow()
                await db.commit()
                return

            with open(dataset_path) as f:
                dataset = json.load(f)

            per_question = []
            start = time.time()

            from app.agents.graph import run_graph
            from app.evaluation.eval_service import EvaluationMetricsSuite

            suite = EvaluationMetricsSuite()
            question_scores = []

            for item in dataset[:5]:  # 5 questions for demo speed
                try:
                    state = await run_graph(
                        query=item["question"],
                        user_id=str(user_uuid),
                        db=db,
                    )
                    answer = state.get("answer", "")
                    retrieved_sources = [s.get("document", "") for s in state.get("sources", [])]
                    expected = item.get("expected_sources", [])
                    context = state.get("context", "")

                    numeric_scores, _detailed = suite.evaluate_qa_item(
                        question=item["question"],
                        expected_sources=expected,
                        retrieved_sources=retrieved_sources,
                        answer=answer,
                        context=context,
                    )
                    question_scores.append(numeric_scores)

                    per_question.append({
                        "question": item["question"],
                        "answer": answer[:200],
                        "sources": retrieved_sources,
                        "confidence": state.get("confidence", 0),
                        "metrics": numeric_scores,
                    })
                except Exception as e:
                    logger.warning("eval_question_failed", q=item["question"][:50], error=str(e))

            total_latency = time.time() - start
            metrics = suite.aggregate_batch(question_scores, total_latency)

            # Optional MLflow tracking
            mlflow_run_id = None
            try:
                mlflow_svc = get_mlflow_service()
                mlflow_run_id = await mlflow_svc.log_experiment_run(
                    experiment_name="insightflow-rag-evaluation",
                    params={
                        "model": settings.LLM_PROVIDER,
                        "chunk_size": settings.CHUNK_SIZE,
                        "top_k": settings.TOP_K_RERANK,
                        "reranker": settings.RERANKER_TYPE,
                    },
                    metrics=metrics,
                    tags={"run_name": run.name},
                )
            except Exception:
                pass

            run.status = "completed"
            run.results = {"metrics": metrics, "per_question": per_question}
            run.mlflow_run_id = mlflow_run_id
            run.completed_at = datetime.utcnow()
            await db.commit()
            logger.info("evaluation_complete", run_id=str(run.id), metrics=metrics)

        except Exception as e:
            logger.error("evaluation_failed", run_id=run_id, error=str(e))
            run.status = "failed"
            run.results = {"error": str(e)}
            run.completed_at = datetime.utcnow()
            await db.commit()


@router.post("/evaluations/run", response_model=EvaluationRunResponse)
async def run_evaluation(
    background_tasks: BackgroundTasks,
    request: dict = Body(default={}),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Trigger a new evaluation run."""
    run = EvaluationRun(
        user_id=current_user.id,
        name=request.get("name") or f"eval_{int(time.time())}",
        status="pending",
        config={
            "model": settings.LLM_PROVIDER,
            "chunk_size": settings.CHUNK_SIZE,
            "top_k": settings.TOP_K_RERANK,
            "reranker": settings.RERANKER_TYPE,
        },
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    background_tasks.add_task(_run_evaluation_task, str(run.id), str(current_user.id))
    logger.info("evaluation_started", run_id=str(run.id))
    return run
