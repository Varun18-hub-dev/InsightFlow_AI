import json
import os
import time
from pathlib import Path

import structlog
from fastapi import APIRouter, BackgroundTasks, Body, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.base import get_db
from app.db.models import EvaluationRun, User
from app.services.mlflow_service import get_mlflow_service

logger = structlog.get_logger()
router = APIRouter()

def _find_eval_dataset_path() -> Path:
    env_path = os.getenv("EVAL_DATASET_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    curr = Path(__file__).resolve()
    candidates = [
        Path("/app/evaluation/datasets/sample_eval.json"),
        Path.cwd() / "evaluation" / "datasets" / "sample_eval.json",
        Path.cwd().parent / "evaluation" / "datasets" / "sample_eval.json",
    ]
    for n in (3, 2, 4):
        if len(curr.parents) > n:
            candidates.append(curr.parents[n] / "evaluation" / "datasets" / "sample_eval.json")
    for c in candidates:
        if c.exists():
            return c
    return Path("/app/evaluation/datasets/sample_eval.json")

EVAL_DATASET_PATH = _find_eval_dataset_path()


@router.get("/evaluations")
async def list_evaluations(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    stmt = select(EvaluationRun).where(EvaluationRun.user_id == current_user.id).order_by(EvaluationRun.started_at.desc())
    result = await db.execute(stmt)
    runs = result.scalars().all()
    return [
        {
            "id": str(r.id),
            "name": r.name,
            "status": r.status,
            "config": r.config,
            "results": r.results,
            "mlflow_run_id": r.mlflow_run_id,
            "started_at": str(r.started_at),
            "completed_at": str(r.completed_at) if r.completed_at else None,
        }
        for r in runs
    ]


async def _run_evaluation_task(run_id: str, user_id: str):
    """Background evaluation task."""
    from datetime import datetime

    from app.db.base import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        stmt = select(EvaluationRun).where(EvaluationRun.id == run_id)
        result = await db.execute(stmt)
        run = result.scalar_one_or_none()
        if not run:
            return

        run.status = "running"
        await db.commit()

        try:
            if not EVAL_DATASET_PATH.exists():
                run.status = "failed"
                run.results = {"error": "Evaluation dataset not found at " + str(EVAL_DATASET_PATH)}
                await db.commit()
                return

            with open(EVAL_DATASET_PATH) as f:
                dataset = json.load(f)

            per_question = []
            start = time.time()

            # Import evaluation modules
            try:
                import sys
                root_candidate = EVAL_DATASET_PATH.parent.parent.parent
                if str(root_candidate) not in sys.path:
                    sys.path.insert(0, str(root_candidate))

                from evaluation.metrics.generation_metrics import (
                    AnswerRelevanceMetric,
                    FaithfulnessMetric,
                )
                from evaluation.metrics.retrieval_metrics import (
                    HitRate,
                    MeanReciprocalRank,
                    RecallAtK,
                )

                recall_metric = RecallAtK()
                hit_rate = HitRate()
                mrr_metric = MeanReciprocalRank()
                faithfulness = FaithfulnessMetric()
                relevance = AnswerRelevanceMetric()
            except ImportError:
                recall_metric = hit_rate = mrr_metric = faithfulness = relevance = None

            from app.agents.graph import run_graph

            recall_scores = []
            hit_scores = []
            mrr_scores = []
            faithfulness_scores = []
            relevance_scores = []

            for item in dataset[:5]:  # 5 questions for demo speed
                try:
                    state = await run_graph(
                        query=item["question"],
                        user_id=user_id,
                        db=db,
                    )
                    answer = state.get("answer", "")
                    retrieved_sources = [s.get("document", "") for s in state.get("sources", [])]
                    expected = item.get("expected_sources", [])

                    # Retrieval metrics
                    if recall_metric:
                        recall_scores.append(recall_metric.compute(retrieved_sources, expected, k=5))
                    if hit_rate:
                        hit_scores.append(hit_rate.compute(retrieved_sources, expected, k=5))
                    if mrr_metric:
                        mrr_scores.append(mrr_metric.compute(retrieved_sources, expected))

                    # Generation metrics
                    context = state.get("context", "")
                    if faithfulness:
                        faithfulness_scores.append(faithfulness.compute(answer, context))
                    if relevance:
                        relevance_scores.append(relevance.compute(item["question"], answer))

                    per_question.append({
                        "question": item["question"],
                        "answer": answer[:200],
                        "sources": retrieved_sources,
                        "confidence": state.get("confidence", 0),
                    })
                except Exception as e:
                    logger.warning("eval_question_failed", q=item["question"][:50], error=str(e))

            def _compute_avg(lst: list[float]) -> float:
                return round(sum(lst) / len(lst), 4) if lst else 0.0

            metrics = {
                "recall_at_5": _compute_avg(recall_scores),
                "hit_rate_at_5": _compute_avg(hit_scores),
                "mrr": _compute_avg(mrr_scores),
                "faithfulness": _compute_avg(faithfulness_scores),
                "answer_relevance": _compute_avg(relevance_scores),
                "questions_evaluated": len(per_question),
                "total_latency_seconds": round(time.time() - start, 2),
            }

            # Log to MLflow
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
            await db.commit()


@router.post("/evaluations/run")
async def run_evaluation(
    background_tasks: BackgroundTasks,
    request: dict = Body(default={}),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
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
    return {"id": str(run.id), "name": run.name, "status": run.status}
