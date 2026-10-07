import json
import time
import uuid
from datetime import datetime

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.api.evaluations import _find_eval_dataset_path
from app.core.config import settings
from app.core.prompt_manager import prompt_manager
from app.db.base import AsyncSessionLocal, get_db
from app.db.models import ExperimentRun, User
from app.db.schemas import ExperimentRunRequest, ExperimentRunResponse
from app.llm.factory import LLMProviderFactory
from app.reranking.factory import get_reranker
from app.retrieval.hybrid_retriever import HybridRetriever
from app.services.mlflow_service import get_mlflow_service

logger = structlog.get_logger()
router = APIRouter()


@router.get("/experiments", response_model=list[ExperimentRunResponse])
async def list_experiments(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all PostgreSQL-persisted experiment runs for the authenticated user."""
    stmt = (
        select(ExperimentRun)
        .where(ExperimentRun.user_id == current_user.id)
        .order_by(ExperimentRun.created_at.desc())
    )
    result = await db.execute(stmt)
    runs = result.scalars().all()
    return runs


@router.get("/experiments/{experiment_id}", response_model=ExperimentRunResponse)
async def get_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get single experiment run details."""
    try:
        exp_uuid = uuid.UUID(experiment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment ID") from None

    stmt = select(ExperimentRun).where(
        ExperimentRun.id == exp_uuid,
        ExperimentRun.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Experiment run not found")
    return run


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete an experiment run."""
    try:
        exp_uuid = uuid.UUID(experiment_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment ID") from None

    stmt = select(ExperimentRun).where(
        ExperimentRun.id == exp_uuid,
        ExperimentRun.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Experiment run not found")

    await db.delete(run)
    await db.commit()
    return {"message": "Experiment run deleted"}


async def _run_experiment_task(run_id: str, user_id: str):
    """Background task to execute real RAG queries and benchmark metrics for an experiment."""
    exp_uuid = uuid.UUID(run_id)
    user_uuid = uuid.UUID(user_id)

    async with AsyncSessionLocal() as db:
        stmt = select(ExperimentRun).where(
            ExperimentRun.id == exp_uuid,
            ExperimentRun.user_id == user_uuid,
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
                run.error_message = f"Evaluation dataset not found at {dataset_path}"
                run.results = {"error": run.error_message}
                run.completed_at = datetime.utcnow()
                await db.commit()
                return

            with open(dataset_path) as f:
                dataset = json.load(f)

            # Import metrics
            from app.evaluation.eval_service import EvaluationMetricsSuite

            suite = EvaluationMetricsSuite()
            config = run.config or {}
            top_k_retrieval = config.get("top_k_retrieval", settings.TOP_K_RETRIEVAL)
            top_k_rerank = config.get("top_k_rerank", settings.TOP_K_RERANK)
            reranker_type = config.get("reranker_type", settings.RERANKER_TYPE)

            retriever = HybridRetriever(db)
            reranker = get_reranker(reranker_type)
            provider = LLMProviderFactory.get_provider()

            per_question = []
            question_scores = []
            start = time.time()

            # Benchmark up to 5 questions
            for item in dataset[:5]:
                q = item["question"]
                expected = item.get("expected_sources", [])

                # 1. Retrieval
                retrieved = await retriever.retrieve(
                    query=q,
                    filters={"user_id": str(user_uuid)},
                    final_top_k=top_k_retrieval,
                )

                # 2. Reranking
                reranked = await reranker.rerank(q, retrieved, top_k=top_k_rerank)

                # 3. Prompt & Generation
                context_parts = []
                retrieved_sources = []
                for chunk in reranked:
                    meta = chunk.metadata or {}
                    fname = meta.get("filename", "doc")
                    page = meta.get("page_number", "?")
                    context_parts.append(f"[Source: {fname}, Page {page}]\n{chunk.content}")
                    retrieved_sources.append(fname)

                context = "\n\n---\n\n".join(context_parts) if context_parts else "No context available."
                system_prompt = prompt_manager.get_prompt("rag", "system")
                answer_prompt = prompt_manager.render("rag", "answer", context=context, question=q)

                answer = await provider.chat([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": answer_prompt},
                ])

                # 4. Metric computation via canonical suite
                numeric_scores, _detailed = suite.evaluate_qa_item(
                    question=q,
                    expected_sources=expected,
                    retrieved_sources=retrieved_sources,
                    answer=answer,
                    context=context,
                )
                question_scores.append(numeric_scores)

                per_question.append({
                    "question": q,
                    "answer": answer[:200],
                    "sources": retrieved_sources,
                    "confidence": 0.85 if retrieved_sources else 0.2,
                    "metrics": numeric_scores,
                })

            total_latency = time.time() - start
            metrics = suite.aggregate_batch(question_scores, total_latency)

            # Optional MLflow tracking (silently no-ops if offline)
            mlflow_run_id = None
            try:
                mlflow_svc = get_mlflow_service()
                mlflow_run_id = await mlflow_svc.log_experiment_run(
                    experiment_name="insightflow-rag-experiments",
                    params=config,
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
            logger.info("experiment_completed", run_id=run_id, metrics=metrics)

        except Exception as e:
            logger.error("experiment_failed", run_id=run_id, error=str(e))
            run.status = "failed"
            run.error_message = str(e)
            run.results = {"error": str(e)}
            run.completed_at = datetime.utcnow()
            await db.commit()


@router.post("/experiments/run", response_model=ExperimentRunResponse)
async def run_experiment(
    request: ExperimentRunRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Start a real RAG hyperparameter experiment run stored in PostgreSQL."""
    exp_name = request.name or f"exp_{request.chunk_size}_{request.reranker_type}_{int(time.time())}"
    config = {
        "chunk_size": request.chunk_size,
        "chunk_overlap": request.chunk_overlap,
        "top_k_retrieval": request.top_k_retrieval,
        "top_k_rerank": request.top_k_rerank,
        "reranker_type": request.reranker_type,
        "model": request.model or settings.GEMINI_MODEL,
    }

    run = ExperimentRun(
        user_id=current_user.id,
        name=exp_name,
        status="pending",
        config=config,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    background_tasks.add_task(_run_experiment_task, str(run.id), str(current_user.id))
    logger.info("experiment_started", run_id=str(run.id), name=run.name)
    return run
