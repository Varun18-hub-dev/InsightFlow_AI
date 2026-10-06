from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.services.mlflow_service import get_mlflow_service

router = APIRouter()


@router.get("/experiments")
async def list_experiments(current_user=Depends(get_current_user)):
    svc = get_mlflow_service()
    return await svc.get_experiments()


@router.get("/experiments/{experiment_name}/runs")
async def list_experiment_runs(
    experiment_name: str,
    current_user=Depends(get_current_user),
):
    svc = get_mlflow_service()
    return await svc.get_runs(experiment_name)
