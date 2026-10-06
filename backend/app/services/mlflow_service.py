"""
MLflow service for LLMOps experiment tracking.
Gracefully handles unavailable MLflow server.
"""
import structlog

logger = structlog.get_logger()

class MLflowService:
    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        self._available = False
        self._initialized = False

    def _setup(self):
        if self._initialized:
            return
        if not self.tracking_uri:
            self._available = False
            self._initialized = True
            return
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            self._available = True
            self._initialized = True
            logger.info("mlflow_initialized", uri=self.tracking_uri)
        except Exception as e:
            logger.warning("mlflow_unavailable", error=str(e))
            self._initialized = True

    async def log_experiment_run(
        self,
        experiment_name: str,
        params: dict,
        metrics: dict,
        tags: dict | None = None
    ) -> str | None:
        self._setup()
        if not self._available:
            logger.warning("mlflow_log_skipped", reason="MLflow not available")
            return None
        try:
            import mlflow
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(tags=tags or {}) as run:
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                run_id = run.info.run_id
                logger.info("mlflow_run_logged", run_id=run_id, experiment=experiment_name)
                return run_id
        except Exception as e:
            logger.error("mlflow_log_failed", error=str(e))
            return None

    async def get_experiments(self) -> list[dict]:
        self._setup()
        if not self._available:
            return []
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            return [
                {
                    "experiment_id": e.experiment_id,
                    "name": e.name,
                    "artifact_location": e.artifact_location,
                    "lifecycle_stage": e.lifecycle_stage,
                }
                for e in experiments
            ]
        except Exception as e:
            logger.error("mlflow_get_experiments_failed", error=str(e))
            return []

    async def get_runs(self, experiment_name: str) -> list[dict]:
        self._setup()
        if not self._available:
            return []
        try:
            import mlflow
            runs = mlflow.search_runs(experiment_names=[experiment_name])
            if runs.empty:
                return []
            return runs.to_dict(orient="records")
        except Exception as e:
            logger.error("mlflow_get_runs_failed", error=str(e))
            return []

_mlflow_service: MLflowService | None = None

def get_mlflow_service() -> MLflowService:
    global _mlflow_service
    if _mlflow_service is None:
        from app.core.config import settings
        _mlflow_service = MLflowService(tracking_uri=settings.MLFLOW_TRACKING_URI)
    return _mlflow_service
