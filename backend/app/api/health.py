import structlog
from fastapi import APIRouter
from sqlalchemy import text

router = APIRouter()
logger = structlog.get_logger()


@router.get("/health")
async def health_check():
    """Health check endpoint. Never raises 5xx."""
    from app.core.config import settings

    status = {
        "status": "ok",
        "version": settings.VERSION,
        "service": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
    }

    # Database
    try:
        from app.db.base import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        status["database"] = "healthy"
    except Exception as e:
        status["database"] = "unhealthy"
        status["status"] = "degraded"
        logger.warning("health_db_failed", error=str(e))

    # Redis
    if not settings.REDIS_URL or settings.REDIS_URL == "redis://localhost:6379/0":
        # Check if actually available or using default placeholder
        try:
            from app.services.redis_service import get_redis_service
            redis = get_redis_service()
            await redis._get_client()
            status["redis"] = "healthy" if redis.is_available() else "not_configured"
        except Exception:
            status["redis"] = "not_configured"
    else:
        try:
            from app.services.redis_service import get_redis_service
            redis = get_redis_service()
            await redis._get_client()
            status["redis"] = "healthy" if redis.is_available() else "unavailable"
        except Exception as e:
            status["redis"] = "unavailable"
            logger.warning("health_redis_failed", error=str(e))

    # Pinecone / Vector store
    try:
        from app.services.pinecone_service import get_vector_store
        vs = get_vector_store()
        if getattr(vs, "backend_name", None) == "pinecone":
            status["pinecone"] = "connected"
        else:
            status["pinecone"] = "using_in_memory_fallback"
    except Exception:
        status["pinecone"] = "unavailable"

    # LLM provider & Gemini API key configuration
    status["llm_provider"] = settings.LLM_PROVIDER
    status["gemini"] = "configured" if bool(settings.GEMINI_API_KEY) else "missing"

    # MLflow (optional — not in critical production path)
    if not settings.MLFLOW_TRACKING_URI:
        status["mlflow"] = "not_configured"
    else:
        try:
            from app.services.mlflow_service import get_mlflow_service
            mf = get_mlflow_service()
            status["mlflow"] = "connected" if mf._available else "unavailable"
        except Exception:
            status["mlflow"] = "unavailable"

    return status
