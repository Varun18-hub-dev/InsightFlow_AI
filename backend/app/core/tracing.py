import os

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

def setup_langsmith():
    if settings.LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
        logger.info("langsmith_configured", project=settings.LANGCHAIN_PROJECT)
    else:
        logger.info("langsmith_not_configured")

def is_tracing_enabled() -> bool:
    return bool(settings.LANGCHAIN_API_KEY and settings.LANGCHAIN_TRACING_V2.lower() == "true")

def get_tracer():
    if is_tracing_enabled():
        from langchain.callbacks.tracers import LangChainTracer
        return LangChainTracer(project_name=settings.LANGCHAIN_PROJECT)
    return None
