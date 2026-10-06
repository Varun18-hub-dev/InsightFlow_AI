from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import auth, chat, documents, evaluations, experiments, feedback, health
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.middleware import RequestIDMiddleware, TimingMiddleware
from app.core.tracing import setup_langsmith
from app.db.base import init_db
from app.services.mlflow_service import get_mlflow_service

setup_logging()
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    setup_langsmith()
    if settings.MLFLOW_TRACKING_URI:
        get_mlflow_service()._setup()
    yield
    # Shutdown (clean up resources if needed)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Enterprise Knowledge, Retrieval and AI Decision Platform",
    lifespan=lifespan,
)

# CORS
origins = settings.get_allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID & Timing Middlewares
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "An unexpected error occurred."}},
    )


# Include routers
app.include_router(health.router, tags=["Health"])  # Provides /health for Render
app.include_router(auth.router, prefix="/api", tags=["Authentication"])
app.include_router(documents.router, prefix="/api", tags=["Documents"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(feedback.router, prefix="/api", tags=["Feedback"])
app.include_router(evaluations.router, prefix="/api", tags=["Evaluations"])
app.include_router(experiments.router, prefix="/api", tags=["Experiments"])
app.include_router(health.router, prefix="/api", tags=["Health"])
