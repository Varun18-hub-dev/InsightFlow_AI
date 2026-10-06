import time
import uuid

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.exceptions import InsightFlowException
from app.core.logging import get_logger

logger = get_logger(__name__)

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info("request_completed", path=request.url.path, method=request.method, latency=process_time)
        return response

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except InsightFlowException as exc:
            logger.error("application_error", error=exc.message, code=exc.code)
            return JSONResponse(
                status_code=400,
                content={"error": {"code": exc.code, "message": exc.message}}
            )
        except Exception as exc:
            logger.error("unhandled_error", error=str(exc))
            return JSONResponse(
                status_code=500,
                content={"error": {"code": "INTERNAL_SERVER_ERROR", "message": "An unexpected error occurred."}}
            )

def configure_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
