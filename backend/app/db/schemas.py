"""Pydantic v2 schemas for InsightFlow AI API."""
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field

# ─── Auth ────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    full_name: str = Field(min_length=1, max_length=255)


class UserResponse(BaseModel):
    id: UUID
    email: EmailStr
    full_name: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    sub: str | None = None


# ─── Documents ───────────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: UUID
    user_id: UUID
    filename: str
    original_filename: str
    document_type: str
    file_size: int
    status: str
    page_count: int | None = None
    chunk_count: int | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int


class DocumentSummaryResponse(BaseModel):
    summary: str
    sources: list[dict[str, Any]] = []


class DocumentCompareResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = []
    document_ids: list[str]


# ─── Chat ─────────────────────────────────────────────────────────────────────

class Source(BaseModel):
    document: str
    page: Any = None
    chunk_id: str = ""
    score: float = 0.0
    document_id: str = ""


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    document_ids: list[str] | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = []
    confidence: float = 0.0
    conversation_id: str
    message_id: str


class StreamChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    document_ids: list[str] | None = None


# ─── Feedback ────────────────────────────────────────────────────────────────

class FeedbackCreate(BaseModel):
    message_id: UUID
    rating: str = Field(pattern="^(helpful|not_helpful)$")
    feedback_text: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    id: UUID
    message_id: UUID
    user_id: UUID
    rating: str
    feedback_text: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Evaluation ──────────────────────────────────────────────────────────────

class EvaluationRunResponse(BaseModel):
    id: UUID
    name: str
    status: str
    config: dict[str, Any] | None = None
    results: dict[str, Any] | None = None
    mlflow_run_id: str | None = None
    started_at: datetime
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


# ─── Health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    database: str = "unknown"
    redis: str = "unknown"
    pinecone: str = "unknown"
    llm_provider: str | None = None


# ─── Error ───────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
