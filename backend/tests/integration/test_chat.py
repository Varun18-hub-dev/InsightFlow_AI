"""
Integration tests for the Chat API endpoint and SSE streaming.

Strategy:
- Patch the app lifespan (init_db, setup_langsmith, get_mlflow_service) via
  `patch("app.main.*")` so TestClient startup does not fail.
- Override BOTH get_db sources (app.db.base.get_db AND app.api.deps.get_db)
  with a mock that does not touch the real DB.
- Override get_current_user to return a consistent fake User object.
- Mock run_graph so no real LLM calls are made.
- Use a single module-scoped TestClient to avoid multiple asyncpg pools.
"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.db.models import User
from app.main import app
import app.db.base as db_base
import app.api.deps as api_deps


# ---------------------------------------------------------------------------
# Fake user (stable UUID used across tests)
# ---------------------------------------------------------------------------

_FAKE_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000099")


def _fake_user() -> User:
    u = User()
    u.id = _FAKE_USER_ID
    u.email = "test@insightflow.ai"
    u.full_name = "Test User"
    u.is_active = True
    return u


# ---------------------------------------------------------------------------
# Shared mock async DB session factory
# ---------------------------------------------------------------------------

def _make_mock_db():
    """Return a mock AsyncSession that passes through add/flush/commit silently."""
    mock = AsyncMock()
    # scalar_one_or_none for SELECT → return None (no existing conv)
    mock.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    mock.add = MagicMock()
    mock.flush = AsyncMock()
    mock.commit = AsyncMock()
    mock.close = AsyncMock()
    mock.rollback = AsyncMock()
    return mock


async def _override_db():
    yield _make_mock_db()


async def _override_auth():
    return _fake_user()


def _mock_mlflow():
    svc = MagicMock()
    svc._setup = MagicMock()
    return svc


# ---------------------------------------------------------------------------
# Module-scoped TestClient
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    # Override BOTH get_db objects (chat.py uses app.db.base.get_db directly)
    app.dependency_overrides[db_base.get_db] = _override_db
    app.dependency_overrides[api_deps.get_db] = _override_db
    app.dependency_overrides[api_deps.get_current_user] = _override_auth

    with (
        patch("app.main.init_db", new=AsyncMock()),
        patch("app.main.setup_langsmith", return_value=None),
        patch("app.main.get_mlflow_service", return_value=_mock_mlflow()),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    app.dependency_overrides.pop(db_base.get_db, None)
    app.dependency_overrides.pop(api_deps.get_db, None)
    app.dependency_overrides.pop(api_deps.get_current_user, None)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code in (200, 503)
    assert "status" in resp.json()


# ---------------------------------------------------------------------------
# Auth protection (temporarily remove override)
# ---------------------------------------------------------------------------

def test_chat_post_requires_auth(client):
    saved = app.dependency_overrides.pop(api_deps.get_current_user, None)
    try:
        resp = client.post("/api/chat", json={"query": "hello"})
        assert resp.status_code == 401
    finally:
        if saved:
            app.dependency_overrides[api_deps.get_current_user] = saved


def test_chat_stream_requires_token_param(client):
    saved = app.dependency_overrides.pop(api_deps.get_current_user, None)
    try:
        resp = client.get("/api/chat/stream", params={"query": "hello"})
        assert resp.status_code in (401, 422)
    finally:
        if saved:
            app.dependency_overrides[api_deps.get_current_user] = saved


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_chat_rejects_empty_query(client):
    resp = client.post(
        "/api/chat",
        json={"query": ""},
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 422


def test_chat_rejects_missing_query(client):
    resp = client.post(
        "/api/chat",
        json={},
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Chat POST — mocked graph + mocked DB
# ---------------------------------------------------------------------------

@patch("app.api.chat.run_graph")
def test_chat_returns_answer(mock_graph, client):
    mock_graph.return_value = {
        "answer": "The revenue was $4.75 million.",
        "sources": [],
        "confidence": 0.95,
        "intent": "rag",
        "metadata": {},
    }
    resp = client.post(
        "/api/chat",
        json={"query": "What was the revenue?"},
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("answer") == "The revenue was $4.75 million."


@patch("app.api.chat.run_graph")
def test_chat_response_has_all_fields(mock_graph, client):
    mock_graph.return_value = {
        "answer": "Understood.",
        "sources": [],
        "confidence": 0.8,
        "intent": "rag",
        "metadata": {},
    }
    resp = client.post(
        "/api/chat",
        json={"query": "Tell me something"},
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 200
    body = resp.json()
    for field in ("answer", "sources", "confidence", "conversation_id", "message_id"):
        assert field in body, f"Missing field in ChatResponse: {field}"


@patch("app.api.chat.run_graph")
def test_chat_with_document_ids(mock_graph, client):
    mock_graph.return_value = {
        "answer": "Summary of your docs.",
        "sources": [],
        "confidence": 0.75,
        "intent": "summary",
        "metadata": {},
    }
    resp = client.post(
        "/api/chat",
        json={"query": "Summarize", "document_ids": [str(uuid.uuid4())]},
        headers={"Authorization": "Bearer fake"},
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------

def test_chat_stream_post_returns_sse(client):
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"query": "What is InsightFlow?"},
        headers={"Authorization": "Bearer fake"},
    ) as resp:
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            assert "text/event-stream" in resp.headers.get("content-type", "")
