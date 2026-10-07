"""
Unit tests for GeminiProvider validation, streaming, and error handling.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import settings
from app.llm.gemini_provider import GeminiProvider


@pytest.fixture
def provider():
    with patch("app.llm.gemini_provider.ChatGoogleGenerativeAI"):
        p = GeminiProvider()
        return p


def test_gemini_validates_empty_messages_list(provider):
    with pytest.raises(ValueError, match="Messages list is empty"):
        provider._validate_messages([])


def test_gemini_validates_empty_system_content(provider):
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Hello"},
    ]
    with pytest.raises(ValueError, match="empty content"):
        provider._validate_messages(messages)


def test_gemini_validates_empty_user_content(provider):
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "   "},
    ]
    with pytest.raises(ValueError, match="empty content"):
        provider._validate_messages(messages)


def test_gemini_validates_missing_api_key(provider):
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User prompt"},
    ]
    provider.api_key = ""
    with patch.object(settings, "GEMINI_API_KEY", ""):
        with patch.object(settings, "GOOGLE_API_KEY", ""):
            with pytest.raises(ValueError, match="is not configured"):
                provider._validate_messages(messages)


def test_extract_text_string_and_list(provider):
    assert provider._extract_text("hello") == "hello"
    assert provider._extract_text([{"type": "text", "text": "GEMINI_OK"}]) == "GEMINI_OK"
    assert provider._extract_text(["part1", "part2"]) == "part1part2"
    assert provider._extract_text(None) == ""


@pytest.mark.asyncio
async def test_gemini_stream_yields_chunks(provider):
    chunk1 = MagicMock(content="Hello ")
    chunk2 = MagicMock(content="world!")

    async def mock_astream(msgs):
        yield chunk1
        yield chunk2

    provider.client.astream = mock_astream

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User prompt"},
    ]

    tokens = []
    async for token in provider.chat_stream(messages):
        tokens.append(token)

    assert tokens == ["Hello ", "world!"]


@pytest.mark.asyncio
async def test_gemini_chat_returns_content(provider):
    mock_response = MagicMock(content="Answer text")
    provider.client.ainvoke = AsyncMock(return_value=mock_response)

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User prompt"},
    ]

    result = await provider.chat(messages)
    assert result == "Answer text"


@pytest.mark.asyncio
async def test_gemini_stream_handles_timeout(provider):
    async def mock_hanging_astream(msgs):
        await asyncio.sleep(2.0)
        yield MagicMock(content="Too late")

    provider.client.astream = mock_hanging_astream

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User prompt"},
    ]

    # Patch STREAM_TIMEOUT_SECONDS inside chat_stream or simulate with short timeout
    with patch("app.llm.gemini_provider.asyncio.wait_for", side_effect=TimeoutError()):
        with pytest.raises(TimeoutError, match="timed out"):
            async for _ in provider.chat_stream(messages):
                pass


@pytest.mark.asyncio
async def test_gemini_embedding_run_diagnostic():
    from app.llm.gemini_provider import GeminiEmbeddingProvider

    with patch("app.llm.gemini_provider.GoogleGenerativeAIEmbeddings"):
        p = GeminiEmbeddingProvider()
        p.embed_documents = AsyncMock(return_value=[[0.1] * 3072])
        p.embed_query = AsyncMock(return_value=[0.1] * 3072)

        res = await p.run_diagnostic()
        assert res["document"]["status"] == "SUCCESS"
        assert res["document"]["dimension"] == 3072
        assert res["query"]["status"] == "SUCCESS"
        assert res["query"]["dimension"] == 3072
        assert res["query"]["task_type"] == "RETRIEVAL_QUERY"
