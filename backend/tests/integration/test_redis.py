"""Integration tests for Redis service (uses mocks — no live Redis required)."""
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_cache_set_and_get():
    """Set a value in cache and retrieve it."""
    from app.services.redis_service import RedisService

    svc = RedisService(url="redis://localhost:6379")
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)
    mock_client.setex = AsyncMock(return_value=True)
    mock_client.get = AsyncMock(return_value='{"answer": "test answer", "sources": []}')

    svc._client = mock_client
    svc._available = True

    await svc.set_cached_response("key1", {"answer": "test answer", "sources": []})
    result = await svc.get_cached_response("key1")

    assert result is not None
    assert result["answer"] == "test answer"


@pytest.mark.asyncio
async def test_cache_miss_returns_none():
    """Missing key returns None."""
    from app.services.redis_service import RedisService

    svc = RedisService(url="redis://localhost:6379")
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=None)
    svc._client = mock_client
    svc._available = True

    result = await svc.get_cached_response("nonexistent_key_xyz")
    assert result is None


@pytest.mark.asyncio
async def test_rate_limit_allows_under_limit():
    """Returns True when request count is under the limit."""
    from app.services.redis_service import RedisService

    svc = RedisService(url="redis://localhost:6379")
    mock_client = AsyncMock()
    mock_pipeline = MagicMock()  # pipeline itself is sync
    mock_pipeline.incr = MagicMock()   # queued calls are sync
    mock_pipeline.expire = MagicMock()
    mock_pipeline.execute = AsyncMock(return_value=[5, True])  # count=5 < limit=60
    mock_client.pipeline = MagicMock(return_value=mock_pipeline)
    svc._client = mock_client
    svc._available = True

    allowed = await svc.rate_limit_check("user123", limit=60)
    assert allowed is True


@pytest.mark.asyncio
async def test_rate_limit_blocks_over_limit():
    """Returns False when request count exceeds the limit."""
    from app.services.redis_service import RedisService

    svc = RedisService(url="redis://localhost:6379")
    mock_client = AsyncMock()
    mock_pipeline = MagicMock()
    mock_pipeline.incr = MagicMock()
    mock_pipeline.expire = MagicMock()
    mock_pipeline.execute = AsyncMock(return_value=[61, True])  # count=61 > limit=60
    mock_client.pipeline = MagicMock(return_value=mock_pipeline)
    svc._client = mock_client
    svc._available = True

    allowed = await svc.rate_limit_check("user123", limit=60)
    assert allowed is False


@pytest.mark.asyncio
async def test_unavailable_redis_always_allows():
    """When Redis is down, rate limit check should allow (fail open)."""
    from app.services.redis_service import RedisService

    svc = RedisService(url="redis://invalid:6379")
    svc._available = False
    allowed = await svc.rate_limit_check("user123")
    assert allowed is True


@pytest.mark.asyncio
async def test_cache_key_generation():
    """Same inputs produce same cache key; different inputs produce different keys."""
    from app.services.redis_service import RedisService

    key1 = RedisService.generate_cache_key("what is revenue?", "user-abc")
    key2 = RedisService.generate_cache_key("what is revenue?", "user-abc")
    key3 = RedisService.generate_cache_key("different query", "user-abc")

    assert key1 == key2
    assert key1 != key3


@pytest.mark.asyncio
async def test_unavailable_redis_returns_none_on_get():
    """get_cached_response returns None when Redis is unavailable."""
    from app.services.redis_service import RedisService

    svc = RedisService(url="redis://invalid:6379")
    svc._available = False

    result = await svc.get_cached_response("any_key")
    assert result is None
