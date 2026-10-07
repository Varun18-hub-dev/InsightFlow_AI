"""
Redis service with graceful fallback when Redis is unavailable.
"""
import hashlib
import json

import structlog

logger = structlog.get_logger()

class RedisService:
    def __init__(self, url: str, ttl: int = 3600):
        self.url = url
        self.default_ttl = ttl
        self._client = None
        self._available = False

    async def _get_client(self):
        if self._client is None:
            if not self.url:
                self._available = False
                return None
            try:
                import redis.asyncio as aioredis
                self._client = aioredis.from_url(self.url, decode_responses=True)
                await self._client.ping()
                self._available = True
                logger.info("redis_connected", url=self.url)
            except Exception as e:
                logger.warning(
                    "redis_unavailable",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                self._available = False
        return self._client

    def is_available(self) -> bool:
        return self._available

    @staticmethod
    def generate_cache_key(query: str, user_id: str, **kwargs) -> str:
        content = f"{query}:{user_id}:{sorted(kwargs.items())}"
        return hashlib.md5(content.encode()).hexdigest()

    async def get_cached_response(self, cache_key: str) -> dict | None:
        client = await self._get_client()
        if not self._available or client is None:
            return None
        try:
            data = await client.get(f"cache:{cache_key}")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning("redis_get_failed", key=cache_key, error=str(e))
            return None

    async def set_cached_response(self, cache_key: str, response: dict, ttl: int | None = None):
        client = await self._get_client()
        if not self._available or client is None:
            return
        try:
            await client.setex(
                f"cache:{cache_key}",
                ttl or self.default_ttl,
                json.dumps(response)
            )
        except Exception as e:
            logger.warning("redis_set_failed", key=cache_key, error=str(e))

    async def delete_cache(self, cache_key: str):
        client = await self._get_client()
        if not self._available or client is None:
            return
        try:
            await client.delete(f"cache:{cache_key}")
        except Exception as e:
            logger.warning("redis_delete_failed", error=str(e))

    async def rate_limit_check(self, user_id: str, limit: int = 60, window: int = 60) -> bool:
        """Returns True if request is allowed, False if rate limited."""
        client = await self._get_client()
        if not self._available or client is None:
            return True  # Allow if Redis unavailable
        try:
            key = f"ratelimit:{user_id}"
            pipe = client.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            results = await pipe.execute()
            count = results[0]
            return count <= limit
        except Exception as e:
            logger.warning("redis_rate_limit_failed", error=str(e))
            return True

    async def store_session_state(self, session_id: str, state: dict, ttl: int = 3600):
        client = await self._get_client()
        if not self._available or client is None:
            return
        try:
            await client.setex(f"session:{session_id}", ttl, json.dumps(state))
        except Exception as e:
            logger.warning("redis_session_store_failed", error=str(e))

    async def get_session_state(self, session_id: str) -> dict | None:
        client = await self._get_client()
        if not self._available or client is None:
            return None
        try:
            data = await client.get(f"session:{session_id}")
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning("redis_session_get_failed", error=str(e))
            return None

# Singleton
_redis_service: RedisService | None = None

def get_redis_service() -> RedisService:
    global _redis_service
    if _redis_service is None:
        from app.core.config import settings
        _redis_service = RedisService(url=settings.REDIS_URL, ttl=settings.CACHE_TTL_SECONDS)
    return _redis_service
