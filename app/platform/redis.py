"""The one Redis client, connected at startup. Empty REDIS_URL means none: fine for local runs, never in production."""

import logging

from redis.asyncio import Redis

from config.settings import Settings

logger = logging.getLogger(__name__)


async def connect_redis(settings: Settings) -> Redis | None:
    url = settings.redis_url.get_secret_value()
    if not url:
        if settings.environment == "production":
            raise RuntimeError("REDIS_URL is required in production")
        logger.warning("REDIS_URL is empty: the rate limit and any cache are per process")
        return None
    redis = Redis.from_url(url, socket_timeout=settings.redis_timeout_seconds)
    await redis.ping()  # refuse to start when Redis is unreachable, rather than fail on the first request
    return redis
