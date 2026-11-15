"""Requests per caller per minute and optional daily caps, counted in Redis so every worker shares one count.

A fixed window: one INCR and one EXPIRE per request, in a single round trip.
"""

import datetime as dt
import secrets
import time
from typing import Protocol

from fastapi import Request
from redis.asyncio import Redis

WINDOW_SECONDS = 60
DAY_SECONDS = 86_400


class RateLimiter(Protocol):
    async def hit(
        self, caller: str, limit: int, window_seconds: int = WINDOW_SECONDS, prefix: str = "rate"
    ) -> int | None:
        """Counts one request. None when allowed, otherwise the seconds until the window resets."""
        ...


def window_key(caller: str, now: float, window_seconds: int = WINDOW_SECONDS, prefix: str = "rate") -> tuple[str, int]:
    if window_seconds == DAY_SECONDS:
        day_start = int(
            dt.datetime.fromtimestamp(now, tz=dt.UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        )
        return f"{prefix}:{caller}:{day_start}", day_start + DAY_SECONDS - int(now)
    window_start = int(now // window_seconds) * window_seconds
    return f"{prefix}:{caller}:{window_start}", window_start + window_seconds - int(now)


class RedisRateLimiter:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def hit(
        self, caller: str, limit: int, window_seconds: int = WINDOW_SECONDS, prefix: str = "rate"
    ) -> int | None:
        key, seconds_left = window_key(caller, time.time(), window_seconds, prefix)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, window_seconds)
            count, _ = await pipe.execute()
        return None if int(count) <= limit else seconds_left


class InMemoryRateLimiter:
    """For tests and a single local process only: each worker would count on its own."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    async def hit(
        self, caller: str, limit: int, window_seconds: int = WINDOW_SECONDS, prefix: str = "rate"
    ) -> int | None:
        key, seconds_left = window_key(caller, time.time(), window_seconds, prefix)
        self._counts[key] = self._counts.get(key, 0) + 1
        return None if self._counts[key] <= limit else seconds_left


def caller_of(request: Request, accepted_keys: list[str]) -> str:
    """An accepted API key when one is sent, otherwise the client address. Any other key would be a fresh count."""
    api_key = request.headers.get("X-API-Key", "")
    if any(secrets.compare_digest(api_key.encode(), key.encode()) for key in accepted_keys):
        return f"key:{api_key[-8:]}"  # the key's end tells callers apart and keeps the full key out of Redis
    return f"ip:{request.client.host if request.client else 'unknown'}"
