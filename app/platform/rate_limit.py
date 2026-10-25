"""Requests per caller per minute, counted in Redis so every worker and replica shares one count.

A fixed one-minute window: one INCR and one EXPIRE per request, in a single round trip.
"""

import time
from typing import Protocol

from fastapi import Request
from redis.asyncio import Redis

WINDOW_SECONDS = 60


class RateLimiter(Protocol):
    async def hit(self, caller: str, limit: int) -> int | None:
        """Counts one request. None when allowed, otherwise the seconds until the window resets."""
        ...


def window_key(caller: str, now: float) -> tuple[str, int]:
    window_start = int(now // WINDOW_SECONDS) * WINDOW_SECONDS
    return f"rate:{caller}:{window_start}", window_start + WINDOW_SECONDS - int(now)


class RedisRateLimiter:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def hit(self, caller: str, limit: int) -> int | None:
        key, seconds_left = window_key(caller, time.time())
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, WINDOW_SECONDS)
            count, _ = await pipe.execute()
        return None if int(count) <= limit else seconds_left


class InMemoryRateLimiter:
    """For tests and a single local process only: each worker would count on its own."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    async def hit(self, caller: str, limit: int) -> int | None:
        key, seconds_left = window_key(caller, time.time())
        self._counts[key] = self._counts.get(key, 0) + 1
        return None if self._counts[key] <= limit else seconds_left


def caller_of(request: Request) -> str:
    """The API key when one is sent, otherwise the client address."""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key[-8:]}"  # the key's end tells callers apart and keeps the full key out of Redis
    return f"ip:{request.client.host if request.client else 'unknown'}"
