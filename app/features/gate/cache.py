"""Exact-match answer cache, keyed on the request as sent (no normalising: a different word is a different question).

In Redis when REDIS_URL is set, so every worker shares one cache and it survives a restart. In memory otherwise.
"""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Protocol

from redis.asyncio import Redis


def cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class AnswerCache(Protocol):
    async def get(self, key: str) -> dict[str, Any] | None: ...

    async def put(self, key: str, answer: dict[str, Any]) -> None: ...


class InMemoryAnswerCache:
    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self._entries: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds

    async def get(self, key: str) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, answer = entry
        if time.monotonic() - stored_at > self._ttl_seconds:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)  # least recently used is evicted first
        return answer

    async def put(self, key: str, answer: dict[str, Any]) -> None:
        if self._max_entries == 0:
            return
        self._entries[key] = (time.monotonic(), answer)
        self._entries.move_to_end(key)
        if len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


class RedisAnswerCache:
    """Each answer expires after its TTL. Redis's own memory policy bounds the total size."""

    def __init__(self, redis: Redis, ttl_seconds: float) -> None:
        self._redis = redis
        self._ttl_seconds = int(ttl_seconds)

    async def get(self, key: str) -> dict[str, Any] | None:
        stored = await self._redis.get(f"answer:{key}")
        if stored is None:
            return None
        answer: dict[str, Any] = json.loads(stored)
        return answer

    async def put(self, key: str, answer: dict[str, Any]) -> None:
        if self._ttl_seconds == 0:
            return
        await self._redis.set(f"answer:{key}", json.dumps(answer, ensure_ascii=False), ex=self._ttl_seconds)
