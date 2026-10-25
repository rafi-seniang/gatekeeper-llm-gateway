"""Exact-match answer cache, keyed on the request as sent (no normalising: a different word is a different question)."""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any


def cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class AnswerCache:
    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self._entries: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        stored_at, answer = entry
        if time.monotonic() - stored_at > self._ttl_seconds:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)  # least recently used is evicted first
        return answer

    def put(self, key: str, answer: dict[str, Any]) -> None:
        if self._max_entries == 0:
            return
        self._entries[key] = (time.monotonic(), answer)
        self._entries.move_to_end(key)
        if len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
