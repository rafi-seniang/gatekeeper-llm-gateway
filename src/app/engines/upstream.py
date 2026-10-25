"""The OpenAI-compatible endpoint allowed requests are forwarded to."""

import time
from typing import Any

import httpx

from app.core.errors import UpstreamUnavailable


class UpstreamClient:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def chat(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """The upstream's completion, unchanged, and how long it took in milliseconds."""
        body = (
            {**payload, "thinking": {"type": "disabled"}}
            if "deepseek" in str(payload.get("model", "")) and "thinking" not in payload
            else payload
        )
        started = time.perf_counter()
        try:
            response = await self._http.post("/chat/completions", json=body)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise UpstreamUnavailable(f"The upstream model did not answer: {type(exc).__name__}") from exc
        completion: dict[str, Any] = response.json()
        return completion, round((time.perf_counter() - started) * 1000, 1)
