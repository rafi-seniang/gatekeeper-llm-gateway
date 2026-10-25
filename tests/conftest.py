from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from app.api.deps import get_gate
from app.core.config import Settings
from app.engines.upstream import UpstreamClient
from app.main import create_app
from app.repositories.answer_cache import AnswerCache
from app.repositories.events import EventLog
from app.services.gate import Gate


class FakeGuard:
    """Calls a text an injection when it contains the word 'IGNORE' (a stand-in for the classifier)."""

    name = "fake-guard"

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def probability(self, text: str) -> float:
        self.seen.append(text)
        return 0.97 if "IGNORE" in text else 0.02


class FakeDecisions:
    """Rates a request hard when it mentions code."""

    model = "fake-decisions"

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def difficulty(self, state: str) -> float:
        self.seen.append(state)
        return 1.8 if "code" in state else 0.2


class FakeUpstream:
    """An OpenAI-compatible upstream that records what it was sent; `down` makes it refuse connections."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.down = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("refused", request=request)
        body = httpx.Response(200, content=request.content).json()
        self.requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )


@pytest.fixture
def upstream() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture
def guard() -> FakeGuard:
    return FakeGuard()


@pytest.fixture
def decisions() -> FakeDecisions:
    return FakeDecisions()


@pytest.fixture
async def client(
    upstream: FakeUpstream, guard: FakeGuard, decisions: FakeDecisions
) -> AsyncIterator[httpx.AsyncClient]:
    settings = Settings(_env_file=None, environment="test", api_keys="test-key")  # type: ignore[call-arg]
    upstream_http = httpx.AsyncClient(base_url="http://upstream/v1", transport=httpx.MockTransport(upstream.handler))
    gate = Gate(
        settings=settings,
        guard=guard,
        decisions=decisions,  # type: ignore[arg-type]
        upstream=UpstreamClient(upstream_http),
        cache=AnswerCache(max_entries=10, ttl_seconds=60),
        events=EventLog(size=50),
    )
    app = create_app(settings)
    app.dependency_overrides[get_gate] = lambda: gate
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers={"X-API-Key": "test-key"}
    ) as http:
        yield http
    await upstream_http.aclose()
