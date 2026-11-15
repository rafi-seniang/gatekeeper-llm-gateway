import httpx
import pytest

from app.platform import rate_limit
from config.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, environment="test", api_keys="test-key", rate_limit_per_minute=2)  # type: ignore[call-arg]


async def test_a_caller_over_the_limit_gets_429_with_retry_after(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rate_limit.time, "time", lambda: 1_000_040.0)  # 20 seconds into a minute, 40 before its end
    statuses = [(await client.get("/v1/gate/stats")).status_code for _ in range(3)]
    assert statuses == [200, 200, 429]
    refused = await client.get("/v1/gate/stats")
    assert refused.headers["Retry-After"] == "40"
    assert refused.headers["content-type"] == "application/problem+json"


async def test_the_limit_counts_each_api_key_on_its_own() -> None:
    limiter = rate_limit.InMemoryRateLimiter()
    assert [await limiter.hit("key:a", 1), await limiter.hit("key:b", 1)] == [None, None]
    assert await limiter.hit("key:a", 1) is not None


async def test_daily_gate_limit_caps_post_endpoints_without_limiting_stats(
    upstream: object,
    guard: object,
    decisions: object,
    agent_guard: object,
) -> None:
    import httpx

    from app.bootstrap import create_app
    from app.engines.upstream import UpstreamClient
    from app.features.gate.cache import InMemoryAnswerCache
    from app.features.gate.events import EventLog
    from app.features.gate.service import Gate
    from app.platform.dependencies import get_resources
    from app.platform.rate_limit import InMemoryRateLimiter
    from app.platform.resources import Resources
    from tests.conftest import FakeFinder

    daily_settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment="test",
        api_keys="test-key",
        rate_limit_per_minute=100,
        daily_requests_per_caller=1,
    )
    upstream_http = httpx.AsyncClient(base_url="http://upstream/v1", transport=httpx.MockTransport(upstream.handler))  # type: ignore[attr-defined]
    gate = Gate(
        settings=daily_settings,
        finder=FakeFinder(),
        guard=guard,  # type: ignore[arg-type]
        decisions=decisions,  # type: ignore[arg-type]
        upstream=UpstreamClient(upstream_http),
        cache=InMemoryAnswerCache(max_entries=10, ttl_seconds=60),
        events=EventLog(size=50),
        agent_guard=agent_guard,  # type: ignore[arg-type]
    )
    app = create_app(daily_settings)
    resources = Resources(gate=gate, rate_limiter=InMemoryRateLimiter(), clients=[upstream_http])
    app.dependency_overrides[get_resources] = lambda: resources
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", headers={"X-API-Key": "test-key"}
    ) as daily_client:
        payload = {"messages": [{"role": "user", "content": "hello"}]}
        first = await daily_client.post("/v1/gate/check", json=payload)
        second = await daily_client.post("/v1/gate/check", json=payload)
        stats_after = await daily_client.get("/v1/gate/stats")
    await upstream_http.aclose()

    assert first.status_code == 200
    assert second.status_code == 429
    assert stats_after.status_code == 200
