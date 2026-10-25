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
