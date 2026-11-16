"""Dependency providers shared by every feature. Tests swap any of them with app.dependency_overrides."""

from typing import Annotated

from fastapi import Depends, Request

from app.platform.errors import RateLimited
from app.platform.rate_limit import DAY_SECONDS, RateLimiter, caller_of
from app.platform.resources import Resources
from config.settings import Settings


def get_resources(request: Request) -> Resources:
    resources: Resources = request.app.state.resources
    return resources


def get_settings(request: Request) -> Settings:
    """The settings this app instance was built with (not a global), so tests and workers can differ."""
    settings: Settings = request.app.state.settings
    return settings


def get_rate_limiter(resources: Annotated[Resources, Depends(get_resources)]) -> RateLimiter:
    return resources.rate_limiter


async def enforce_rate_limit(
    request: Request,
    rate_limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    retry_after = await rate_limiter.hit(caller_of(request, settings.api_key_list), settings.rate_limit_per_minute)
    if retry_after is not None:
        raise RateLimited(f"Over {settings.rate_limit_per_minute} requests a minute.", retry_after)


async def enforce_daily_gate_limit(
    request: Request,
    rate_limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    if caller_of(request, settings.trusted_api_key_list).startswith("key:"):
        return  # one of the lab's own apps: its own budget limits it, not the visitors' daily cap
    caller = caller_of(request, settings.api_key_list)
    if settings.daily_requests_per_caller > 0:
        daily_retry = await rate_limiter.hit(
            caller, settings.daily_requests_per_caller, window_seconds=DAY_SECONDS, prefix="daily"
        )
        if daily_retry is not None:
            raise RateLimited(f"Over {settings.daily_requests_per_caller} requests today.", daily_retry)
    if settings.daily_requests_total > 0:
        total_retry = await rate_limiter.hit(
            "all", settings.daily_requests_total, window_seconds=DAY_SECONDS, prefix="daily-total"
        )
        if total_retry is not None:
            raise RateLimited("Service daily request limit reached.", total_retry)
