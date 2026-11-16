"""Long-lived clients and models, built once in the lifespan and closed on shutdown. Held on app.state."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI
from redis.asyncio import Redis

from app.engines.agent_guard import AgentGuard
from app.engines.decisions import DecisionClient, build_gemini_clients, load_gemini_api_keys
from app.engines.identifier_tagger import IdentifierFinder, NoopIdentifierTagger, OnnxIdentifierTagger
from app.engines.injection import CascadeGuard, ClassifierGuard, DecisionGuard, InjectionGuard, JevGuard
from app.engines.upstream import UpstreamClient
from app.features.gate.cache import AnswerCache, InMemoryAnswerCache, RedisAnswerCache
from app.features.gate.events import EventLog
from app.features.gate.service import Gate
from app.platform.rate_limit import InMemoryRateLimiter, RateLimiter, RedisRateLimiter
from app.platform.redis import connect_redis
from config.settings import Settings


@dataclass
class Resources:
    gate: Gate
    rate_limiter: RateLimiter
    clients: list[httpx.AsyncClient]
    redis: Redis | None = None


def jev_client(settings: Settings) -> httpx.AsyncClient:
    """An HTTP client for TypeSafe's API. Refuses to start without the key rather than failing on every request."""
    api_key = settings.jev_api_key.get_secret_value()
    if not api_key:
        raise RuntimeError("The configured guards need JEV_API_KEY in the secrets file")
    return httpx.AsyncClient(
        base_url=settings.jev_base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=settings.jev_timeout_seconds,
    )


async def build_resources(settings: Settings) -> Resources:
    redis = await connect_redis(settings)
    cache: AnswerCache = (
        RedisAnswerCache(redis, settings.cache_ttl_seconds)
        if redis is not None
        else InMemoryAnswerCache(settings.cache_max_entries, settings.cache_ttl_seconds)
    )
    rate_limiter: RateLimiter = RedisRateLimiter(redis) if redis is not None else InMemoryRateLimiter()
    upstream_http = httpx.AsyncClient(
        base_url=settings.upstream_base_url,
        headers={"Authorization": f"Bearer {settings.upstream_api_key.get_secret_value()}"},
        timeout=settings.upstream_timeout_seconds,
    )
    decision_http = httpx.AsyncClient(
        base_url=settings.decision_base_url,
        headers={"Authorization": f"Bearer {settings.decision_api_key.get_secret_value()}"},
        timeout=settings.decision_timeout_seconds,
    )
    decisions = DecisionClient(decision_http, settings.decision_model, settings.decision_protocol)
    clients = [upstream_http, decision_http]
    gemini_keys = load_gemini_api_keys(settings)
    gemini_clients = build_gemini_clients(gemini_keys, settings.gemini_timeout_seconds) if gemini_keys else []

    jev_decisions: DecisionClient | None = None
    if settings.agent_guard_enabled or settings.injection_guard in ("jev", "cascade"):
        has_jev = bool(settings.jev_api_key.get_secret_value().strip())
        jev_http: httpx.AsyncClient | None = None
        if has_jev:
            jev_http = jev_client(settings)
            clients.append(jev_http)
        elif not gemini_clients:
            # Refuses to start when neither JEV_API_KEY nor any GEMINI_API_KEY is configured
            jev_client(settings)

        if has_jev and gemini_clients:
            model_label = f"{settings.jev_model} + {settings.gemini_decision_model}"
        elif has_jev:
            model_label = settings.jev_model
        else:
            model_label = settings.gemini_decision_model

        jev_decisions = DecisionClient(
            jev_http,
            settings.jev_model,
            "systemone",
            gemini_clients=gemini_clients,
            gemini_model=settings.gemini_decision_model,
            label=model_label,
        )
    guard: InjectionGuard
    if settings.injection_guard == "decision":
        guard = DecisionGuard(decisions)
    elif settings.injection_guard == "classifier":
        guard = ClassifierGuard(settings.injection_classifier_model, settings.device)
    elif jev_decisions is not None:  # always set when jev or cascade guard is configured
        jev = JevGuard(jev_decisions, default_scope=settings.default_allowed_scope)
        guard = (
            jev
            if settings.injection_guard == "jev"
            else CascadeGuard(
                ClassifierGuard(settings.injection_classifier_model, settings.device),
                jev,
                settings.block_from_injection_probability,
            )
        )
    finder: IdentifierFinder = (
        OnnxIdentifierTagger(
            settings.identifier_model,
            settings.identifier_model_revision,
            settings.identifier_span_threshold,
            settings.identifier_threads,
        )
        if settings.mask_identifiers_enabled
        else NoopIdentifierTagger()
    )
    gate = Gate(
        settings=settings,
        finder=finder,
        guard=guard,
        decisions=decisions,
        upstream=UpstreamClient(upstream_http),
        cache=cache,
        events=EventLog(settings.event_log_size),
        agent_guard=AgentGuard(jev_decisions) if settings.agent_guard_enabled and jev_decisions else None,
    )
    return Resources(gate=gate, rate_limiter=rate_limiter, clients=clients, redis=redis)


def lifespan(settings: Settings) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def run(app: FastAPI) -> AsyncIterator[None]:
        app.state.resources = await build_resources(settings)  # models and Redis load here, before any request
        yield
        resources: Resources = app.state.resources
        for client in resources.clients:  # two or three clients, closed in turn
            await client.aclose()
        if resources.redis is not None:
            await resources.redis.aclose()

    return run
