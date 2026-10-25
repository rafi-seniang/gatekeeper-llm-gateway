"""Long-lived clients and models, built once in the lifespan and closed on shutdown. Held on app.state."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI

from app.core.config import Settings
from app.engines.decisions import DecisionClient
from app.engines.injection import CascadeGuard, ClassifierGuard, DecisionGuard, InjectionGuard, JevGuard
from app.engines.upstream import UpstreamClient
from app.repositories.answer_cache import AnswerCache
from app.repositories.events import EventLog
from app.services.gate import Gate


@dataclass
class Resources:
    gate: Gate
    clients: list[httpx.AsyncClient]


def jev_client(settings: Settings) -> httpx.AsyncClient:
    """An HTTP client for TypeSafe's API. Refuses to start without the key rather than failing on every request."""
    api_key = settings.jev_api_key.get_secret_value()
    if not api_key:
        raise RuntimeError(f"INJECTION_GUARD={settings.injection_guard} needs JEV_API_KEY in the secrets file")
    return httpx.AsyncClient(
        base_url=settings.jev_base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=settings.jev_timeout_seconds,
    )


def build_resources(settings: Settings) -> Resources:
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
    guard: InjectionGuard
    if settings.injection_guard == "decision":
        guard = DecisionGuard(decisions)
    elif settings.injection_guard == "classifier":
        guard = ClassifierGuard(settings.injection_classifier_model, settings.device)
    else:
        jev_http = jev_client(settings)
        clients.append(jev_http)
        jev = JevGuard(DecisionClient(jev_http, settings.jev_model, "systemone"))
        guard = (
            jev
            if settings.injection_guard == "jev"
            else CascadeGuard(
                ClassifierGuard(settings.injection_classifier_model, settings.device),
                jev,
                settings.block_from_injection_probability,
            )
        )
    gate = Gate(
        settings=settings,
        guard=guard,
        decisions=decisions,
        upstream=UpstreamClient(upstream_http),
        cache=AnswerCache(settings.cache_max_entries, settings.cache_ttl_seconds),
        events=EventLog(settings.event_log_size),
    )
    return Resources(gate=gate, clients=clients)


def lifespan(settings: Settings) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def run(app: FastAPI) -> AsyncIterator[None]:
        app.state.resources = build_resources(settings)  # the classifier loads here, before the first request
        yield
        resources: Resources = app.state.resources
        for client in resources.clients:  # two clients, closed in turn
            await client.aclose()

    return run
