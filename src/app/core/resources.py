"""Long-lived clients and models, built once in the lifespan and closed on shutdown. Held on app.state."""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI

from app.core.config import Settings
from app.gate.cache import AnswerCache
from app.gate.decisions import DecisionClient
from app.gate.events import EventLog
from app.gate.injection import ClassifierGuard, DecisionGuard, InjectionGuard
from app.gate.service import Gate
from app.gate.upstream import UpstreamClient


@dataclass
class Resources:
    gate: Gate
    clients: list[httpx.AsyncClient]


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
    guard: InjectionGuard = (
        ClassifierGuard(settings.injection_classifier_model, settings.device)
        if settings.injection_guard == "classifier"
        else DecisionGuard(decisions)
    )
    gate = Gate(
        settings=settings,
        guard=guard,
        decisions=decisions,
        upstream=UpstreamClient(upstream_http),
        cache=AnswerCache(settings.cache_max_entries, settings.cache_ttl_seconds),
        events=EventLog(settings.event_log_size),
    )
    return Resources(gate=gate, clients=[upstream_http, decision_http])


def lifespan(settings: Settings) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def run(app: FastAPI) -> AsyncIterator[None]:
        app.state.resources = build_resources(settings)  # the classifier loads here, before the first request
        yield
        resources: Resources = app.state.resources
        for client in resources.clients:  # two clients, closed in turn
            await client.aclose()

    return run
