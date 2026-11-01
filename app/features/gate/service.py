"""The gate: masking, cache, guards, routing, then the upstream. Every request is logged."""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any, Literal

from app.engines.decisions import DecisionClient
from app.engines.identifier_tagger import IdentifierFinder
from app.engines.injection import InjectionGuard
from app.engines.upstream import UpstreamClient
from app.features.gate.cache import AnswerCache, cache_key
from app.features.gate.events import EventLog
from app.features.gate.identifiers import mask_conversation, restore_completion, with_tag_rule
from app.features.gate.schemas import ChatMessage, ChatRequest, GateDecision, GateEvent, Route
from app.platform.errors import AppError, RequestBlocked
from config.logging import request_id
from config.settings import Settings


def last_user_text(messages: list[ChatMessage]) -> str:
    return next(message.content or "" for message in reversed(messages) if message.role == "user")


class Gate:
    def __init__(
        self,
        settings: Settings,
        finder: IdentifierFinder,
        guard: InjectionGuard,
        decisions: DecisionClient,
        upstream: UpstreamClient,
        cache: AnswerCache,
        events: EventLog,
    ) -> None:
        self._settings = settings
        self._finder = finder
        self._guard = guard
        self._decisions = decisions
        self._upstream = upstream
        self._cache = cache
        self.events = events

    @property
    def guard_name(self) -> str:
        return self._guard.name

    async def _decide(
        self, masked_messages: list[ChatMessage], masked_count: int, pinned_route: Route | None, started: float
    ) -> GateDecision:
        guarded_texts = [message.content or "" for message in masked_messages if message.role == "tool"]
        guarded_texts.append(last_user_text(masked_messages))
        probabilities = await asyncio.gather(*(self._guard.probability(text) for text in guarded_texts))
        injection_probability = max(probabilities)
        blocked = injection_probability >= self._settings.block_from_injection_probability
        # Only an allowed request needs a route, so a blocked one never waits on the decision endpoint
        difficulty = None
        if pinned_route is None and not blocked:
            difficulty = await self._decisions.difficulty(last_user_text(masked_messages))

        needs_strong_model = difficulty is not None and difficulty >= self._settings.strong_from_difficulty
        route: Route | None = None if blocked else pinned_route or ("strong" if needs_strong_model else "small")
        return GateDecision(
            injection_probability=round(injection_probability, 4),
            blocked=blocked,
            route=route,
            difficulty=None if difficulty is None else round(difficulty, 3),
            masked_identifiers=masked_count,
            decision_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    async def check(self, messages: list[ChatMessage]) -> GateDecision:
        """The decision for a conversation, without sending it anywhere."""
        started = time.perf_counter()
        masked_messages, vault = await mask_conversation(messages, self._finder)
        return await self._decide(masked_messages, vault.replaced, None, started)

    def _model_for(self, route: Route) -> str:
        return self._settings.strong_model if route == "strong" else self._settings.small_model

    def _log(
        self,
        preview: str,
        outcome: Literal["answered", "cached", "blocked", "failed"],
        decision: GateDecision | None,
        **fields: Any,
    ) -> None:
        self.events.add(
            GateEvent(
                request_id=request_id.get(),
                at=datetime.now(UTC).isoformat(timespec="seconds"),
                outcome=outcome,
                route=decision.route if decision else None,
                model=fields.get("model"),
                injection_probability=decision.injection_probability if decision else None,
                difficulty=decision.difficulty if decision else None,
                masked_identifiers=decision.masked_identifiers if decision else fields.get("masked_identifiers", 0),
                input_tokens=fields.get("input_tokens"),
                output_tokens=fields.get("output_tokens"),
                decision_ms=decision.decision_ms if decision else None,
                upstream_ms=fields.get("upstream_ms"),
                preview=preview,  # always from the masked text: this log is served by /v1/gate/events
            )
        )

    async def complete(self, request: ChatRequest) -> dict[str, Any]:
        """An OpenAI chat completion with real values restored and a `gate` field describing what the gate decided."""
        started = time.perf_counter()
        masked_messages, vault = await mask_conversation(request.messages, self._finder)
        preview = last_user_text(masked_messages)[:120]
        sampling = request.model_extra or {}
        masked_dump = [message.model_dump() for message in masked_messages]
        # Everything after masking sees only tags, so requests that differ only in their values share one stored answer
        key = cache_key({"model": request.model, "messages": masked_dump, **sampling})
        cached = await self._cache.get(key)
        if cached is not None:
            self._log(preview, "cached", None, model=cached.get("model"), masked_identifiers=vault.replaced)
            return restore_completion({**cached, "gate": {**cached["gate"], "cached": True}}, vault)

        pinned_route = None if request.model == "auto" else request.model
        decision = await self._decide(masked_messages, vault.replaced, pinned_route, started)
        if decision.blocked:
            self._log(preview, "blocked", decision)
            raise RequestBlocked(
                "The request reads as a prompt injection and was not sent to a model.",
                {"gate": decision.model_dump(), "guard": self.guard_name},
            )

        route: Route = decision.route or "small"  # set whenever the request was not blocked
        model = self._model_for(route)
        payload = {**sampling, "model": model, "messages": with_tag_rule(masked_dump, vault), "stream": False}
        try:
            completion, upstream_ms = await self._upstream.chat(payload)
        except AppError:
            self._log(preview, "failed", decision, model=model)
            raise
        usage = completion.get("usage") or {}
        log_fields = {
            "model": model,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "upstream_ms": upstream_ms,
        }
        self._log(preview, "answered", decision, **log_fields)
        answer = {
            **completion,
            "gate": {**decision.model_dump(), "model": model, "upstream_ms": upstream_ms, "cached": False},
        }
        await self._cache.put(key, answer)  # with its tags: the cache never holds a real value
        return restore_completion(answer, vault)
