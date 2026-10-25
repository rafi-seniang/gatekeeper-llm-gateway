"""The gate: cache, injection guard, identifier masking, routing, then the upstream. Every request is logged."""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any, Literal

from app.core.config import Settings
from app.core.errors import AppError, RequestBlocked
from app.core.logging import request_id
from app.gate.cache import AnswerCache, cache_key
from app.gate.decisions import DecisionClient
from app.gate.events import EventLog
from app.gate.identifiers import mask_identifiers
from app.gate.injection import InjectionGuard
from app.gate.upstream import UpstreamClient
from app.schemas import ChatMessage, ChatRequest, GateDecision, GateEvent, Route

GUARDED_ROLES = {"user", "tool"}  # text from outside the application, where injections arrive


def last_user_text(messages: list[ChatMessage]) -> str:
    return next(message.content for message in reversed(messages) if message.role == "user")


class Gate:
    def __init__(
        self,
        settings: Settings,
        guard: InjectionGuard,
        decisions: DecisionClient,
        upstream: UpstreamClient,
        cache: AnswerCache,
        events: EventLog,
    ) -> None:
        self._settings = settings
        self._guard = guard
        self._decisions = decisions
        self._upstream = upstream
        self._cache = cache
        self.events = events

    @property
    def guard_name(self) -> str:
        return self._guard.name

    async def decide(
        self, messages: list[ChatMessage], pinned_route: Route | None
    ) -> tuple[GateDecision, list[ChatMessage]]:
        """The decision for a request, and its messages with identifiers masked."""
        started = time.perf_counter()
        guarded_texts = [message.content for message in messages if message.role == "tool"] + [last_user_text(messages)]
        probabilities = await asyncio.gather(*(self._guard.probability(text) for text in guarded_texts))
        injection_probability = max(probabilities)
        blocked = injection_probability >= self._settings.block_from_injection_probability
        # Only an allowed request needs a route, so a blocked one never waits on the decision endpoint
        difficulty = None
        if pinned_route is None and not blocked:
            difficulty = await self._decisions.difficulty(last_user_text(messages))

        masked_messages = []
        masked_count = 0
        for message in messages:  # each message is masked on its own; a conversation holds a handful
            if message.role in GUARDED_ROLES:
                masked_text, count = mask_identifiers(message.content)
                masked_messages.append(message.model_copy(update={"content": masked_text}))
                masked_count += count
            else:
                masked_messages.append(message)

        needs_strong_model = difficulty is not None and difficulty >= self._settings.strong_from_difficulty
        route: Route | None = None if blocked else pinned_route or ("strong" if needs_strong_model else "small")
        decision = GateDecision(
            injection_probability=round(injection_probability, 4),
            blocked=blocked,
            route=route,
            difficulty=None if difficulty is None else round(difficulty, 3),
            masked_identifiers=masked_count,
            decision_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return decision, masked_messages

    def _model_for(self, route: Route) -> str:
        return self._settings.strong_model if route == "strong" else self._settings.small_model

    def _log(
        self,
        request: ChatRequest,
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
                masked_identifiers=decision.masked_identifiers if decision else 0,
                input_tokens=fields.get("input_tokens"),
                output_tokens=fields.get("output_tokens"),
                decision_ms=decision.decision_ms if decision else None,
                upstream_ms=fields.get("upstream_ms"),
                preview=last_user_text(request.messages)[:120],
            )
        )

    async def complete(self, request: ChatRequest) -> dict[str, Any]:
        """An OpenAI chat completion with a `gate` field describing what the gate decided."""
        sampling = request.model_extra or {}
        key = cache_key({"model": request.model, "messages": [m.model_dump() for m in request.messages], **sampling})
        cached = self._cache.get(key)
        if cached is not None:
            self._log(request, "cached", None, model=cached.get("model"))
            return {**cached, "gate": {**cached["gate"], "cached": True}}

        pinned_route = None if request.model == "auto" else request.model
        decision, masked_messages = await self.decide(request.messages, pinned_route)
        if decision.blocked:
            self._log(request, "blocked", decision)
            raise RequestBlocked(
                "The request reads as a prompt injection and was not sent to a model.",
                {"gate": decision.model_dump(), "guard": self.guard_name},
            )

        route: Route = decision.route or "small"  # set whenever the request was not blocked
        model = self._model_for(route)
        payload = {
            **sampling,
            "model": model,
            "messages": [message.model_dump() for message in masked_messages],
            "stream": False,
        }
        try:
            completion, upstream_ms = await self._upstream.chat(payload)
        except AppError:
            self._log(request, "failed", decision, model=model)
            raise
        usage = completion.get("usage") or {}
        self._log(
            request,
            "answered",
            decision,
            model=model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            upstream_ms=upstream_ms,
        )
        answer = {
            **completion,
            "gate": {**decision.model_dump(), "model": model, "upstream_ms": upstream_ms, "cached": False},
        }
        self._cache.put(key, answer)
        return answer
