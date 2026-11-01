"""The gate: masking, cache, guards, routing, the upstream, then the tool-call check. Every request is logged."""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any, Literal

from app.engines.agent_guard import AgentGuard
from app.engines.decisions import DecisionClient
from app.engines.identifier_tagger import IdentifierFinder
from app.engines.injection import InjectionGuard
from app.engines.upstream import UpstreamClient
from app.features.gate.cache import AnswerCache, cache_key
from app.features.gate.events import EventLog
from app.features.gate.identifiers import mask_conversation, restore_completion, tool_calls_of, with_tag_rule
from app.features.gate.schemas import BlockedBy, ChatMessage, ChatRequest, GateDecision, GateEvent, Route
from app.platform.errors import AppError, RequestBlocked
from config.logging import request_id
from config.settings import Settings


def last_user_text(messages: list[ChatMessage]) -> str:
    return next(message.content or "" for message in reversed(messages) if message.role == "user")


def tool_results_after(messages: list[ChatMessage], role: str) -> list[str]:
    """Tool results since the last message of `role`. An earlier turn already checked the ones before it."""
    last = max((i for i, message in enumerate(messages) if message.role == role), default=-1)
    return [message.content or "" for message in messages[last + 1 :] if message.role == "tool"]


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
        agent_guard: AgentGuard | None = None,
    ) -> None:
        self._settings = settings
        self._finder = finder
        self._guard = guard
        self._decisions = decisions
        self._upstream = upstream
        self._cache = cache
        self._agent_guard = agent_guard
        self.events = events

    @property
    def guard_name(self) -> str:
        return self._guard.name

    async def _tool_result_probability(self, masked_messages: list[ChatMessage]) -> float | None:
        results = tool_results_after(masked_messages, "assistant")
        if self._agent_guard is None or not results:
            return None
        user_task = last_user_text(masked_messages)
        guard = self._agent_guard
        return max(await asyncio.gather(*(guard.tool_result_probability(user_task, result) for result in results)))

    async def _decide(
        self, masked_messages: list[ChatMessage], masked_count: int, pinned_route: Route | None, started: float
    ) -> GateDecision:
        guarded_texts = [message.content or "" for message in masked_messages if message.role == "tool"]
        guarded_texts.append(last_user_text(masked_messages))
        probabilities, tool_result_probability = await asyncio.gather(
            asyncio.gather(*(self._guard.probability(text) for text in guarded_texts)),
            self._tool_result_probability(masked_messages),
        )
        injection_probability = max(probabilities)
        blocked_by: BlockedBy | None = None
        if injection_probability >= self._settings.block_from_injection_probability:
            blocked_by = "injection"
        elif tool_result_probability is not None and (
            tool_result_probability >= self._settings.block_from_tool_result_probability
        ):
            blocked_by = "tool_result"
        blocked = blocked_by is not None
        # Only an allowed request needs a route, so a blocked one never waits on the decision endpoint
        difficulty = None
        if pinned_route is None and not blocked:
            difficulty = await self._decisions.difficulty(last_user_text(masked_messages))

        needs_strong_model = difficulty is not None and difficulty >= self._settings.strong_from_difficulty
        route: Route | None = None if blocked else pinned_route or ("strong" if needs_strong_model else "small")
        return GateDecision(
            injection_probability=round(injection_probability, 4),
            tool_result_probability=None if tool_result_probability is None else round(tool_result_probability, 4),
            blocked=blocked,
            blocked_by=blocked_by,
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
                blocked_by=decision.blocked_by if decision else None,
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

    async def _stop_unasked_tool_calls(
        self,
        completion: dict[str, Any],
        masked_messages: list[ChatMessage],
        decision: GateDecision,
        preview: str,
        **log_fields: Any,
    ) -> None:
        """Held before the application can run them. Jev reads the masked request and the call as the model wrote it."""
        calls = [
            call for choice in completion.get("choices", []) for call in tool_calls_of(choice.get("message") or {})
        ]
        if self._agent_guard is None or not calls:
            return
        scores = await self._agent_guard.tool_calls_scores(
            last_user_text(masked_messages), tool_results_after(masked_messages, "user"), calls
        )
        stopped = [
            {"tool": score.tool, "asked": score.asked, "scope": score.scope}
            for score in scores
            if score.asked < self._settings.tool_call_asked_below or score.scope >= self._settings.tool_call_scope_from
        ]
        if not stopped:
            return
        stopped_decision = decision.model_copy(update={"blocked": True, "blocked_by": "tool_call"})
        self._log(preview, "blocked", stopped_decision, **log_fields)
        raise RequestBlocked(
            "The model proposed a tool call the user did not ask for, and it was not returned.",
            {"gate": stopped_decision.model_dump(), "stopped_tool_calls": stopped},
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
        await self._stop_unasked_tool_calls(completion, masked_messages, decision, preview, **log_fields)
        self._log(preview, "answered", decision, **log_fields)
        answer = {
            **completion,
            "gate": {**decision.model_dump(), "model": model, "upstream_ms": upstream_ms, "cached": False},
        }
        await self._cache.put(key, answer)  # with its tags: the cache never holds a real value
        return restore_completion(answer, vault)
