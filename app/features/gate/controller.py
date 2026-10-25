"""Turns the gate's HTTP requests into calls on the gate and shapes what goes back."""

from app.features.gate.schemas import ChatCompletion, ChatRequest, CheckRequest, GateDecision, GateEvent, GateStats
from app.features.gate.service import Gate


class GateController:
    def __init__(self, gate: Gate) -> None:
        self._gate = gate

    async def complete(self, body: ChatRequest) -> ChatCompletion:
        return await self._gate.complete(body)

    async def check(self, body: CheckRequest) -> GateDecision:
        decision, _ = await self._gate.decide(body.messages, pinned_route=None)
        return decision

    def stats(self) -> GateStats:
        return self._gate.events.stats()

    def recent_events(self, limit: int) -> list[GateEvent]:
        return self._gate.events.recent(limit)
