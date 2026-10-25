"""Turns the gate's HTTP requests into calls on the gate and shapes what goes back."""

from app import VERSION
from app.schemas.chat import ChatCompletion, ChatRequest, CheckRequest
from app.schemas.gate import GateDecision, GateEvent, GateStats
from app.schemas.meta import ServiceMeta
from app.services.gate import Gate


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

    def meta(self) -> ServiceMeta:
        return ServiceMeta(
            id="gatekeeper",
            name="Gatekeeper",
            summary=(
                "An OpenAI-compatible gateway: blocks prompt injection, masks identifiers, "
                "routes to a small or a strong model."
            ),
            version=VERSION,
            engine=self._gate.guard_name,
            own_ui_path="/docs",
            docs_path="/docs",
        )
