"""The OpenAI-compatible entry point and the gate's own views: check without sending, stats, recent events."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_gate
from app.gate.service import Gate
from app.schemas import ChatRequest, CheckRequest, GateDecision, GateEvent, GateStats

router = APIRouter(tags=["gate"])


@router.post("/chat/completions")
async def chat_completions(body: ChatRequest, gate: Annotated[Gate, Depends(get_gate)]) -> dict[str, Any]:
    return await gate.complete(body)


@router.post("/gate/check")
async def check(body: CheckRequest, gate: Annotated[Gate, Depends(get_gate)]) -> GateDecision:
    decision, _ = await gate.decide(body.messages, pinned_route=None)
    return decision


@router.get("/gate/stats")
async def stats(gate: Annotated[Gate, Depends(get_gate)]) -> GateStats:
    return gate.events.stats()


@router.get("/gate/events")
async def events(
    gate: Annotated[Gate, Depends(get_gate)], limit: Annotated[int, Query(ge=1, le=500)] = 50
) -> list[GateEvent]:
    return gate.events.recent(limit)
