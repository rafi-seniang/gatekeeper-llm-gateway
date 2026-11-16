"""The OpenAI-compatible entry point and the gate's own views: check without sending, stats, recent events."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.features.gate.controller import GateController
from app.features.gate.dependencies import get_gate_controller
from app.features.gate.schemas import (
    ChatCompletion,
    ChatRequest,
    CheckRequest,
    GateDecision,
    GateEvent,
    GateStats,
    NamedScope,
)
from app.platform.dependencies import enforce_daily_gate_limit

router = APIRouter(tags=["gate"])
Controller = Annotated[GateController, Depends(get_gate_controller)]


@router.post("/chat/completions", dependencies=[Depends(enforce_daily_gate_limit)])
async def chat_completions(body: ChatRequest, controller: Controller) -> ChatCompletion:
    return await controller.complete(body)


@router.post("/gate/check", dependencies=[Depends(enforce_daily_gate_limit)])
async def check(body: CheckRequest, controller: Controller) -> GateDecision:
    return await controller.check(body)


@router.get("/gate/stats")
async def stats(controller: Controller) -> GateStats:
    return controller.stats()


@router.get("/gate/events")
async def events(
    controller: Controller,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    scope_name: Annotated[str | None, Query(max_length=40, description="Only the checks of one named scope")] = None,
) -> list[GateEvent]:
    return controller.recent_events(limit, scope_name)


@router.get("/gate/scopes")
async def scopes(controller: Controller) -> list[NamedScope]:
    """The scopes apps use by name, each with the exact questions the guard asks."""
    return controller.named_scopes()
