"""The OpenAI-compatible entry point and the gate's own views: check without sending, stats, recent events."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_gate_controller
from app.controllers.gate import GateController
from app.schemas.chat import ChatCompletion, ChatRequest, CheckRequest
from app.schemas.gate import GateDecision, GateEvent, GateStats

router = APIRouter(tags=["gate"])
Controller = Annotated[GateController, Depends(get_gate_controller)]


@router.post("/chat/completions")
async def chat_completions(body: ChatRequest, controller: Controller) -> ChatCompletion:
    return await controller.complete(body)


@router.post("/gate/check")
async def check(body: CheckRequest, controller: Controller) -> GateDecision:
    return await controller.check(body)


@router.get("/gate/stats")
async def stats(controller: Controller) -> GateStats:
    return controller.stats()


@router.get("/gate/events")
async def events(controller: Controller, limit: Annotated[int, Query(ge=1, le=500)] = 50) -> list[GateEvent]:
    return controller.recent_events(limit)
