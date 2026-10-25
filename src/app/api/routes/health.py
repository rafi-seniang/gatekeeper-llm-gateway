"""Liveness touches nothing (the lobby polls it). /meta describes the service for the lab frontend."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_gate_controller
from app.controllers.gate import GateController
from app.schemas.meta import ServiceMeta

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/meta")
async def meta(controller: Annotated[GateController, Depends(get_gate_controller)]) -> ServiceMeta:
    return controller.meta()
