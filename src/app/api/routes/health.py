"""Liveness touches nothing (the lobby polls it); /meta describes the service for the lab frontend."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_gate
from app.gate.service import Gate
from app.schemas import ServiceMeta

router = APIRouter(tags=["health"])
VERSION = "0.1.0"


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/meta")
async def meta(gate: Annotated[Gate, Depends(get_gate)]) -> ServiceMeta:
    return ServiceMeta(
        id="gatekeeper",
        name="Gatekeeper",
        summary=(
            "An OpenAI-compatible gateway: blocks prompt injection, masks identifiers, "
            "routes to a small or a strong model."
        ),
        version=VERSION,
        engine=gate.guard_name,
        own_ui_path="/docs",
        docs_path="/docs",
    )
