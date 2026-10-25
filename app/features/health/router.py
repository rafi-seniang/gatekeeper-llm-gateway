"""Liveness touches nothing (the lobby polls it). /meta describes the service for the lab frontend."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app import VERSION
from app.features.health.schemas import ServiceMeta
from app.platform.dependencies import get_resources
from app.platform.resources import Resources

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/meta")
async def meta(resources: Annotated[Resources, Depends(get_resources)]) -> ServiceMeta:
    return ServiceMeta(
        id="gatekeeper",
        name="Gatekeeper",
        summary=(
            "An OpenAI-compatible gateway: blocks prompt injection, masks identifiers, "
            "routes to a small or a strong model."
        ),
        version=VERSION,
        engine=resources.gate.guard_name,
        own_ui_path="/docs",
        docs_path="/docs",
    )
