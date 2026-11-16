"""Liveness touches nothing (the lobby polls it). /meta describes the service for the lab frontend."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app import VERSION
from app.features.health.schemas import ServiceMeta
from app.platform.dependencies import get_resources, get_settings
from app.platform.resources import Resources
from config.settings import Settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/meta")
async def meta(
    resources: Annotated[Resources, Depends(get_resources)], settings: Annotated[Settings, Depends(get_settings)]
) -> ServiceMeta:
    return ServiceMeta(
        id="gatekeeper",
        name="Gatekeeper",
        summary=(
            "Bilingual (EN/ID) LLM security gateway: blocks prompt injection, out-of-topic requests, "
            "and malicious tool results."
        ),
        version=VERSION,
        engine=resources.gate.guard_name,
        own_ui_path="/docs",
        docs_path="/docs",
        block_from_injection_probability=settings.block_from_injection_probability,
        block_from_off_topic_probability=settings.block_from_off_topic_probability,
    )
