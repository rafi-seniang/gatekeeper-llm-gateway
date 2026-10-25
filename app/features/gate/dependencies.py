"""How the gate feature is reached per request: the gate built at startup, then a controller around it."""

from typing import Annotated

from fastapi import Depends

from app.features.gate.controller import GateController
from app.features.gate.service import Gate
from app.platform.dependencies import get_resources
from app.platform.resources import Resources


def get_gate(resources: Annotated[Resources, Depends(get_resources)]) -> Gate:
    return resources.gate


def get_gate_controller(gate: Annotated[Gate, Depends(get_gate)]) -> GateController:
    return GateController(gate)
