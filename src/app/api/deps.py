"""Dependency providers. Tests swap any of them with app.dependency_overrides."""

from typing import Annotated

from fastapi import Depends, Request

from app.controllers.gate import GateController
from app.core.resources import Resources
from app.services.gate import Gate


def get_resources(request: Request) -> Resources:
    resources: Resources = request.app.state.resources
    return resources


def get_gate(resources: Annotated[Resources, Depends(get_resources)]) -> Gate:
    return resources.gate


def get_gate_controller(gate: Annotated[Gate, Depends(get_gate)]) -> GateController:
    return GateController(gate)
