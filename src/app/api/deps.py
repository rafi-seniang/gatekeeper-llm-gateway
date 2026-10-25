"""Dependency providers. Tests swap any of them with app.dependency_overrides."""

from fastapi import Request

from app.core.resources import Resources
from app.gate.service import Gate


def get_resources(request: Request) -> Resources:
    resources: Resources = request.app.state.resources
    return resources


def get_gate(request: Request) -> Gate:
    return get_resources(request).gate
