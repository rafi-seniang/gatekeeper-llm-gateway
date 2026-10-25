"""What the gate decided about each request, and its log."""

from typing import Literal

from pydantic import BaseModel

Route = Literal["small", "strong"]


class GateDecision(BaseModel):
    """What the gate decided about one request, returned with every answer and logged."""

    injection_probability: float
    blocked: bool
    route: Route | None  # None when blocked: a blocked request is never routed
    difficulty: float | None  # None when the caller pinned the route
    masked_identifiers: int
    decision_ms: float


class GateEvent(BaseModel):
    request_id: str
    at: str
    outcome: Literal["answered", "cached", "blocked", "failed"]
    route: Route | None
    model: str | None
    injection_probability: float | None
    difficulty: float | None
    masked_identifiers: int
    input_tokens: int | None
    output_tokens: int | None
    decision_ms: float | None
    upstream_ms: float | None
    preview: str


class GateStats(BaseModel):
    requests: int
    answered: int
    cached: int
    blocked: int
    failed: int
    small_routed: int
    strong_routed: int
    masked_identifiers: int
    small_tokens: int
    strong_tokens: int
    median_decision_ms: float | None
    median_upstream_ms: float | None
