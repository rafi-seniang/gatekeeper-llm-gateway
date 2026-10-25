"""The OpenAI chat format the gate accepts and returns, and what the gate decided about each request."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")  # name, tool_call_id and the like pass through untouched

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str = Field(max_length=32_000)


class Conversation(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def has_a_user_message(self) -> "Conversation":
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("messages must contain at least one user message")
        return self


class ChatRequest(Conversation):
    model_config = ConfigDict(extra="allow")  # sampling options pass through to the upstream

    # "auto" lets the gate choose; "small" or "strong" pins a route
    model: Literal["auto", "small", "strong"] = "auto"
    stream: Literal[False] = False  # streaming is not proxied yet: the gate answers once the upstream has


class CheckRequest(Conversation):
    """A request to judge without sending it anywhere."""


ChatCompletion = dict[str, Any]
