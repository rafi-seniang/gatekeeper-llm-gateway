"""The OpenAI chat format the gate accepts and returns, and what the gate decided about each request."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BlockedBy = Literal["injection", "off_topic", "tool_result", "tool_call"]


class GateDecision(BaseModel):
    """What the gate decided about one request, returned with every answer and logged."""

    injection_probability: float
    off_topic_probability: float = 0.0
    tool_result_probability: float | None = None  # None when the agent guard did not run
    blocked: bool
    blocked_by: BlockedBy | None = None
    masked_identifiers: int
    decision_ms: float


class GateEvent(BaseModel):
    request_id: str
    at: str
    outcome: Literal["answered", "cached", "blocked", "failed"]
    blocked_by: BlockedBy | None = None
    model: str | None
    injection_probability: float | None
    off_topic_probability: float | None = None
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
    off_topic_blocked: int = 0
    failed: int
    masked_identifiers: int
    tokens: int  # input and output of the answered requests
    median_decision_ms: float | None
    median_upstream_ms: float | None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")  # name, tool_call_id and the like pass through untouched

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | None = Field(default=None, max_length=32_000)

    @model_validator(mode="after")
    def only_an_assistant_may_have_no_text(self) -> "ChatMessage":
        if self.content is None and self.role != "assistant":
            raise ValueError("only an assistant message may leave content empty, when it holds tool calls")
        return self


class Conversation(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def has_a_user_message(self) -> "Conversation":
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("messages must contain at least one user message")
        return self


class ChatRequest(Conversation):
    model_config = ConfigDict(extra="allow")  # sampling options pass through to the upstream

    # Accepted so any OpenAI client works unchanged. The gate answers with its one configured model
    model: str = Field(default="auto", max_length=200)
    stream: Literal[False] = False  # streaming is not proxied yet: the gate answers once the upstream has
    scope: str | None = Field(default=None, max_length=600)


class CheckRequest(Conversation):
    """A request to judge without sending it anywhere."""

    scope: str | None = Field(default=None, max_length=600)


ChatCompletion = dict[str, Any]
