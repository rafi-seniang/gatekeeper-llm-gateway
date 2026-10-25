"""The OpenAI chat format the gate accepts and returns."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
