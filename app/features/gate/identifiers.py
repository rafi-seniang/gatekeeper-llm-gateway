"""Replaces personal data with tags before anything leaves the machine, and puts the real values back in the answer.

The tags and their values live in one request's vault, in memory. They are never stored, logged or cached.
"""

import asyncio
import json
import re
from collections import Counter
from typing import Any

from app.engines.identifier_tagger import IdentifierFinder, IdentifierSpan
from app.features.gate.schemas import ChatMessage

MASKED_ROLES = {"user", "tool", "assistant"}  # an assistant turn in the history holds the values the gate put back
TAG_RULE = (
    "Bracketed tags such as [NAME_1] stand for real values. Copy a tag exactly when you need its value. "
    "Never write a tag that is not in the conversation. When a value is unknown, leave it out."
)


class IdentifierVault:
    def __init__(self) -> None:
        self._tag_of: dict[tuple[str, str], str] = {}
        self._value_of: dict[str, str] = {}
        self._label_counts: Counter[str] = Counter()
        self.replaced = 0

    def tag_for(self, label: str, value: str) -> str:
        """The same value always gets the same tag, so the model can tell people apart."""
        self.replaced += 1
        key = (label, value)
        if key not in self._tag_of:
            self._label_counts[label] += 1
            tag = f"[{label}_{self._label_counts[label]}]"
            self._tag_of[key] = tag
            self._value_of[tag] = value
        return self._tag_of[key]

    def restore(self, text: str, as_json_string: bool = False) -> str:
        """Known tags only: a tag the model invented stays as written."""
        if not self._value_of:
            return text

        def value(match: re.Match[str]) -> str:
            found = self._value_of[match.group(0)]
            return json.dumps(found, ensure_ascii=False)[1:-1] if as_json_string else found

        return re.sub("|".join(re.escape(tag) for tag in self._value_of), value, text)


def mask_text(text: str, spans: list[IdentifierSpan], vault: IdentifierVault) -> str:
    pieces: list[str] = []
    position = 0
    for span in spans:  # spans run left to right, and each tag depends on the ones before it
        pieces += [text[position : span.start], vault.tag_for(span.label, text[span.start : span.end])]
        position = span.end
    return "".join([*pieces, text[position:]])


def tool_calls_of(message: ChatMessage | dict[str, Any]) -> list[dict[str, Any]]:
    extra = message if isinstance(message, dict) else (message.model_extra or {})
    calls: list[dict[str, Any]] = extra.get("tool_calls") or []
    return calls


def _with_arguments(call: dict[str, Any], arguments: str) -> dict[str, Any]:
    return {**call, "function": {**call["function"], "arguments": arguments}}


async def mask_conversation(
    messages: list[ChatMessage], finder: IdentifierFinder
) -> tuple[list[ChatMessage], IdentifierVault]:
    """User, tool and assistant text masked, with one vault across the conversation. The app's own text is untouched."""
    contents = [(m.content or "") if m.role in MASKED_ROLES else "" for m in messages]
    arguments = [[str(c["function"].get("arguments") or "") for c in tool_calls_of(m)] for m in messages]
    content_spans, argument_spans = await asyncio.gather(
        asyncio.gather(*(finder.find(text) for text in contents)),
        asyncio.gather(*(asyncio.gather(*(finder.find(text) for text in texts)) for texts in arguments)),
    )
    vault = IdentifierVault()
    masked: list[ChatMessage] = []
    for index, message in enumerate(messages):  # in conversation order, so a value's tag does not depend on timing
        update: dict[str, Any] = {}
        if message.role in MASKED_ROLES and message.content:
            update["content"] = mask_text(message.content, content_spans[index], vault)
        if arguments[index]:
            update["tool_calls"] = [
                _with_arguments(call, mask_text(text, spans, vault))
                for call, text, spans in zip(
                    tool_calls_of(message), arguments[index], argument_spans[index], strict=True
                )
            ]
        masked.append(message.model_copy(update=update))
    return masked, vault


def with_tag_rule(messages: list[dict[str, Any]], vault: IdentifierVault) -> list[dict[str, Any]]:
    """The rule goes after the app's own leading system messages, and only when something was masked."""
    if not vault.replaced:
        return messages
    position = next((i for i, m in enumerate(messages) if m["role"] != "system"), len(messages))
    return [*messages[:position], {"role": "system", "content": TAG_RULE}, *messages[position:]]


def restore_completion(completion: dict[str, Any], vault: IdentifierVault) -> dict[str, Any]:
    """A copy of the completion with real values in each reply and tool call. The stored answer keeps its tags."""

    def restored_message(message: dict[str, Any]) -> dict[str, Any]:
        restored = dict(message)
        if isinstance(message.get("content"), str):
            restored["content"] = vault.restore(message["content"])
        if tool_calls_of(message):
            restored["tool_calls"] = [
                _with_arguments(call, vault.restore(str(call["function"].get("arguments") or ""), as_json_string=True))
                for call in tool_calls_of(message)
            ]
        return restored

    choices = [{**c, "message": restored_message(c.get("message") or {})} for c in completion.get("choices", [])]
    return {**completion, "choices": choices}
