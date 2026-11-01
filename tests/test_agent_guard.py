from typing import Any

import httpx

from app.engines.agent_guard import AgentGuard
from app.engines.decisions import DecisionClient


def call(tool: str, arguments: str) -> dict[str, Any]:
    return {"id": "call_1", "type": "function", "function": {"name": tool, "arguments": arguments}}


async def test_the_guard_sends_jev_the_questions_and_applies_the_weights() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = httpx.Response(200, content=request.content).json()
        sent.append(body)
        return httpx.Response(200, json={"answers": {name: {"noul": 1.0} for name in body["questions"]}})

    async with httpx.AsyncClient(base_url="https://jev", transport=httpx.MockTransport(handler)) as http:
        guard = AgentGuard(DecisionClient(http, "jev-1.14.0"))
        probability = await guard.tool_result_probability("Summarise", "Some text")
        scores = await guard.tool_call_scores("Send it", ["a result"], call("send", '{"to": "[EMAIL_1]"}'))
    assert probability > 0.99  # every question at 1.0
    assert sent[0]["state"] == {"user_task": "Summarise", "tool_result": "Some text"}
    assert sent[1]["state"]["proposed_action"] == {"tool": "send", "arguments": {"to": "[EMAIL_1]"}}
    assert sent[1]["state"]["tool_result_seen"] == "a result"
    assert (scores.asked, scores.scope) == (1.0, 1.0)


async def test_broken_arguments_still_reach_the_check() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = httpx.Response(200, content=request.content).json()
        sent.append(body)
        return httpx.Response(200, json={"answers": {name: {"noul": 0.2} for name in body["questions"]}})

    async with httpx.AsyncClient(base_url="https://jev", transport=httpx.MockTransport(handler)) as http:
        await AgentGuard(DecisionClient(http, "jev-1.14.0")).tool_call_scores("Hi", [], call("send", "{broken"))
    assert sent[0]["state"]["proposed_action"]["arguments"] == "{broken"
    assert "tool_result_seen" not in sent[0]["state"]
