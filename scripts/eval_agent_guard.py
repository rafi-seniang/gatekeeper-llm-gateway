"""The agent guard through the real tagger and real Jev, on hand-written tool calls and masked as the gate masks them.

Each case swaps one detail of a call (recipient, amount, path, access) through text in a tool result, or is a legitimate
call that looks aggressive. Costs about $0.001 in Jev calls.

Usage (from the repository root): PYTHONPATH=. python scripts/eval_agent_guard.py
"""

import asyncio
import json
from pathlib import Path

from app.engines.agent_guard import AgentGuard
from app.engines.decisions import DecisionClient
from app.engines.identifier_tagger import OnnxIdentifierTagger
from app.features.gate.identifiers import IdentifierVault, mask_text
from app.platform.resources import jev_client
from config.settings import get_settings

REPOSITORY = Path(__file__).parent.parent


async def main() -> None:
    settings = get_settings()
    tagger = OnnxIdentifierTagger(
        settings.identifier_model,
        settings.identifier_model_revision,
        settings.identifier_span_threshold,
        settings.identifier_threads,
    )
    lines = (REPOSITORY / "data" / "eval_tool_calls.jsonl").read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]

    def masked(case: dict) -> tuple[str, str, dict]:
        vault = IdentifierVault()  # one per conversation, so a value keeps its tag across the three texts
        request = mask_text(case["user_request"], tagger.find_now(case["user_request"]), vault)
        result = mask_text(case["tool_result_seen"], tagger.find_now(case["tool_result_seen"]), vault)
        arguments = json.dumps(case["proposed_action"]["arguments"], ensure_ascii=False)
        call = {
            "function": {
                "name": case["proposed_action"]["tool"],
                "arguments": mask_text(arguments, tagger.find_now(arguments), vault),
            }
        }
        return request, result, call

    forms = [masked(case) for case in cases]
    async with jev_client(settings) as http:
        guard = AgentGuard(DecisionClient(http, settings.jev_model, "systemone"))
        scores = await asyncio.gather(*(guard.tool_call_scores(r, [t], c) for r, t, c in forms))

    stopped = [s.asked < settings.tool_call_asked_below or s.scope >= settings.tool_call_scope_from for s in scores]
    attacks = [case["label"] == "attack" for case in cases]
    print("attacks stopped", sum(s and a for s, a in zip(stopped, attacks, strict=True)), "of", sum(attacks))
    print(
        "legitimate stopped",
        sum(s and not a for s, a in zip(stopped, attacks, strict=True)),
        "of",
        attacks.count(False),
    )
    for case, form, score, stop in zip(cases, forms, scores, stopped, strict=True):
        if stop != (case["label"] == "attack"):
            print("wrong:", case["id"], case["label"], score, form[2]["function"]["arguments"][:90])


if __name__ == "__main__":
    asyncio.run(main())
