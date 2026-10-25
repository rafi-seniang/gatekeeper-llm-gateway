"""Compare the injection guards on two labelled sets: the DeBERTa classifier, Jev, and the cascade of the two.

Sets:
- ours: data/eval_requests.jsonl, 46 requests written for this gate (16 injections in English and Indonesian, some
  hidden in pasted emails or tool results, and 30 benign ones, six of them support messages with identifiers).
- deepset: the test split of deepset/prompt-injections (Apache 2.0, 116 prompts in English and German), a public
  set not named among the classifier's training data. Its labels count some role-play requests as injections.
Guards:
- classifier: protectai/deberta-v3-base-prompt-injection-v2, local
- jev: TypeSafe's Jev with the gate's question and its true/false criteria (app.gate.injection.JEV_INJECTION_QUESTION)
- jev plain: the same model with a one-line question and no criteria, to show how much the wording matters
- cascade: the classifier, and Jev only for the requests the classifier would block; Jev must agree to block.
  Worked out from the two guards' answers on each request, so it costs no extra calls.
Every guard sees the text as the gate passes it on: identifiers masked first ([CARD], [PHONE], [EMAIL], [NIK]).
Each request is sent on its own, so every latency is a single request's. Needs JEV_API_KEY in the secrets file.
Run from the repository root: python scripts/eval_injection.py
"""

import asyncio
import json
import time
from pathlib import Path

import httpx
import pandas as pd

from app.core.config import get_settings
from app.core.resources import jev_client
from app.gate.decisions import INJECTION_QUESTION, DecisionClient
from app.gate.identifiers import mask_identifiers
from app.gate.injection import ClassifierGuard, InjectionGuard, JevGuard

REPOSITORY = Path(__file__).resolve().parents[1]
SETS = {
    "ours": REPOSITORY / "data" / "eval_requests.jsonl",
    "deepset": REPOSITORY / "data" / "deepset_prompt_injections_test.jsonl",
}
DEEPSET_ROWS = "https://datasets-server.huggingface.co/rows"


def download_deepset(path: Path) -> None:
    """The deepset/prompt-injections test split (Apache 2.0) through the Hugging Face dataset viewer, as JSONL."""
    rows = []
    for offset in (0, 100):  # 116 rows, served at most 100 at a time
        response = httpx.get(
            DEEPSET_ROWS,
            params={"dataset": "deepset/prompt-injections", "config": "default", "split": "test",
                    "offset": offset, "length": 100},
            timeout=60,
        )  # fmt: skip
        response.raise_for_status()
        rows += [item["row"] for item in response.json()["rows"]]
    lines = [json.dumps({"text": row["text"], "injection": bool(row["label"])}, ensure_ascii=False) for row in rows]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")


async def timed_probabilities(guard: InjectionGuard, texts: list[str]) -> tuple[list[float], list[float]]:
    probabilities, milliseconds = [], []
    for text in texts:  # one at a time, so each latency is a single request's
        started = time.perf_counter()
        probabilities.append(await guard.probability(text))
        milliseconds.append((time.perf_counter() - started) * 1000)
    return probabilities, milliseconds


def scores(name: str, labels: pd.Series, blocked: pd.Series, milliseconds: pd.Series) -> dict[str, object]:
    caught = int((blocked & labels).sum())
    false_blocks = int((blocked & ~labels).sum())
    precision = caught / max(int(blocked.sum()), 1)
    recall = caught / int(labels.sum())
    return {
        "guard": name,
        "caught": f"{caught} of {int(labels.sum())}",
        "false_blocks": f"{false_blocks} of {int((~labels).sum())}",
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 3),
        "median_ms": round(float(milliseconds.median()), 1),
    }


async def main() -> None:
    settings = get_settings()
    threshold = settings.block_from_injection_probability
    if not SETS["deepset"].is_file():
        download_deepset(SETS["deepset"])
    classifier = ClassifierGuard(settings.injection_classifier_model, settings.device)
    await classifier.probability("warm-up")  # loads weights and compiles kernels outside the timing
    results: dict[str, list[dict[str, object]]] = {}
    per_request = []
    async with jev_client(settings) as http:
        jev_decisions = DecisionClient(http, settings.jev_model, "systemone")
        guards: dict[str, InjectionGuard] = {
            "classifier": classifier,
            "jev": JevGuard(jev_decisions),
            "jev plain": JevGuard(jev_decisions, INJECTION_QUESTION),
        }
        for set_name, path in SETS.items():  # two sets, each through every guard
            requests = pd.read_json(path, lines=True)
            labels = requests["injection"].astype(bool)
            frame = pd.DataFrame({"set": set_name, "text": requests["text"].str.slice(0, 90), "injection": labels})
            masked = [mask_identifiers(text)[0] for text in requests["text"]]
            for guard_name, guard in guards.items():
                probabilities, milliseconds = await timed_probabilities(guard, masked)
                frame[f"{guard_name} p"] = probabilities
                frame[f"{guard_name} ms"] = milliseconds
            escalated = frame["classifier p"] >= threshold
            frame["cascade p"] = frame["classifier p"].where(~escalated, frame["jev p"])
            frame["cascade ms"] = frame["classifier ms"] + frame["jev ms"].where(escalated, 0)
            results[set_name] = [
                scores(name, labels, frame[f"{name} p"] >= threshold, frame[f"{name} ms"])
                for name in ["classifier", "jev", "jev plain", "cascade"]
            ]
            results[set_name][-1]["share_sent_to_jev"] = round(float(escalated.mean()), 3)
            per_request.append(frame)
            print(
                f"\n{set_name}: {len(requests)} requests, {int(labels.sum())} injections, blocking at p >= {threshold}"
            )
            print(pd.DataFrame(results[set_name]).to_string(index=False))

    every_request = pd.concat(per_request, ignore_index=True)
    probability_columns = [column for column in every_request if column.endswith(" p")]
    wrong = every_request[
        (every_request[probability_columns] >= threshold).ne(every_request["injection"], axis=0).any(axis=1)
    ]
    print("\nRequests at least one guard got wrong:")
    print(wrong[["set", "injection", *probability_columns, "text"]].round(2).to_string(index=False))

    output = REPOSITORY / "results"
    output.mkdir(exist_ok=True)
    (output / "eval_injection.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    every_request.round(3).to_csv(output / "eval_injection_requests.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main())
