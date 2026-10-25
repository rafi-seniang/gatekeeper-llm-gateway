"""Measure the gate's two decisions on data/eval_requests.jsonl.

Injection: the classifier and the decision endpoint side by side (precision, recall, false blocks, latency).
Routing: the decision endpoint's difficulty score against the labelled level (benign requests only).
Run from the repository root with the decision endpoint up: python scripts/eval_gate.py
"""

import asyncio
import json
import time
from pathlib import Path

import httpx
import pandas as pd

from app.core.config import get_settings
from app.engines.decisions import DecisionClient
from app.engines.injection import ClassifierGuard, DecisionGuard, InjectionGuard

DATA = Path(__file__).resolve().parents[1] / "data" / "eval_requests.jsonl"


async def timed_probabilities(guard: InjectionGuard, texts: list[str]) -> tuple[list[float], list[float]]:
    probabilities, milliseconds = [], []
    for text in texts:  # one at a time, so each latency is a single request's
        started = time.perf_counter()
        probabilities.append(await guard.probability(text))
        milliseconds.append((time.perf_counter() - started) * 1000)
    return probabilities, milliseconds


def injection_scores(
    name: str, labels: pd.Series, probabilities: list[float], milliseconds: list[float], threshold: float
) -> dict[str, object]:
    predicted = pd.Series(probabilities, index=labels.index) >= threshold
    true_positive = int((predicted & labels).sum())
    return {
        "guard": name,
        "caught": f"{true_positive} of {int(labels.sum())}",
        "false_blocks": f"{int((predicted & ~labels).sum())} of {int((~labels).sum())}",
        "precision": round(true_positive / max(int(predicted.sum()), 1), 3),
        "recall": round(true_positive / int(labels.sum()), 3),
        "median_ms": round(float(pd.Series(milliseconds).median()), 1),
    }


async def main() -> None:
    settings = get_settings()
    requests = pd.read_json(DATA, lines=True)
    labels = requests["injection"].astype(bool)
    async with httpx.AsyncClient(
        base_url=settings.decision_base_url,
        headers={"Authorization": f"Bearer {settings.decision_api_key.get_secret_value()}"},
        timeout=settings.decision_timeout_seconds,
    ) as http:
        decisions = DecisionClient(http, settings.decision_model, settings.decision_protocol)
        guards: list[InjectionGuard] = [
            ClassifierGuard(settings.injection_classifier_model, settings.device),
            DecisionGuard(decisions),
        ]
        rows = []
        per_request = requests[["id", "injection", "text"]].copy()
        for guard in guards:  # two guards, compared on the same requests
            await guard.probability("warm-up")  # loads weights and compiles kernels outside the timing
            probabilities, milliseconds = await timed_probabilities(guard, requests["text"].tolist())
            rows.append(
                injection_scores(
                    guard.name, labels, probabilities, milliseconds, settings.block_from_injection_probability
                )
            )
            per_request[guard.name] = [round(probability, 3) for probability in probabilities]

        benign = requests[~labels].copy()
        benign["difficulty_score"] = [
            await decisions.difficulty(text) for text in benign["text"]
        ]  # sequential: one GPU
    print(f"Injection guard, blocking at probability >= {settings.block_from_injection_probability}\n")
    print(pd.DataFrame(rows).to_string(index=False))

    benign["predicted_route"] = (benign["difficulty_score"] >= settings.strong_from_difficulty).map(
        {True: "strong", False: "small"}
    )
    benign["labelled_route"] = (benign["difficulty"] >= 2).map({True: "strong", False: "small"})
    route_accuracy = (benign["predicted_route"] == benign["labelled_route"]).mean()
    print(
        f"\nRouting on {len(benign)} benign requests (strong when labelled level 2, "
        f"predicted strong from score {settings.strong_from_difficulty}):"
    )
    print(f"route accuracy {route_accuracy:.3f}, small model share {(benign['predicted_route'] == 'small').mean():.3f}")
    print(benign.groupby("difficulty")["difficulty_score"].agg(["mean", "min", "max"]).round(2).to_string())

    mistakes = per_request[
        (per_request.iloc[:, 3:] >= settings.block_from_injection_probability)
        .ne(per_request["injection"], axis=0)
        .any(axis=1)
    ]
    print("\nRequests at least one guard got wrong:")
    print(mistakes.assign(text=mistakes["text"].str.slice(0, 70)).to_string(index=False))
    output = DATA.parent.parent / "results" / "eval_gate.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"injection": rows, "route_accuracy": round(float(route_accuracy), 3)}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
