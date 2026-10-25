"""Compare the injection guards on labelled sets: the DeBERTa classifier, Jev, Laya, and the cascade of DeBERTa and Jev.

Sets:
- ours: data/eval_requests.jsonl, 46 requests written for this gate (16 injections in English and Indonesian, some
  hidden in pasted emails or tool results, and 30 benign ones, six of them support messages with identifiers).
- deepset: the test split of deepset/prompt-injections (Apache 2.0, 116 prompts in English and German), a public
  set not named among the classifier's training data. Its labels count some role-play requests as injections.
- qualifire: rogue-security/prompt-injections-benchmark (formerly qualifire/, CC BY-NC 4.0, 5,000 prompts labelled
  jailbreak or benign). Gated: it is used only when HF_TOKEN in the secrets file belongs to an account that accepted
  its terms, and it is kept out of git.
Guards:
- classifier: protectai/deberta-v3-base-prompt-injection-v2, local
- jev: TypeSafe's Jev with the gate's question and its criteria (app.engines.injection.JEV_INJECTION_QUESTION)
- jev plain: the same model with a one-line question and no criteria, to show how much the wording matters
  (not run on qualifire, to keep its cost to one Jev call per prompt)
- laya: Convai Innovations' open Laya (laya.Router, local), asked the same question as jev
- laya preset: Laya with its own prompt_injection question from laya.guard_questions()
- cascade: the classifier, and Jev only for the requests the classifier would block; Jev must agree to block.
  Worked out from the two guards' answers on each request, so it costs no extra calls.
Every guard sees the text as the gate passes it on: identifiers masked first ([CARD], [PHONE], [EMAIL], [NIK]).
In the small sets each request is sent on its own, so every latency is a single request's. The large set runs at full
speed (Laya in batches, 8 Jev requests in flight) and reports no latency. The question wording is the one fixed before
the first run, and the threshold table shows what a stricter block threshold trades. Needs JEV_API_KEY.
Run from the repository root: python scripts/eval_injection.py [set ...]   (default: every set available)
"""

import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from dotenv import dotenv_values

from app.core.config import get_settings
from app.core.errors import UpstreamUnavailable
from app.core.resources import jev_client
from app.engines.decisions import INJECTION_QUESTION, DecisionClient
from app.engines.injection import JEV_INJECTION_QUESTION, ClassifierGuard, InjectionGuard, JevGuard
from app.services.identifiers import mask_identifiers

REPOSITORY = Path(__file__).resolve().parents[1]
SECRETS = REPOSITORY.parents[1] / "secrets" / "secrets.env"
SETS = {
    "ours": REPOSITORY / "data" / "eval_requests.jsonl",
    "deepset": REPOSITORY / "data" / "deepset_prompt_injections_test.jsonl",
    "qualifire": REPOSITORY / "data" / "qualifire_prompt_injections_benchmark.jsonl",
}
DATASET_ROWS = "https://datasets-server.huggingface.co/rows"
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
SKIPPED_GUARDS = {"qualifire": {"jev plain"}}
BULK_SETS = {"qualifire"}  # run for throughput, not latency
JEV_REQUESTS_IN_FLIGHT = 8
LAYA_BATCH_SIZE = 32
THRESHOLDS = [0.5, 0.7, 0.9]
JEV_PRICE_PER_MILLION_INPUT_TOKENS = 0.042


def download_rows(dataset: str, total: int, token: str | None = None) -> list[dict[str, Any]]:
    """Every row of a dataset's test split through the Hugging Face dataset viewer, 100 at a time."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    rows = []
    for offset in range(0, total, 100):  # pages in order, one script run
        params = {"dataset": dataset, "config": "default", "split": "test", "offset": offset, "length": 100}
        time.sleep(1)  # the viewer rate-limits bursts of pages
        response = httpx.get(DATASET_ROWS, params=params, headers=headers, timeout=60)
        for attempt in range(5):  # 429 and 5xx come and go: wait as told, or 2, 4, 8, 16, 32 seconds, and ask again
            if response.status_code not in RETRYABLE_STATUSES:
                break
            time.sleep(float(response.headers.get("Retry-After", 2 ** (attempt + 1))))
            response = httpx.get(DATASET_ROWS, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        rows += [item["row"] for item in response.json()["rows"]]
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")


def download_set(name: str, path: Path) -> bool:
    """Fetches a public set on its first use. False when it cannot be read (the gated one without access)."""
    if name == "deepset":
        rows = download_rows("deepset/prompt-injections", 116)
        write_jsonl(path, [{"text": row["text"], "injection": bool(row["label"])} for row in rows])
        return True
    if name == "qualifire":
        token = dotenv_values(SECRETS).get("HF_TOKEN")
        try:
            rows = download_rows("rogue-security/prompt-injections-benchmark", 5000, token)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            reason = (
                "the dataset viewer is busy or failing, try again later"
                if status in RETRYABLE_STATUSES
                else "accept the terms at https://huggingface.co/datasets/rogue-security/prompt-injections-benchmark "
                "and let HF_TOKEN read public gated repositories"
            )
            print(f"qualifire skipped: the dataset viewer answered {status}, {reason}.")
            return False
        write_jsonl(path, [{"text": row["text"], "injection": row["label"] == "jailbreak"} for row in rows])
        return True
    return False


class LayaGuard:
    """Laya through laya.Router: the English checkpoint for English text, the multilingual one otherwise."""

    def __init__(self, router: Any, question: dict[str, Any], name: str) -> None:
        self.name = name
        self._router = router
        self._question = question

    async def probability(self, text: str) -> float:
        result = await asyncio.to_thread(self._router.predict, text, {"injection": self._question})
        return float(result["answers"]["injection"]["noul"])

    async def probabilities(self, texts: list[str]) -> list[float]:
        requests = [{"state": text, "questions": {"injection": self._question}} for text in texts]
        results = await asyncio.to_thread(self._router.predict_batch, requests, LAYA_BATCH_SIZE)
        return [float(result["answers"]["injection"]["noul"]) for result in results]


class TokenCount:
    """Adds up the input tokens Jev reports, so the run states what it cost."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.requests = 0

    async def __call__(self, response: httpx.Response) -> None:
        await response.aread()
        if response.is_success:
            self.input_tokens += int(response.json().get("usage", {}).get("input_tokens", 0))
            self.requests += 1


async def timed_probabilities(guard: InjectionGuard, texts: list[str]) -> tuple[list[float], list[float]]:
    probabilities, milliseconds = [], []
    for text in texts:  # one at a time, so each latency is a single request's
        started = time.perf_counter()
        probabilities.append(await guard.probability(text))
        milliseconds.append((time.perf_counter() - started) * 1000)
    return probabilities, milliseconds


async def jev_probability_with_retries(guard: InjectionGuard, text: str) -> float:
    """Jev answers 429 or 529 when busy. Waits 1, 2, 4 and 8 seconds before giving up."""
    for attempt in range(4):  # each retry waits on the one before it
        try:
            return await guard.probability(text)
        except UpstreamUnavailable:
            await asyncio.sleep(2**attempt)
    return await guard.probability(text)


async def bulk_probabilities(guard: InjectionGuard, texts: list[str]) -> tuple[list[float], list[float]]:
    """The large set at full speed: Laya in batches, Jev with requests in flight, the classifier in turn.

    Latency is measured on the small sets, so here every latency is NaN.
    """
    if isinstance(guard, LayaGuard):
        probabilities = await guard.probabilities(texts)
    elif isinstance(guard, JevGuard):
        request_slots = asyncio.Semaphore(JEV_REQUESTS_IN_FLIGHT)

        async def ask(text: str) -> float:
            async with request_slots:
                return await jev_probability_with_retries(guard, text)

        probabilities = list(await asyncio.gather(*(ask(text) for text in texts)))
    else:
        probabilities, _ = await timed_probabilities(guard, texts)
    return probabilities, [math.nan] * len(texts)


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
        "median_ms": None if milliseconds.isna().all() else round(float(milliseconds.median()), 1),
    }


def threshold_rows(name: str, labels: pd.Series, probabilities: pd.Series) -> list[dict[str, object]]:
    """Catches and false blocks at each block threshold: what a stricter gate gives up."""
    return [
        {
            "guard": name,
            "block_from": threshold,
            "caught": int(((probabilities >= threshold) & labels).sum()),
            "false_blocks": int(((probabilities >= threshold) & ~labels).sum()),
        }
        for threshold in THRESHOLDS
    ]


async def main(set_names: list[str]) -> None:
    settings = get_settings()
    threshold = settings.block_from_injection_probability
    sets = {}
    for name in set_names:
        if SETS[name].is_file() or download_set(name, SETS[name]):
            sets[name] = SETS[name]
    if not sets:
        raise SystemExit("No set to evaluate.")

    from laya import Router, guard_questions  # heavy import, after the sets are known

    classifier = ClassifierGuard(settings.injection_classifier_model, settings.device)
    router = Router(device=settings.device)
    router.preload(["english", "multilingual"])
    await classifier.probability("warm-up")  # loads weights and compiles kernels outside the timing
    router.predict("warm-up", {"injection": JEV_INJECTION_QUESTION})
    results: dict[str, Any] = {}
    per_request = []
    tokens = TokenCount()
    async with jev_client(settings) as http:
        http.event_hooks["response"].append(tokens)
        jev_decisions = DecisionClient(http, settings.jev_model, "systemone")
        guards: dict[str, InjectionGuard] = {
            "classifier": classifier,
            "jev": JevGuard(jev_decisions),
            "jev plain": JevGuard(jev_decisions, INJECTION_QUESTION),
            "laya": LayaGuard(router, JEV_INJECTION_QUESTION, "laya"),
            "laya preset": LayaGuard(router, guard_questions()["prompt_injection"], "laya preset"),
        }
        for set_name, path in sets.items():  # up to three sets, each through every guard it is not skipped for
            requests = pd.read_json(path, lines=True)
            labels = requests["injection"].astype(bool)
            frame = pd.DataFrame({"set": set_name, "text": requests["text"].str.slice(0, 90), "injection": labels})
            masked = [mask_identifiers(text)[0] for text in requests["text"]]
            names = [name for name in guards if name not in SKIPPED_GUARDS.get(set_name, set())]
            tokens_before = tokens.input_tokens
            for guard_name in names:
                started = time.perf_counter()
                measure = bulk_probabilities if set_name in BULK_SETS else timed_probabilities
                probabilities, milliseconds = await measure(guards[guard_name], masked)
                frame[f"{guard_name} p"] = probabilities
                frame[f"{guard_name} ms"] = milliseconds
                print(f"{set_name}: {guard_name} done in {time.perf_counter() - started:.0f} s", flush=True)
            escalated = frame["classifier p"] >= threshold
            frame["cascade p"] = frame["classifier p"].where(~escalated, frame["jev p"])
            frame["cascade ms"] = frame["classifier ms"] + frame["jev ms"].where(escalated, 0)
            names.append("cascade")
            set_scores = [scores(name, labels, frame[f"{name} p"] >= threshold, frame[f"{name} ms"]) for name in names]
            set_scores[-1]["share_sent_to_jev"] = round(float(escalated.mean()), 3)
            set_thresholds = [row for name in names for row in threshold_rows(name, labels, frame[f"{name} p"])]
            set_tokens = tokens.input_tokens - tokens_before
            results[set_name] = {
                "scores": set_scores,
                "thresholds": set_thresholds,
                "jev_input_tokens": set_tokens,
                "jev_cost_usd": round(set_tokens * JEV_PRICE_PER_MILLION_INPUT_TOKENS / 1e6, 4),
            }
            per_request.append(frame)
            print(
                f"\n{set_name}: {len(requests)} requests, {int(labels.sum())} injections, blocking at p >= {threshold}"
            )
            print(pd.DataFrame(set_scores).to_string(index=False))
            print(pd.DataFrame(set_thresholds).pivot(index="guard", columns="block_from").to_string())
            print(f"Jev input tokens: {set_tokens}, about ${results[set_name]['jev_cost_usd']}\n")

    every_request = pd.concat(per_request, ignore_index=True)
    small_sets = every_request[every_request["set"] != "qualifire"]
    probability_columns = [column for column in small_sets if column.endswith(" p")]
    wrong = small_sets[(small_sets[probability_columns] >= threshold).ne(small_sets["injection"], axis=0).any(axis=1)]
    print("Requests in the small sets at least one guard got wrong:")
    print(wrong[["set", "injection", *probability_columns, "text"]].round(2).to_string(index=False))

    output = REPOSITORY / "results"
    output.mkdir(exist_ok=True)
    suffix = "" if set(sets) == set(SETS) or set(sets) == {"ours", "deepset"} else "_" + "_".join(sets)
    (output / f"eval_injection{suffix}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    every_request.round(3).to_csv(output / f"eval_injection_requests{suffix}.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or list(SETS)))
