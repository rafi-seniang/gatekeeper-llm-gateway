# 1. A gate in front of the model, with a classifier for injection and a decision model for routing

Date: 2026-10-25. Status: accepted.

## Context

Apps call LLMs directly. Every request costs the strong model's price and latency, prompt injections reach the
model, and personal identifiers leave the building. A gateway can decide before the model is called, if its own
decisions are fast and measured.

## Decision

- Speak the OpenAI chat API, so an app changes only its base URL.
- Guard injection with a fine-tuned classifier (`protectai/deberta-v3-base-prompt-injection-v2`), loaded at startup.
- Route with a decision model through the Jev wire format (`POST /v1/systemone`), so the decision provider is a
  setting: the Lab's `transformers serve`, `laya-serve`, Jev or OpenAI's Decisions API.
- Guard first, route second: a blocked request never waits on the decision endpoint.
- Mask identifier formats (email, phone, NIK, card) with patterns: formats, not meaning.
- Fail with a 503 that names the failing dependency; never forward unguarded or route by default.

## Evidence

On 46 labelled requests the classifier caught 15 of 16 injections with 7 false blocks in 30 benign requests, in 40 ms;
Qwen3.5-2B asked the same question as a decision caught 7 of 16 with 12 false blocks. The decision model routes well
(90%) but does not guard. See README and `scripts/eval_gate.py`.

## Consequences

The classifier wrongly blocks some Indonesian text and most support messages that carry a card or phone number, so a
real deployment needs it fine-tuned on its own benign traffic first. Two
models run per request, so the gate adds about half a second on a shared laptop GPU.
