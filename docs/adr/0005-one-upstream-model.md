# 0005. One upstream model, no small or strong router

Date: 16 Nov 2026. Status: accepted. Replaces the routing step of 0001.

## Context

An allowed request went to a decision model (DeepSeek, read through letter log-probabilities) that rated its
difficulty from 0 to 2, then to `deepseek-flash` or `deepseek-v4-pro`. On the live demo this cost 300 to 600 ms on every
allowed request. The router was also a language model call: for a moment on 16 Nov the router ran beside the guards,
so a blocked request reached it too, which defeats the gate. And the demo's traffic (a cinema assistant, short
questions) never needs the strong model, so there was nothing to choose between.

## Decision

- Every allowed request goes to one model, `UPSTREAM_MODEL` (`deepseek-flash` in production).
- `route`, `difficulty`, `STRONG_MODEL`, `STRONG_FROM_DIFFICULTY` and the routing evaluation are removed.
- A client's `model` field is accepted, so any OpenAI client works unchanged, and is not used.
- The decision endpoint settings (`DECISION_*`) remain only for `INJECTION_GUARD=decision`.

## Consequences

- An allowed request costs one Jev call (about 0.2 s) before the model, and a blocked request reaches no language model.
- If a deployment ever needs two models, a Jev Score question can rate the difficulty in the guard's own request, at no
  extra round trip.
