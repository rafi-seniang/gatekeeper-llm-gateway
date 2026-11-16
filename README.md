# Gatekeeper

An OpenAI-compatible LLM gateway. An app points its OpenAI SDK at Gatekeeper instead of at a model, and every request
is checked before any language model sees it. Part of the AI Lab (rafilab.com): it keeps the Lab contract (`/health`,
`/meta`, OpenAPI) and has screens in the lab frontend, including one that shows the scopes other apps use.

## What it does

1. **Guard.** One call to Jev (TypeSafe's decision model) answers two questions about the latest user message:
   is it a prompt injection, and is it outside the allowed scope? At `BLOCK_FROM_INJECTION_PROBABILITY` or
   `BLOCK_FROM_OFF_TOPIC_PROBABILITY` (0.90 each on rafilab.com) the request is refused with a 403 and never sent. A
   call takes about 0.2 s, blocked or allowed. If Jev fails, Gemini Decision 1 answers the same questions, and the
   fallback is logged as a warning.
2. **Scopes in plain language.** The allowed scope is written as a sentence: sent with the request (`scope`), kept as
   the default (`DEFAULT_ALLOWED_SCOPE`), or kept here under a name (`NAMED_SCOPES`) that an app sends as
   `scope_name`. Tiketa's chat uses `tiketa`. `GET /v1/gate/scopes` shows each named scope with the exact questions
   Jev is asked.
3. **One model.** Every allowed request goes to `UPSTREAM_MODEL` (`deepseek-flash` on rafilab.com). There is no
   small or strong router (`docs/adr/0005-one-upstream-model.md`).
4. **For agents** (`AGENT_GUARD_ENABLED`): new tool results are read against the user's task for hidden instructions,
   and a tool call the user did not ask for is stopped before it reaches the app.
5. **Masking of personal data** (`MASK_IDENTIFIERS_ENABLED`, off on rafilab.com): a local tagger replaces names, NIKs,
   phones and other identifiers with tags before any model sees them, and the answer gets the real values back. It is
   off because the tagger knows Indonesian only.
6. **Answer cache and rate limit**, both in Redis. A caller gets `RATE_LIMIT_PER_MINUTE` requests a minute and a 429
   with `Retry-After` after that. Keys in `TRUSTED_API_KEYS` (the lab's own apps) skip the daily visitor caps.

When Jev, the upstream or Redis fails, the answer is a 503 that names it, never a quiet fallback.

## Run

```powershell
# from Lab/backends/gatekeeper, with DEEPSEEK_API_KEY and JEV_API_KEY in Lab/secrets/secrets.env
copy .env.example .env
docker compose up -d redis
..\..\.venv\Scripts\python.exe main.py           # or: uvicorn main:app --port 8020
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8020/v1", api_key="local")
answer = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "Hello"}])
```

Every setting is in `config/settings.py` with a comment, and `.env.example` lists the ones to change. Without
`REDIS_URL` the rate limit and cache are per process (local only). With `ENVIRONMENT=production` the gate refuses to
start without Redis.

## API

| Endpoint | What |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat format in, the upstream's completion plus a `gate` field out. Not streamed. |
| `POST /v1/gate/check` | the decision for a message (with `scope` or `scope_name`), without sending it. Logged. |
| `GET /v1/gate/scopes` | the named scopes and the questions Jev is asked with each |
| `GET /v1/gate/stats` | totals over the last `EVENT_LOG_SIZE` requests |
| `GET /v1/gate/events?limit=50&scope_name=` | the latest requests, newest first |
| `GET /health`, `GET /meta`, `GET /docs` | the Lab contract (`/meta` includes the block cutoffs) |

## Layout

Feature-first (`docs/adr/0003-feature-first-layout.md`): one folder per feature, shared pieces in `app/platform`.

```
main.py          entry point
config/          settings and JSON logging
app/
  bootstrap.py   the app factory
  platform/      resources, dependencies, errors, middleware, rate limit, Redis
  engines/       guards (Jev, classifier, cascade), the agent guard, the decision client, the upstream client
  features/
    gate/        router, controller, service (the checks), masking, cache, events, schemas
    health/      /health and /meta
scripts/         evaluations (from the repository root with PYTHONPATH=.)
tests/           fakes for every model and client, no network
docs/            adr/ (decisions), experiments/ (measurements)
```

## Checks

`ruff check . && ruff format --check . && mypy && pytest -q` (54 tests, no network).

## Measured

- **Production path** (26 EN/ID messages, Jev, 0.90 cutoffs): 0 of 14 ordinary messages blocked, 5 of 6 injections
  and 6 of 6 off-topic requests caught. The one miss is an instruction pasted inside data. A set this small shows the
  cutoff is sane, not a rate.
- **Guards compared** on two labelled sets and a 5,000-prompt public benchmark (Jev, a DeBERTa classifier, Laya, a
  cascade): `docs/experiments/injection-guards-compared.md` and `injection-guard-thresholds.md`.
- **Agent checks** (tool results and tool calls) on public benchmarks: `docs/experiments/agent-guards.md`.
- **Masking:** 4.1% of values leaked on its benchmark, against 79.8% for regex patterns
  (`docs/adr/0004-model-masking-and-agent-guards.md`).

## Known limits

- `stream: true` is rejected with 422.
- The event log is in memory: stats and events describe one process and restart with it.
- No guard catches every attack, and the small hand-written sets show a threshold is sane, not a rate. The tool
  checks rest on public benchmarks, not real traffic.
- Masking misses some values (more on English IDs and spelled-out numbers), and a wrongly masked word reaches the
  model as a tag.
