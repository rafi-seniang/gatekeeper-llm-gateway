# 0003. Feature-first layout with a shared platform package and Redis

Date: 25 Oct 2026. Status: accepted. Replaces 0002 (the same folders by kind under `src/app`).

## Context

Grouping by kind (`routes/`, `controllers/`, `services/`, `repositories/`, `schemas/`) spread one feature over five
folders, and nothing showed which files belong together. The `src/` level protects a library from importing itself
from the working directory, but the Lab backends are services run from a checkout or an image, never installed as
libraries, so it only added `--app-dir src` to every command. The cache and any rate limit lived in one process.

## Decision

```
main.py           entry point (uvicorn main:app, or python main.py)
config/           settings and logging
app/
  bootstrap.py    the app factory
  platform/       what every feature shares: resources, dependencies, errors, middleware, security, rate limit
  engines/        models and model clients shared by features
  features/<name>/
    router.py     the HTTP contract, one controller call per route
    controller.py uploads, limits, response shape
    service.py    the rules, no FastAPI imports
    schemas.py    request, response and domain models
    dependencies.py  how the feature is built per request
    (plus the feature's own stores and helpers: repository.py, cache.py, ...)
scripts/  tests/  docs/
```

- A feature imports from `platform`, `engines` and `config`, never from another feature. `platform/resources.py` is
  the one place that builds everything at startup.
- Redis is part of the platform: the `/v1` rate limit per API key (one `INCR` and `EXPIRE` per request), and the
  place for caches and locks. An empty `REDIS_URL` keeps them per process for local runs. Production refuses to
  start without Redis, and any configured Redis must answer at startup.

## Consequences

- One feature is one folder to read, review, test or delete, and could later become its own service.
- Scripts import the app from the repository root (`PYTHONPATH=.`).
- More small files than one module per layer. Worth it from the second feature on.
