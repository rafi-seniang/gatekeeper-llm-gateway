# 0002. One layout for every Lab backend: routes, controllers, services, engines, repositories

Date: 25 Oct 2026. Status: accepted. Extends 0001.

## Context

The Lab backends grew one at a time, each with its own domain folder (`gate/`, `search/`, `counting/`, `ledger/`)
or as flat files. Finding where a piece of logic lives meant reading each service first.

## Decision

Every backend uses the same folders under `src/app`, and each layer only calls the one below it.

| Folder | Holds | Knows about HTTP |
|---|---|---|
| `api/routes/` | the HTTP contract: path, parameters, request and response models. One call to a controller per route. | yes |
| `controllers/` | one class per feature, built per request from `api/deps.py`. Reads uploads, applies limits and gates, maps domain errors to responses, builds `/meta`. | yes |
| `services/` | the use cases and the domain rules (gate, search, ledger, counting, routing). Plain Python, tested without a server. | no |
| `engines/` | models and model clients: classifiers, embedders, detectors, decision models, LLM clients. | no |
| `repositories/` | storage: Typesense, files on disk, in-memory caches and logs. | no |
| `schemas/` | Pydantic request, response and domain models, one module per feature, plus `meta.py` for the Lab contract. | no |
| `core/` | settings, errors, logging, security, and `resources.py`, which builds models and clients once in the lifespan. | no |

`VERSION` lives in `app/__init__.py`. Scripts (`scripts/`) import from these folders like the app does.

## Consequences

- The same question has the same answer in every backend: a model is in `engines/`, a rule is in `services/`.
- Routes stay a few lines each, and tests swap a service or a controller through `app.dependency_overrides`.
- Two more folders than the template's first version. A controller that only passes a call through is still kept,
  so every feature has the same shape.
