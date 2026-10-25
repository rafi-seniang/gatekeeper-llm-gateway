# On Windows use Git Bash, or run the commands below by hand.
PY ?= python

.PHONY: install dev test lint typecheck check eval

install:            ## create the venv and install with dev tools
	$(PY) -m venv .venv && .venv/bin/pip install -e ".[dev]" || .venv/Scripts/pip install -e ".[dev]"

dev:                ## run with reload
	uvicorn main:app --reload --port 8020

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

typecheck:
	mypy

check: lint typecheck test   ## what CI runs

eval:              ## measure the guard and the router (needs the decision endpoint)
	PYTHONPATH=. python scripts/eval_gate.py
