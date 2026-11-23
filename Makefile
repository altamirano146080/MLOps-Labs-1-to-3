# We use Bash for complex commands and consistency
SHELL := /bin/bash
# We ensure that ~/.local/bin (where uv can be installed) is in the PATH
PATH := $(HOME)/.local/bin:$(PATH)

# Local port of the API (can be overridden: make run-api PORT=9000)
PORT ?= 8000
# Source folders checked by black and pylint
SRC := mylib api cli

# Specify targets that are orders, not files, so that it executes them if someone creates a file/folder with the same name
.PHONY: install format format-check lint test run-api refactor all

# Installs uv (if needed) and the dependencies, failing if uv.lock is out of date
install:
	@if ! command -v uv &> /dev/null; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	uv sync --locked

# Formats the code (locally)
format:
	uv run black $(SRC) tests

# Checks the format without modifying files (CI): fails if black would change something
format-check:
	uv run black --check --diff $(SRC) tests

lint:
	uv run pylint --recursive=y --disable=R,C $(SRC)

test:
	uv run python -m pytest tests -vv --cov=mylib --cov=api --cov=cli \
		--cov-report=term-missing --cov-fail-under=90

# Runs the API locally with auto-reload: http://localhost:8000/docs
run-api:
	uv run uvicorn api.api:app --reload --port $(PORT)

refactor: format lint

all: install format-check lint test
