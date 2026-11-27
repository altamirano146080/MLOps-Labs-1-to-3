[![CI](https://github.com/altamirano146080/MLOps-Labs-1-to-3/actions/workflows/CI.yml/badge.svg)](https://github.com/altamirano146080/MLOps-Labs-1-to-3/actions/workflows/CI.yml)

# MLOps-Labs-1-to-3 — Image Classifier

Image classification, resizing, and preprocessing application built incrementally for the MLOps labs. It includes a reusable core library, a Click CLI, a FastAPI REST API, automated tests with `pytest`, and a GitHub Actions CI pipeline.

## Project Structure

- `mylib/` — Core image processing and prediction logic (`mylib/image_classifier.py`).
- `cli/` — Command-line interface built with `click` (`cli/cli.py`).
- `api/` — FastAPI REST API (`api/api.py`).
- `templates/` — HTML home page served at `/` (`templates/home.html`).
- `tests/` — Unit and integration tests (`test_logic.py`, `test_cli.py`, `test_api.py`).
- `.github/workflows/CI.yml` — GitHub Actions CI workflow (install, format check, lint, and tests).
- `Makefile` — Common development and CI tasks.

## Setup

Requires Python 3.13 (pinned in `.python-version`) and [`uv`](https://docs.astral.sh/uv/).

```bash
make install
```

## Usage

### CLI

Run `uv run python -m cli.cli --help` to see all available commands:

- `predict`: Predict a class label for an image (`--classes`, `--seed`).
- `resize`: Resize an image to `WIDTH` and `HEIGHT` (`--output` / `-o`).
- `normalize-orientation`: Normalize image orientation using EXIF tags (`--output` / `-o`).
- `adjust-contrast`: Adjust image contrast (`--factor`, `--output` / `-o`).
- `adjust-brightness`: Adjust image brightness (`--factor`, `--output` / `-o`).
- `equalize-histogram`: Equalize image histogram across RGB channels (`--output` / `-o`).
- `apply-gaussian-blur`: Apply Gaussian blur (`--radius`, `--output` / `-o`).
- `to-grayscale-rgb`: Convert an image to grayscale and back to 3-channel RGB (`--output` / `-o`).

Example:

```bash
uv run python -m cli.cli predict sample.jpg --seed 7
uv run python -m cli.cli resize sample.jpg 224 224 -o resized.jpg
```

### REST API

Start the FastAPI server locally (defaults to port `8000`):

```bash
make run-api
```

Then open:

- Home page: `http://localhost:8000/`
- Interactive Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

Available endpoints:

- `GET /` — HTML home page.
- `GET /health` — Health check returning `{"status": "ok"}`.
- `POST /predict` — Predict a class for an uploaded image (`file`, optional `classes`).
- `POST /resize` — Resize an uploaded image (`file`, `width`, `height`).
- `POST /preprocess` — Preprocess an uploaded image (`file`, `method`, `factor`, `radius`).

## Development & CI Checks

- `make format` — Format code with `black`.
- `make format-check` — Verify formatting without modifying files.
- `make lint` — Run `pylint` on `mylib`, `api`, and `cli`.
- `make test` — Run `pytest` with coverage reporting (`--cov-fail-under=90`).
- `make refactor` — Run `format` and `lint`.
- `make all` — Run `install`, `format-check`, `lint`, and `test`.

# MLOps-Labs-1-to-3 — Image Classifier

Image classification, resizing, and preprocessing application built incrementally for the MLOps labs. It includes a reusable core library, a Click CLI, a FastAPI REST API, automated tests with `pytest`, and a GitHub Actions CI pipeline.

## Project Structure

- `mylib/` — Core image processing and prediction logic (`mylib/image_classifier.py`).
- `cli/` — Command-line interface built with `click` (`cli/cli.py`).
- `api/` — FastAPI REST API (`api/api.py`).
- `templates/` — HTML home page served at `/` (`templates/home.html`).
- `tests/` — Unit and integration tests (`test_logic.py`, `test_cli.py`, `test_api.py`).
- `.github/workflows/CI.yml` — GitHub Actions CI workflow (install, format check, lint, and tests).
- `Makefile` — Common development and CI tasks.

## Setup

Requires Python 3.13 (pinned in `.python-version`) and [`uv`](https://docs.astral.sh/uv/).

```bash
make install
```

## Usage

### CLI

Run `uv run python -m cli.cli --help` to see all available commands:

- `predict`: Predict a class label for an image (`--classes`, `--seed`).
- `resize`: Resize an image to `WIDTH` and `HEIGHT` (`--output` / `-o`).
- `normalize-orientation`: Normalize image orientation using EXIF tags (`--output` / `-o`).
- `adjust-contrast`: Adjust image contrast (`--factor`, `--output` / `-o`).
- `adjust-brightness`: Adjust image brightness (`--factor`, `--output` / `-o`).
- `equalize-histogram`: Equalize image histogram across RGB channels (`--output` / `-o`).
- `apply-gaussian-blur`: Apply Gaussian blur (`--radius`, `--output` / `-o`).
- `to-grayscale-rgb`: Convert an image to grayscale and back to 3-channel RGB (`--output` / `-o`).

Example:

```bash
uv run python -m cli.cli predict sample.jpg --seed 7
uv run python -m cli.cli resize sample.jpg 224 224 -o resized.jpg
```

### REST API

Start the FastAPI server locally (defaults to port `8000`):

```bash
make run-api
```

Then open:

- Home page: `http://localhost:8000/`
- Interactive Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

Available endpoints:

- `GET /` — HTML home page.
- `GET /health` — Health check returning `{"status": "ok"}`.
- `POST /predict` — Predict a class for an uploaded image (`file`, optional `classes`).
- `POST /resize` — Resize an uploaded image (`file`, `width`, `height`).
- `POST /preprocess` — Preprocess an uploaded image (`file`, `method`, `factor`, `radius`).

## Development & CI Checks

- `make format` — Format code with `black`.
- `make format-check` — Verify formatting without modifying files.
- `make lint` — Run `pylint` on `mylib`, `api`, and `cli`.
- `make test` — Run `pytest` with coverage reporting (`--cov-fail-under=90`).
- `make refactor` — Run `format` and `lint`.
- `make all` — Run `install`, `format-check`, `lint`, and `test`.
