# Repository Guidelines

## Project Structure & Module Organization
The backend FastAPI application lives in `src/backend`; core Python packages sit under `src/backend/base/langflow/`, while service wrappers and runtime utilities are under `src/backend/langflow/`. Frontend assets built with Node live in `src/frontend`, and reusable LFX tooling resides in `src/lfx/src/lfx/`. Automated tests are grouped in `src/backend/tests` (unit, integration, locust) and `src/lfx/tests`. Shared operational scripts and deployment assets are in `scripts/`, `docker/`, and `deploy/`.

## Build, Test, and Development Commands
- `make init` — installs backend and frontend dependencies and configures pre-commit hooks.
- `make run_cli` — builds the frontend (reusing caches) and launches Langflow locally at `http://localhost:7860`.
- `make backend` — starts the FastAPI dev server with hot reload; set `LFX_DEV=1 make backend` to regenerate component indexes on change.
- `make tests` — runs unit, integration, and coverage suites; use `make unit_tests` or `uv run pytest src/backend/tests/unit -k name` for targeted runs.

## Coding Style & Naming Conventions
Python code uses Ruff with 120-character lines. Prefer four-space indentation, type hints for new modules, and snake_case module names (e.g., `langflow/components/agent_router.py`). Run `make format_backend` (Ruff format and fix) before committing. Frontend TypeScript/JS follows Biome defaults; run `cd src/frontend && npx @biomejs/biome check --write`. Keep new React components in PascalCase files inside feature folders, and colocate CSS modules alongside components.

## Testing Guidelines
Primary tests run with Pytest via `uv run pytest`. Name Python tests `test_<feature>.py` and use `Test*` classes or `test_*` functions. Apply markers defined in `pyproject.toml` (`@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.api_key_required`) to aid selection. Store fixture data under `src/backend/tests/fixtures/`. For coverage parity, ensure backend changes keep `make tests` green and inspect `coverage/` HTML reports when touching critical flows.

## Commit & Pull Request Guidelines
Follow Conventional Commits for titles (e.g., `feat: add redis message cache agent`). Reference issues with `Fixes #123` and describe user-visible changes plus rollout considerations. Before opening a PR, run `make format` and the relevant tests, attach screenshots or terminal output for UI or CLI-facing updates, and note any required config changes. PR descriptions should call out required environment variables, migrations, or external service keys.

## Security & Configuration Tips
Never commit secrets; load them from a local `.env` referenced by `make backend`. Use `LFX_DEV` to limit component loading while iterating (for example, `LFX_DEV=openai,anthropic` keeps startup fast). When adding new integrations, document required credentials in `docs/docs/Contributing/` and gate tests with the `api_key_required` marker.
