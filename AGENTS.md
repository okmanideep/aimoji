# AGENTS.md (repo guide)

## Build/Test
- Install: `uv sync --group dev` (or `pip install -e . && pip install pytest`).
- Run all tests: `uv run pytest -q`.
- Run one test: `uv run pytest -q tests/test_preprocess.py::test_repeated_emoji_collapsed` (or `-k <expr>`).
- No build step; Python 3.12 runtime.

## Code Style
- Formatting: Black (88 cols), newline at EOF; keep imports one per line.
- Linting: Ruff defaults; fix warnings; keep imports sorted (stdlib, third‑party, local).
- Imports: prefer absolute imports; avoid relative; module paths assume `src/` layout.
- Types: add type hints; prefer builtin generics (`list[str]`, `dict[str, Any]`).
- Naming: snake_case for functions/vars; PascalCase for classes; UPPER_SNAKE_CASE constants.
- Errors: raise precise exceptions; avoid bare `except`; return int exit codes; write user‑visible errors to stderr.
- Functions: keep small, pure; avoid side effects; docstrings for public functions.

## Testing
- Tests live in `tests/`; name files `test_*.py`; minimize fixtures; assert behavior not internals.

## Repo Notes
- Source currently loaded via `importlib` from `src/`; do not depend on installed package yet.
- No Cursor or Copilot rules found in `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md`.
