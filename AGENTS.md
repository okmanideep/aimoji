# AGENTS.md (repo guide)

## TODO
1. Added stratified splits to preprocess to avoid the following error.
```
Traceback (most recent call last):
  File "/home/manideep/Documents/code/personal/aimoji/.venv/lib/python3.12/site-packages/sklearn/utils/_encode.py", line 235, in _encode
    return _map_to_integer(values, uniques)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/manideep/Documents/code/personal/aimoji/.venv/lib/python3.12/site-packages/sklearn/utils/_encode.py", line 174, in _map_to_integer
    return xp.asarray([table[v] for v in values], device=device(values))
                       ~~~~~^^^
  File "/home/manideep/Documents/code/personal/aimoji/.venv/lib/python3.12/site-packages/sklearn/utils/_encode.py", line 167, in __missing__
    raise KeyError(key)
KeyError: np.str_('🗲')

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/manideep/Documents/code/personal/aimoji/src/train.py", line 142, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/home/manideep/Documents/code/personal/aimoji/src/train.py", line 85, in main
    y_eval_ids = np.asarray(le.transform(y_eval))
                            ^^^^^^^^^^^^^^^^^^^^
  File "/home/manideep/Documents/code/personal/aimoji/.venv/lib/python3.12/site-packages/sklearn/preprocessing/_label.py", line 134, in transform
    return _encode(y, uniques=self.classes_)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/manideep/Documents/code/personal/aimoji/.venv/lib/python3.12/site-packages/sklearn/utils/_encode.py", line 237, in _encode
    raise ValueError(f"y contains previously unseen labels: {e}")
ValueError: y contains previously unseen labels: np.str_('🗲')
```

2. Added a sanity check at the start of train.py as well

But both are WIP. Not sure if they are running

## Build/Test
- Install: `uv sync --group dev` (or `pip install -e . && pip install pytest`).
- Run all tests: `uv run pytest -q`.
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
