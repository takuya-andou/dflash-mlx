# Repository Guidelines

## Project Structure & Module Organization
`dflash_mlx/` contains the library and CLI entry points: runtime logic in `runtime.py`, public API in `api.py`, model adapters in `adapters.py`, and user-facing CLIs such as `cli.py`, `chat_cli.py`, and `benchmark_cli.py`. Keep new model-family integration work aligned with `ADDING_MODELS.md`. Put tests in `tests/`, examples in `examples/`, benchmark notes in `benchmarks/`, chart assets in `assets/`, and one-off support scripts in `scripts/`.

## Build, Test, and Development Commands
Use `uv` for local work:

- `uv sync`: install the package and core dependencies.
- `uv run dflash-mlx --max-new-tokens 128`: run the default CLI against the packaged entry point.
- `uv run dflash-mlx-chat`: start the interactive chat CLI.
- `uv run pytest`: run the test suite.
- `uv run python -m py_compile dflash_mlx/*.py`: quick syntax check used in the model-adding workflow.
- `uv sync --extra charts`: install optional charting dependencies for `scripts/generate_benchmark_chart.py`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, type hints on public APIs, concise docstrings where behavior is non-obvious, and imports grouped at the top of each file. Use `snake_case` for modules, functions, variables, and CLI flags; use `PascalCase` for classes such as `DFlashGenerator`. Prefer small, focused helpers over large monolithic functions, and keep CLI defaults and help text explicit.

## Testing Guidelines
Tests currently live in `tests/test_smoke.py` and focus on importability and safe CLI defaults. Add new `test_*.py` files under `tests/` and keep tests lightweight by default: do not require downloading model weights or Apple Silicon GPU execution unless the test is explicitly integration-only. Run `uv run pytest` before opening a PR.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Fuse draft KV projections` and `Make CLI history logging opt-in`. Keep commit titles concise, capitalized, and focused on one change. PRs should explain the user-visible effect, note any model or benchmark impact, link relevant issues, and include command output or screenshots when changing CLI behavior or benchmark visuals.

## Configuration & Performance Notes
The default first run downloads large Hugging Face checkpoints. Avoid baking heavyweight downloads into tests or setup scripts. When changing benchmark code, call out warmup behavior, cache assumptions, and any effect on reported tok/s so results stay comparable.
