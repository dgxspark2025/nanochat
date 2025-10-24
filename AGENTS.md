# Repository Guidelines

## Project Structure & Module Organization
- `nanochat/` hosts the core runtime: models in `gpt.py`, training loops in `engine.py`/`execution.py`, and the UI shell in `ui.html`.
- `scripts/` collects runnable entrypoints (`python -m scripts.base_train`, `python -m scripts.chat_web`, etc.); docstrings flag required arguments.
- `rustbpe/` contains the tokenizer crate; rebuild with `maturin` after Rust edits before invoking Python workflows.
- `tasks/` bundles evaluation adapters (ARC, GSM8K, MMLU, Humaneval) consumed by the eval scripts.
- `tests/` stores the pytest suite—mirror `tests/test_rustbpe.py` when expanding coverage.
- `speedrun.sh` and `run1000.sh` coordinate multi-stage pipelines, while `dev/` holds shared assets.

## Build, Test, and Development Commands
- `uv sync --all-extras` installs the environment declared in `pyproject.toml`.
- `uv run maturin develop --manifest-path rustbpe/Cargo.toml` builds the editable `rustbpe` wheel required by tokenizer flows.
- `uv run python -m scripts.base_train` (or `chat_sft`, `chat_rl`) launches pipeline stages; override via `--flag=value` or config files read by `nanochat/configurator.py`.
- `uv run python -m scripts.chat_web --host 0.0.0.0 --port 8000` serves the chat UI for smoke tests.
- `uv run python -m pytest -m "not slow"` covers the fast subset locally; remove the marker before merging to exercise slow suites.

## Coding Style & Naming Conventions
- Keep to Black-style four-space indentation, snake_case identifiers, and concise module docstrings, matching existing `nanochat/*.py`.
- Prefer typed signatures and lean config objects so overrides stay compatible with `configurator.py`.
- Centralize shared logic in `nanochat/common.py`; define constants in UPPER_CASE at module scope.

## Testing Guidelines
- Use pytest markers from `pyproject.toml`; tag heavyweight runs with `@pytest.mark.slow`.
- Follow discovery rules (`tests/test_*.py`, `Test*` classes, `test_*` functions) to keep CI predictable.
- Keep fixtures lightweight and reference cached assets under `~/.cache/nanochat` rather than committing bulk data.

## Commit & Pull Request Guidelines
- Write imperative, scoped commit titles (`fix(ui): prevent toolbar overlap`, `Docs: refresh speedrun instructions`) consistent with recent history.
- Squash experimental checkpoints so each commit is reviewable and intentional.
- PR descriptions should cite the issue, summarize the solution, and list validation (training/eval commands, key metrics, screenshots when behavior shifts); flag dependency updates and rollout risks.

## Security & Configuration Tips
- Store secrets outside the repo; rely on environment variables such as `NANOCHAT_BASE_DIR`, `WANDB_API_KEY`, and provider credentials.
- Confirm long-running scripts write into `~/.cache/nanochat` unless a custom base directory is intended.
- Scrub IPs, cluster names, and private dataset paths before sharing logs from `speedrun.sh` or `run1000.sh`.
