# AGENTS

## Project Shape

`llml` is a Python CLI package. Keep CLI parsing in `llml/cli.py`, provider work in `llml/actions.py`, instance parsing/rendering in `llml/instances.py`, and user/config-repo settings in `llml/settings.py` and `llml/config_repo.py`.

## Development

Use `uv` for local commands:

```sh
uv sync
uv run ruff check .
uv run pytest
```

Prefer focused tests under `tests/` for config parsing, dry-run output, and generated provider artifacts.

## Config Contract

Do not add CLI provider arguments for concrete providers such as `hf` or `llama-server`. Providers are selected by the instance TOML. Keep dry-run output explicit enough that a user can see the downstream provider command shape.
