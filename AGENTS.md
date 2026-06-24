# AGENTS

## Project Shape

`llml` is a Python CLI package. Keep CLI parsing in `llml/cli.py`, provider work in `llml/actions.py`, instance/model parsing, matching, and rendering in `llml/instances.py`, config-repo git work in `llml/repos.py`, and user/config settings in `llml/settings.py`.

## Command Surface

The CLI is a flat set of verbs, each scoped to one target. Do not reintroduce a Docker/Ollama-style noun-subcommand hierarchy.

- Instance-scoped: `list`, `find`, `sync`, `remove`, `serve`
- Model-store-scoped: `tidy`
- Config-repo-scoped: `refresh`
- Environment: `doctor`

Each CLI verb mirrors the action namespace in the instance TOML: the `sync` verb reads `[models.*.sync.hf]`, the `serve` verb reads `[serve.llama-server]` and `[models.*.serve.llama-server]`. Keep these names in sync when adding or renaming a verb. There is no backwards compatibility or migration for renamed config keys.

`find` lists all defined models from the config; `list` lists only models whose files exist on disk. They share `_filter_instances` (the term filter) and `echo_matches` (the output) in `cli.py`/`instances.py` — the only difference is the candidate set. The filter forms `<instance>-<model>` per model and keeps it when every term is a case-insensitive substring (ANDed, order-independent). With no terms the filter passes everything.

`tidy` is store-wide: it keeps the `local-dir` of every model defined in every instance (plus the parent folders leading to them) and deletes everything else under `model_dir`. It reconciles against current definitions, so it depends on `refresh` having been run.

`--dry-run` / `-n` is parsed in `main()` so it can appear anywhere in the args, and is also a root option. Destructive verbs (`remove`, `tidy`) must honor it and must refuse to touch paths outside `model_dir`.

## Argument Resolution

`sync`, `remove`, and `serve` resolve their `<instance>` and `[model ...]` arguments through `resolve_instance` / `resolve_models`, built on `_loose_match` in `instances.py`. The invariant: an exact (case-insensitive) match short-circuits substring matching; otherwise the term matches every name it is a substring of. An instance must resolve to exactly one (ambiguous or unknown is an error). A model term may glob to several, unless it matches one exactly. Preserve this short-circuit rule and route any new instance/model argument through these resolvers rather than matching names directly.

## Config Contract

Do not add CLI provider arguments for concrete providers such as `hf` or `llama-server`. Providers are selected by the instance TOML. Keep dry-run output explicit enough that a user can see the downstream provider command shape.

## Development

Use `uv` for local commands:

```sh
uv sync
uv run ruff check .
uv run pytest
```

Prefer focused tests under `tests/` for config parsing, argument resolution, filter output, dry-run output, and generated provider artifacts.
