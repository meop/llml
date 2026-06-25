# AGENTS

> **Design philosophy:** the CLI follows the WIDE/PINPOINT command model from the companion `wut` project. Read it before changing any command or matching behavior: [`wut/docs/COMMANDS.md`](https://github.com/meop/wut/blob/main/docs/COMMANDS.md). The [Matching: WIDE and PINPOINT](#matching-wide-and-pinpoint) section below is the llml-specific application of it.

## Project Shape

`llml` is a Python CLI package. Keep CLI parsing in `llml/cli.py`, provider work in `llml/actions.py`, instance/model parsing, matching, and rendering in `llml/instances.py`, config-repo git work in `llml/repos.py`, and user/config settings in `llml/settings.py`.

## Command Surface

The CLI is a flat set of verbs, each scoped to one target. Do not reintroduce a Docker/Ollama-style noun-subcommand hierarchy.

- Model-scoped (take `[term ...]`): `list`, `find`, `sync`, `remove`
- Instance-scoped (take `[term ...]`): `serve`
- Model-store-scoped (no args): `tidy`
- Config-repo-scoped (no args): `refresh`
- Environment (no args): `doctor`

Each CLI verb mirrors the action namespace in the instance TOML: the `sync` verb reads `[models.*.sync.hf]`, the `serve` verb reads `[serve.llama-server]` and `[models.*.serve.llama-server]`. Keep these names in sync when adding or renaming a verb. There is no backwards compatibility or migration for renamed config keys.

`tidy` is store-wide: it keeps the `local-dir` of every model defined in every instance (plus the parent folders leading to them) and deletes everything else under `model_dir`. It reconciles against current definitions, so it depends on `refresh` having been run.

`--dry-run` / `-n` is parsed in `main()` so it can appear anywhere in the args, and is also a root option. Destructive verbs (`remove`, `tidy`) must honor it and must refuse to touch paths outside `model_dir`.

## Matching: WIDE and PINPOINT

This is the core CLI philosophy, mirrored from the `wut` project ([`wut/docs/COMMANDS.md`](https://github.com/meop/wut/blob/main/docs/COMMANDS.md)). All term-taking commands share one matcher in `instances.py`: `_filter_instances` forms `<instance>-<model>` per model and keeps it when every term is a case-insensitive substring (ANDed, order-independent; no terms passes everything). What differs is cardinality:

- **WIDE** — act on every match. Read-only and bulk-idempotent verbs: `find`, `list`, `sync`. `find`/`installed`/`wide_models` flatten the filter to the full set; `find`/`list` differ only in candidate set (defined vs on-disk) and both render via `echo_matches`.
- **PINPOINT** — narrow with the same filter, then reduce to one via `_pinpoint`: prefer an exact name match, else take the first by stable sort. Destructive/single-effect verbs: `remove` (one installed model, via `pinpoint_installed_model`) and `serve` (one instance, via `pinpoint_instance`).

Invariants to preserve: exact-match is only the PINPOINT tie-breaker, never a third mode (do not add exact-short-circuit to WIDE). `remove` and `serve` require at least one term. Route any new term argument through these helpers rather than matching names directly, and classify new verbs as WIDE (read-only/bulk/idempotent) or PINPOINT (destructive/single-effect).

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
