# llml

`llml` is a small CLI for running local LLM workflows from named TOML instances.

By default, `llml` expects a config repo at `~/.config/llml/config` and clones the default source there when an action first needs instances.

## Usage

```sh
uv run llml doctor
uv run llml list [term ...]
uv run llml find [term ...]
uv run llml sync [term ...]
uv run llml remove <term ...>
uv run llml tidy
uv run llml serve <term ...>
uv run llml refresh
```

`--dry-run` (or `-n`) can appear anywhere in the args and prints the command or action without executing it.

## Matching: WIDE and PINPOINT

Every command that takes `[term ...]` matches the same way: a model's identity is the string `<instance>-<model>`, terms are matched case-insensitively by substring, and multiple terms are ANDed (each must match, narrowing the set). What differs is how many matches a command acts on:

- **WIDE** — act on **all** matches; no terms means everything. Used by the read-only and bulk-idempotent commands: `find`, `list`, `sync`.
- **PINPOINT** — narrow with the same match, then reduce to **one**: prefer an exact name match, otherwise take the first by stable sort. Used by the single-effect and destructive commands: `serve` (one instance), `remove` (one model).

Exact-match is only the tie-breaker inside PINPOINT, not a separate mode. So `sync gemma` (WIDE) downloads every `gemma*` model in every instance, while `remove gemma` (PINPOINT) removes exactly one — the model named exactly `gemma` if there is one, else the first match. `remove` and `serve` require at least one term; use `find` and `list` to see what is available.

## Actions

`find [term ...]` (WIDE) reads the config repo and prints each instance with its models indented below it, whether or not they are installed. With no terms it lists every defined model. It never touches disk beyond reading the config.

`list [term ...]` (WIDE) is `find` restricted to models whose files exist on disk. So `list gemma` shows installed `gemma*` models and bare `list` shows everything installed. Instances with nothing installed are omitted.

`sync [term ...]` (WIDE) downloads every matched model. For each one it reads the model's `sync.hf.arguments`, keeps the CLI-shaped command visible for dry runs, then uses `huggingface_hub.snapshot_download` to reconcile the exact configured files into `--local-dir`. With no terms it syncs every model in every instance. If `hf_token` is set in `llml` config, it is passed to the library. Otherwise the library uses its normal auth mechanisms, such as `HF_TOKEN` or `huggingface-cli login`.

`remove <term ...>` (PINPOINT) deletes one installed model's directory — the single model the terms pinpoint. It refuses to delete paths outside `model_dir`, and requires at least one term so it never deletes by accident.

`tidy` reconciles `model_dir` against the model definitions across every instance. It collects the `local-dir` of every model in every instance, keeps those directories (and the parent folders leading to them, such as a shared `unsloth/` org folder), and deletes everything else under `model_dir` — orphaned content from models that were dropped from a definition, as well as files and folders that never belonged to any instance. Unlike `remove`, `tidy` is not pinpointed; it cleans the whole `model_dir` based on what is still defined. Run `refresh` first so it reconciles against current definitions, and use `--dry-run` to preview the removals.

`serve <term ...>` (PINPOINT) generates `~/.cache/llml/instances/<instance>/llama-server/models.ini` for the one instance the terms pinpoint, from models that actually exist on disk, then starts `llama-server` with that instance's configured arguments.

`refresh` uses GitPython to clone the config repo if it is missing or refresh it from the remote if it already exists.

## Configuration

User configuration lives at `~/.config/llml/config.toml`.

```toml
config_uri = "https://github.com/meop/llml-config.git"
config_dir = "~/.config/llml/config"
model_dir = "~/.config/llm/model"
hf_token = "hf_..."
```

Every `llml` config value can be overridden by an environment variable:

| TOML key | Environment variable | Default |
| --- | --- | --- |
| `config_uri` | `LLML_CONFIG_URI` | Same as the example above |
| `config_dir` | `LLML_CONFIG_DIR` | `~/.config/llml/config` |
| `model_dir` | `LLML_MODEL_DIR` | `~/.config/llm/model` |
| `hf_token` | `LLML_HF_TOKEN` | Unset |

`config_uri` is the Git URL or local path used when `config_dir` does not exist. `config_dir` is the local clone. `model_dir` is where model symlinks or downloaded model directories live.

Instance files can use `${LLML_CONFIG_DIR}` and `${LLML_MODEL_DIR}`. `llml` expands those values before calling providers. `llama-server` receives a generated concrete `models.ini`; it is not expected to interpret `${...}` placeholders itself.

## Instance Shape

Instances live in `instances/*.toml` inside the config repo. Each instance is a cohesive profile: sync settings, serve settings, and model definitions all live together. Providers such as `hf` and `llama-server` are baked into the instance and are not CLI arguments.

```toml
[serve.llama-server]
arguments = [
  "--host 0.0.0.0",
  "--models-preset ${LLML_LLAMA_SERVER_MODELS_INI}",
]

[models.gemma-4-e4b]
repo = "unsloth/gemma-4-E4B-it-GGUF"
local-dir = "${LLML_MODEL_DIR}/unsloth/gemma-4-E4B-it-GGUF"
model-file = "model.gguf"
mmproj-file = "mmproj.gguf"

[models.gemma-4-e4b.sync.hf]
arguments = [
  "${repo}",
  "${model-file}",
  "${mmproj-file}",
  "--local-dir ${local-dir}",
]

[models.gemma-4-e4b.serve.llama-server]
model = "${local-dir}/${model-file}"
mmproj = "${local-dir}/${mmproj-file}"
temperature = 1.0
```

## Packaging

Runtime dependencies are `click`, `huggingface-hub`, and `GitPython`.

Hatchling is used only as the build backend so the project can produce a PyPI-ready wheel and sdist. It does not upload anything to PyPI; publishing is a separate step handled by a publisher such as `uv publish`:

```sh
uv build
```

## Development

```sh
uv sync
uv run ruff check .
uv run pytest
```
