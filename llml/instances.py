import shlex
import tomllib
from pathlib import Path
from typing import Any

from llml.config_repo import ensure_config_repo
from llml.errors import CliError
from llml.settings import Settings, variables


def instances_dir(settings: Settings) -> Path:
  return ensure_config_repo(settings) / 'instances'


def list_instances(settings: Settings) -> list[str]:
  directory = instances_dir(settings)
  if not directory.is_dir():
    return []
  return sorted(path.stem for path in directory.glob('*.toml'))


def load_instance(settings: Settings, name: str) -> dict[str, Any]:
  path = instances_dir(settings) / f'{name}.toml'
  if not path.is_file():
    raise CliError(f'unknown instance: {name}')
  with path.open('rb') as f:
    instance = tomllib.load(f)
  if not isinstance(instance, dict):
    raise CliError(f'invalid instance: {name}')
  return instance


def expand_text(value: str, values: dict[str, str]) -> str:
  for name, replacement in values.items():
    value = value.replace('${' + name + '}', replacement)
  return value


def expand_arg_list(items: list[Any], values: dict[str, str]) -> list[str]:
  return [expand_text(token, values) for item in items for token in shlex.split(str(item), posix=True)]


def all_models(instance: dict[str, Any]) -> dict[str, dict[str, Any]]:
  models = instance.get('models')
  if not isinstance(models, dict):
    return {}
  return {name: model for name, model in models.items() if isinstance(model, dict)}


def selected_models(instance: dict[str, Any], requested: tuple[str, ...]) -> dict[str, dict[str, Any]]:
  models = all_models(instance)
  if not requested:
    return models

  missing = [name for name in requested if name not in models]
  if missing:
    raise CliError(f'unknown model(s): {", ".join(missing)}')
  return {name: models[name] for name in requested}


def nested_table(node: dict[str, Any], path: tuple[str, ...], label: str) -> dict[str, Any]:
  current: Any = node
  for key in path:
    if not isinstance(current, dict):
      raise CliError(f'{label} must be a table')
    current = current.get(key)
  if not isinstance(current, dict):
    raise CliError(f'missing {label}')
  return current


def model_pull_hf(model: dict[str, Any]) -> dict[str, Any]:
  return nested_table(model, ('pull', 'hf'), 'model pull.hf config')


def model_serve_llama_server(model: dict[str, Any]) -> dict[str, Any]:
  return nested_table(model, ('serve', 'llama-server'), 'model serve.llama-server config')


def model_values(model: dict[str, Any], settings: Settings) -> dict[str, str]:
  scoped = variables(settings)
  for key, value in model.items():
    if isinstance(value, dict | list):
      continue
    scoped[key] = expand_text(str(value), scoped)

  raw = scoped.get('local-dir') or scoped.get('local_dir')
  if raw is None:
    raise CliError('model is missing local-dir')
  local_dir = Path(raw).as_posix()
  return {**scoped, 'local-dir': local_dir, 'local_dir': local_dir}


def model_local_dir(model: dict[str, Any], settings: Settings) -> Path:
  return Path(model_values(model, settings)['local-dir'])


def model_file_path(model: dict[str, Any], settings: Settings) -> Path | None:
  llama_server = model_serve_llama_server(model)
  raw = llama_server.get('model')
  if not isinstance(raw, str):
    return None
  scoped = model_values(model, settings)
  expanded = Path(expand_text(raw, scoped))
  if expanded.is_absolute():
    return expanded
  return model_local_dir(model, settings) / expanded


def format_ini_value(value: Any, values: dict[str, str]) -> str:
  if isinstance(value, bool):
    return str(value).lower()
  if isinstance(value, str):
    return expand_text(value, values)
  return str(value)


def write_models_ini(instance_name: str, instance: dict[str, Any], settings: Settings) -> Path:
  destination = settings.cache_root / 'instances' / instance_name / 'llama-server' / 'models.ini'
  destination.parent.mkdir(parents=True, exist_ok=True)

  lines: list[str] = ['version = 1', '']
  llama_server = nested_table(instance, ('serve', 'llama-server'), 'serve.llama-server config')

  defaults = llama_server.get('defaults', {})
  if isinstance(defaults, dict):
    lines.append('[*]')
    lines.extend(f'{key} = {format_ini_value(value, variables(settings))}' for key, value in defaults.items())
    lines.append('')

  for name, model in all_models(instance).items():
    path = model_file_path(model, settings)
    if path is None or not path.exists():
      continue

    scoped_values = model_values(model, settings)
    lines.append(f'[{name}]')
    for key, value in model_serve_llama_server(model).items():
      lines.append(f'{key} = {format_ini_value(value, scoped_values)}')
    lines.append('')

  destination.write_text('\n'.join(lines), encoding='utf-8')
  return destination
