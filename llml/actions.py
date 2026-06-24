import shlex
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from huggingface_hub import snapshot_download

from llml.errors import CliError
from llml.instances import (
  all_models,
  expand_arg_list,
  model_fetch_hf,
  model_local_dir,
  model_values,
  nested_table,
  selected_models,
  write_models_ini,
)
from llml.settings import APP_NAME, Settings, variables


def app_version() -> str:
  try:
    return version(APP_NAME)
  except PackageNotFoundError:
    return '0.0.0+editable'


def hf_command_preview(arguments: list[str]) -> str:
  return shlex.join(['hf', 'download', *arguments])


def parse_hf_download_args(arguments: list[str]) -> tuple[str, list[str], Path]:
  if not arguments:
    raise CliError('model fetch.hf arguments must start with a repo id')

  repo_id = arguments[0]
  files: list[str] = []
  local_dir: Path | None = None
  index = 1

  while index < len(arguments):
    arg = arguments[index]
    if arg == '--local-dir':
      index += 1
      if index >= len(arguments):
        raise CliError('--local-dir needs a value')
      local_dir = Path(arguments[index])
    elif arg.startswith('--'):
      raise CliError(f'unsupported hf argument for library fetch: {arg}')
    else:
      files.append(arg)
    index += 1

  if local_dir is None:
    raise CliError('model fetch.hf arguments need --local-dir')
  return repo_id, files, local_dir


def fetch_models(instance: dict, model_names: tuple[str, ...], settings: Settings, dry_run: bool) -> list[str]:
  output: list[str] = []
  for _, model in selected_models(instance, model_names).items():
    hf = model_fetch_hf(model)
    arguments = hf.get('arguments')
    if not isinstance(arguments, list):
      raise CliError('model fetch.hf needs an arguments list')

    expanded = expand_arg_list(arguments, model_values(model, settings))
    if dry_run:
      output.append(hf_command_preview(expanded))
      continue

    repo_id, files, local_dir = parse_hf_download_args(expanded)
    snapshot_download(repo_id=repo_id, allow_patterns=files or None, local_dir=str(local_dir), token=settings.hf_token)
    output.append(f'fetched {repo_id} to {local_dir}')
  return output


def serve_instance(instance_name: str, instance: dict, settings: Settings, dry_run: bool) -> tuple[int, str]:
  serve = nested_table(instance, ('serve', 'llama-server'), 'serve.llama-server config')
  provider = 'llama-server'

  values = variables(settings)
  values['LLML_LLAMA_SERVER_MODELS_INI'] = write_models_ini(instance_name, instance, settings).as_posix()
  cmd = [provider, *expand_arg_list(serve.get('arguments', []), values)]

  if dry_run:
    return 0, shlex.join(cmd)
  return subprocess.run(cmd).returncode, ''


def is_under(path: Path, parent: Path) -> bool:
  try:
    path.resolve(strict=False).relative_to(parent.resolve(strict=False))
  except ValueError:
    return False
  return True


def purge_models(instance: dict, keep_model_names: tuple[str, ...], settings: Settings, dry_run: bool) -> list[str]:
  keep = set(keep_model_names)
  models = all_models(instance)
  missing = sorted(keep - set(models))
  if missing:
    raise CliError(f'unknown model(s): {", ".join(missing)}')

  output: list[str] = []
  targets = [model_local_dir(model, settings) for name, model in models.items() if name not in keep]
  for target in targets:
    if not is_under(target, settings.model_dir):
      raise CliError(f'refusing to purge path outside model_dir: {target}')
    if dry_run:
      output.append(f'would remove {target}')
    elif target.exists():
      shutil.rmtree(target)
      output.append(f'removed {target}')
  return output


def executable_version(name: str) -> tuple[str | None, str | None]:
  path = shutil.which(name)
  if path is None:
    return None, None
  for version_args in (['--version'], ['version']):
    try:
      result = subprocess.run([name, *version_args], text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
      continue
    output = (result.stdout or result.stderr).strip().splitlines()
    if output:
      return path, output[0]
  return path, 'version unavailable'
