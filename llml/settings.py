import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llml.errors import CliError

APP_NAME = 'llml'
DEFAULT_CONFIG_SOURCE = 'https://github.com/meop/llml-config.git'
DEFAULT_CONFIG_DIR = Path.home() / '.config' / APP_NAME / 'config'
DEFAULT_MODEL_BASE_DIR = Path.home() / '.config' / 'llm' / 'model'


@dataclass(frozen=True)
class Settings:
  user_config_path: Path
  config_uri: str
  config_dir: Path
  model_dir: Path
  hf_token: str | None

  @property
  def cache_root(self) -> Path:
    return Path.home() / '.cache' / APP_NAME


def read_user_config(path: Path | None = None) -> dict[str, Any]:
  path = path or user_config_path()
  if not path.is_file():
    return {}

  with path.open('rb') as f:
    config = tomllib.load(f)
  if not isinstance(config, dict):
    raise CliError(f'user config must be a TOML table: {path}')
  return config


def user_config_path() -> Path:
  return Path.home() / '.config' / APP_NAME / 'config.toml'


def _string_value(config: dict[str, Any], key: str, env_name: str, default: str) -> str:
  value = os.environ.get(env_name) or config.get(key)
  if value is None:
    value = default
  if not isinstance(value, str):
    raise CliError(f'{key} must be a string')
  return value


def _optional_string_value(config: dict[str, Any], key: str, env_name: str) -> str | None:
  value = os.environ.get(env_name) or config.get(key)
  if value is None:
    return None
  if not isinstance(value, str):
    raise CliError(f'{key} must be a string')
  return value


def load_settings(path: Path | None = None) -> Settings:
  user_path = path or user_config_path()
  config = read_user_config(user_path)

  config_dir = os.environ.get('LLML_CONFIG_DIR') or config.get('config_dir')
  config_dir = config_dir or str(DEFAULT_CONFIG_DIR)
  if not isinstance(config_dir, str):
    raise CliError('config_dir must be a string')

  config_uri = _string_value(config, 'config_uri', 'LLML_CONFIG_URI', DEFAULT_CONFIG_SOURCE)
  model_dir = _string_value(config, 'model_dir', 'LLML_MODEL_DIR', str(DEFAULT_MODEL_BASE_DIR))
  hf_token = _optional_string_value(config, 'hf_token', 'LLML_HF_TOKEN')

  return Settings(
    user_config_path=user_path,
    config_uri=config_uri,
    config_dir=Path(config_dir).expanduser(),
    model_dir=Path(model_dir).expanduser(),
    hf_token=hf_token,
  )


def variables(settings: Settings) -> dict[str, str]:
  return {
    'LLML_CONFIG_DIR': settings.config_dir.as_posix(),
    'LLML_MODEL_DIR': settings.model_dir.as_posix(),
  }
