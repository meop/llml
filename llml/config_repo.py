from pathlib import Path

from git import InvalidGitRepositoryError, NoSuchPathError, Repo

from llml.errors import CliError
from llml.settings import Settings


def ensure_config_repo(settings: Settings) -> Path:
  if settings.config_dir.exists():
    if not (settings.config_dir / '.git').exists():
      raise CliError(f'config_dir exists but is not a git repo: {settings.config_dir}')
    return settings.config_dir

  settings.config_dir.parent.mkdir(parents=True, exist_ok=True)
  try:
    Repo.clone_from(settings.config_uri, settings.config_dir)
  except Exception as exc:
    raise CliError(f'failed to clone config repo from {settings.config_uri}: {exc}') from exc
  return settings.config_dir


def update_config_repo(settings: Settings, dry_run: bool = False) -> str:
  if not settings.config_dir.exists():
    if dry_run:
      return f'would clone {settings.config_uri} to {settings.config_dir}'
    ensure_config_repo(settings)
    return f'cloned {settings.config_uri} to {settings.config_dir}'

  try:
    repo = Repo(settings.config_dir)
  except (InvalidGitRepositoryError, NoSuchPathError) as exc:
    raise CliError(f'config_dir is not a git repo: {settings.config_dir}') from exc

  if dry_run:
    return f'would pull {settings.config_dir}'

  if not repo.remotes:
    raise CliError(f'config repo has no remotes: {settings.config_dir}')
  repo.remotes.origin.pull()
  return f'updated {settings.config_dir}'
