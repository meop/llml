from pathlib import Path

import pytest

from llml.config_repo import ensure_config_repo
from llml.errors import CliError
from llml.settings import Settings


def make_settings(tmp_path: Path, config_dir: Path) -> Settings:
  return Settings(
    user_config_path=tmp_path / 'config.toml',
    config_uri='git@example.test:meop/config.git',
    config_dir=config_dir,
    model_dir=tmp_path / 'models',
    hf_token=None,
  )


def test_ensure_config_repo_clones_into_empty_existing_dir(tmp_path: Path, monkeypatch) -> None:
  config_dir = tmp_path / 'config'
  config_dir.mkdir()
  calls = []

  def fake_clone_from(uri: str, destination: Path) -> None:
    calls.append((uri, destination))
    (destination / '.git').mkdir()

  monkeypatch.setattr('llml.config_repo.Repo.clone_from', fake_clone_from)

  assert ensure_config_repo(make_settings(tmp_path, config_dir)) == config_dir
  assert calls == [('git@example.test:meop/config.git', config_dir)]


def test_ensure_config_repo_rejects_non_empty_non_git_dir(tmp_path: Path, monkeypatch) -> None:
  config_dir = tmp_path / 'config'
  config_dir.mkdir()
  (config_dir / 'README.md').write_text('not a repo', encoding='utf-8')
  calls = []
  monkeypatch.setattr('llml.config_repo.Repo.clone_from', lambda *args: calls.append(args))

  with pytest.raises(CliError, match='exists but is not a git repo'):
    ensure_config_repo(make_settings(tmp_path, config_dir))

  assert calls == []
