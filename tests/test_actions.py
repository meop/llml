from pathlib import Path

import pytest

from llml.actions import fetch_models, parse_hf_download_args
from llml.errors import CliError
from llml.instances import load_instance
from llml.settings import Settings


def make_config_repo(tmp_path: Path, body: str) -> tuple[Settings, Path]:
  config_dir = tmp_path / 'config'
  instances_dir = config_dir / 'instances'
  instances_dir.mkdir(parents=True)
  (config_dir / '.git').mkdir()
  (instances_dir / 'desktop.toml').write_text(body, encoding='utf-8')
  model_dir = tmp_path / 'models'
  return (
    Settings(
      user_config_path=tmp_path / 'config.toml',
      config_uri='unused',
      config_dir=config_dir,
      model_dir=model_dir,
      hf_token=None,
    ),
    model_dir,
  )


def test_parse_hf_download_args() -> None:
  repo_id, files, local_dir = parse_hf_download_args(
    [
      'unsloth/example-GGUF',
      'model.gguf',
      'mmproj.gguf',
      '--local-dir',
      'C:/models/example',
    ]
  )

  assert repo_id == 'unsloth/example-GGUF'
  assert files == ['model.gguf', 'mmproj.gguf']
  assert local_dir == Path('C:/models/example')


def test_parse_hf_download_args_rejects_unmapped_options() -> None:
  with pytest.raises(CliError, match='unsupported hf argument'):
    parse_hf_download_args(['repo/name', '--revision', 'main', '--local-dir', 'models'])


def test_fetch_dry_run_preserves_hf_cli_shape(tmp_path: Path) -> None:
  settings, model_dir = make_config_repo(
    tmp_path,
    """
[models.gemma]
repo = "unsloth/gemma-GGUF"
local-dir = "${LLML_MODEL_DIR}/unsloth/gemma-GGUF"
model-file = "gemma.gguf"
mmproj-file = "mmproj.gguf"

[models.gemma.fetch.hf]
arguments = [
  "${repo}",
  "${model-file}",
  "${mmproj-file}",
  "--local-dir ${local-dir}",
]

[models.gemma.serve.llama-server]
model = "${local-dir}/${model-file}"
mmproj = "${local-dir}/${mmproj-file}"
""",
  )
  instance = load_instance(settings, 'desktop')

  output = fetch_models(instance, ('gemma',), settings, dry_run=True)

  assert output == [
    f'hf download unsloth/gemma-GGUF gemma.gguf mmproj.gguf --local-dir {model_dir.as_posix()}/unsloth/gemma-GGUF'
  ]


def test_fetch_passes_configured_hf_token(tmp_path: Path, monkeypatch) -> None:
  calls = []
  settings, _ = make_config_repo(
    tmp_path,
    """
[models.gemma]
repo = "unsloth/gemma-GGUF"
local-dir = "${LLML_MODEL_DIR}/unsloth/gemma-GGUF"
model-file = "gemma.gguf"

[models.gemma.fetch.hf]
arguments = [
  "${repo}",
  "${model-file}",
  "--local-dir ${local-dir}",
]

[models.gemma.serve.llama-server]
model = "${local-dir}/${model-file}"
""",
  )
  settings = Settings(
    user_config_path=settings.user_config_path,
    config_uri=settings.config_uri,
    config_dir=settings.config_dir,
    model_dir=settings.model_dir,
    hf_token='secret-token',
  )

  def fake_snapshot_download(**kwargs):
    calls.append(kwargs)
    return str(tmp_path / 'downloaded')

  monkeypatch.setattr('llml.actions.snapshot_download', fake_snapshot_download)
  instance = load_instance(settings, 'desktop')

  fetch_models(instance, ('gemma',), settings, dry_run=False)

  assert calls[0]['token'] == 'secret-token'
