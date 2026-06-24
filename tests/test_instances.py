from pathlib import Path

from llml.instances import load_instance, model_values, write_models_ini
from llml.settings import Settings


def make_settings(tmp_path: Path, body: str) -> Settings:
  config_dir = tmp_path / 'config'
  instances_dir = config_dir / 'instances'
  instances_dir.mkdir(parents=True)
  (config_dir / '.git').mkdir()
  (instances_dir / 'desktop.toml').write_text(body, encoding='utf-8')
  return Settings(
    user_config_path=tmp_path / 'config.toml',
    config_uri='unused',
    config_dir=config_dir,
    model_dir=tmp_path / 'models',
    hf_token=None,
  )


def test_model_values_expand_shared_model_variables(tmp_path: Path) -> None:
  settings = make_settings(
    tmp_path,
    """
[models.gemma]
repo = "unsloth/gemma-GGUF"
local-dir = "${LLML_MODEL_DIR}/unsloth/gemma-GGUF"
model-file = "gemma.gguf"

[models.gemma.fetch.hf]
arguments = []

[models.gemma.serve.llama-server]
model = "${local-dir}/${model-file}"
""",
  )
  instance = load_instance(settings, 'desktop')

  values = model_values(instance['models']['gemma'], settings)

  assert values['repo'] == 'unsloth/gemma-GGUF'
  assert values['model-file'] == 'gemma.gguf'
  assert values['local-dir'] == f'{settings.model_dir.as_posix()}/unsloth/gemma-GGUF'


def test_write_models_ini_only_includes_models_present_on_disk(tmp_path: Path, monkeypatch) -> None:
  monkeypatch.setenv('HOME', str(tmp_path))
  monkeypatch.setenv('USERPROFILE', str(tmp_path))
  settings = make_settings(
    tmp_path,
    """
[serve.llama-server]
arguments = []

[serve.llama-server.defaults]
ctx-size = 32768

[models.present]
local-dir = "${LLML_MODEL_DIR}/present"
model-file = "present.gguf"

[models.present.fetch.hf]
arguments = []

[models.present.serve.llama-server]
model = "${local-dir}/${model-file}"
temperature = 1.0

[models.missing]
local-dir = "${LLML_MODEL_DIR}/missing"
model-file = "missing.gguf"

[models.missing.fetch.hf]
arguments = []

[models.missing.serve.llama-server]
model = "${local-dir}/${model-file}"
temperature = 0.6
""",
  )
  (settings.model_dir / 'present').mkdir(parents=True)
  (settings.model_dir / 'present' / 'present.gguf').write_text('', encoding='utf-8')
  instance = load_instance(settings, 'desktop')

  ini_path = write_models_ini('desktop', instance, settings)
  text = ini_path.read_text(encoding='utf-8')

  assert '[present]' in text
  assert '[missing]' not in text
  assert f'model = {settings.model_dir.as_posix()}/present/present.gguf' in text
