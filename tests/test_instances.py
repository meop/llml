from pathlib import Path

import pytest

from llml.errors import CliError
from llml.instances import (
  find_instances,
  installed_instances,
  load_instance,
  model_values,
  pinpoint_installed_model,
  pinpoint_instance,
  wide_models,
  write_models_ini,
)
from llml.settings import Settings


def make_settings(tmp_path: Path, body: str, instances: dict[str, str] | None = None) -> Settings:
  config_dir = tmp_path / 'config'
  instances_dir = config_dir / 'instances'
  instances_dir.mkdir(parents=True)
  (config_dir / '.git').mkdir()
  (instances_dir / 'desktop.toml').write_text(body, encoding='utf-8')
  for name, instance_body in (instances or {}).items():
    (instances_dir / f'{name}.toml').write_text(instance_body, encoding='utf-8')
  return Settings(
    user_config_path=tmp_path / 'config.toml',
    config_uri='unused',
    config_dir=config_dir,
    model_dir=tmp_path / 'models',
    hf_token=None,
  )


def make_find_settings(tmp_path: Path) -> Settings:
  return make_settings(
    tmp_path,
    """
[models.gemma]
local-dir = "x"
[models.qwen]
local-dir = "y"
""",
    instances={
      'laptop': """
[models.gemma]
local-dir = "x"
[models.phi]
local-dir = "z"
""",
    },
  )


def test_find_no_terms_dumps_every_instance_and_model(tmp_path: Path) -> None:
  settings = make_find_settings(tmp_path)

  assert find_instances(settings, ()) == [
    ('desktop', ['gemma', 'qwen']),
    ('laptop', ['gemma', 'phi']),
  ]


def test_find_instance_term_keeps_all_its_models(tmp_path: Path) -> None:
  settings = make_find_settings(tmp_path)

  assert find_instances(settings, ('desk',)) == [('desktop', ['gemma', 'qwen'])]


def test_find_model_term_spans_instances(tmp_path: Path) -> None:
  settings = make_find_settings(tmp_path)

  assert find_instances(settings, ('gemma',)) == [
    ('desktop', ['gemma']),
    ('laptop', ['gemma']),
  ]


def test_find_terms_are_order_independent_and_anded(tmp_path: Path) -> None:
  settings = make_find_settings(tmp_path)

  expected = [('desktop', ['gemma'])]
  assert find_instances(settings, ('desk', 'gemma')) == expected
  assert find_instances(settings, ('gemma', 'desk')) == expected


def test_find_is_case_insensitive(tmp_path: Path) -> None:
  settings = make_find_settings(tmp_path)

  assert find_instances(settings, ('DESK', 'GEMMA')) == [('desktop', ['gemma'])]


INSTALLED_BODY = """
[models.gemma]
local-dir = "${LLML_MODEL_DIR}/gemma"
[models.gemma.serve.llama-server]
model = "${local-dir}/gemma.gguf"

[models.qwen]
local-dir = "${LLML_MODEL_DIR}/qwen"
[models.qwen.serve.llama-server]
model = "${local-dir}/qwen.gguf"
"""


def install(settings: Settings, *relative_files: str) -> None:
  for relative in relative_files:
    path = settings.model_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('', encoding='utf-8')


def test_installed_includes_only_models_present_on_disk(tmp_path: Path) -> None:
  settings = make_settings(tmp_path, INSTALLED_BODY)
  install(settings, 'gemma/gemma.gguf')

  assert installed_instances(settings, ()) == [('desktop', ['gemma'])]


def test_installed_applies_the_same_term_filter(tmp_path: Path) -> None:
  settings = make_settings(tmp_path, INSTALLED_BODY)
  install(settings, 'gemma/gemma.gguf', 'qwen/qwen.gguf')

  assert installed_instances(settings, ('qwen',)) == [('desktop', ['qwen'])]


def test_installed_omits_instances_with_nothing_installed(tmp_path: Path) -> None:
  settings = make_settings(
    tmp_path,
    INSTALLED_BODY,
    instances={
      'laptop': """
[models.phi]
local-dir = "${LLML_MODEL_DIR}/phi"
[models.phi.serve.llama-server]
model = "${local-dir}/phi.gguf"
""",
    },
  )
  install(settings, 'gemma/gemma.gguf')

  assert installed_instances(settings, ()) == [('desktop', ['gemma'])]


WIDE_BODY = """
[models.gemma]
local-dir = "x"
[models.gemma-lite]
local-dir = "y"
[models.qwen]
local-dir = "z"
"""


def make_wide_settings(tmp_path: Path) -> Settings:
  return make_settings(tmp_path, WIDE_BODY, instances={'desktop-lite': WIDE_BODY})


def test_wide_models_no_terms_returns_every_pair(tmp_path: Path) -> None:
  settings = make_wide_settings(tmp_path)

  assert wide_models(settings, ()) == [
    ('desktop', 'gemma'),
    ('desktop', 'gemma-lite'),
    ('desktop', 'qwen'),
    ('desktop-lite', 'gemma'),
    ('desktop-lite', 'gemma-lite'),
    ('desktop-lite', 'qwen'),
  ]


def test_wide_models_term_does_not_short_circuit_on_exact(tmp_path: Path) -> None:
  settings = make_wide_settings(tmp_path)

  # 'gemma' is an exact model name but WIDE still keeps every substring match.
  assert wide_models(settings, ('gemma',)) == [
    ('desktop', 'gemma'),
    ('desktop', 'gemma-lite'),
    ('desktop-lite', 'gemma'),
    ('desktop-lite', 'gemma-lite'),
  ]


def test_wide_models_ands_terms(tmp_path: Path) -> None:
  settings = make_wide_settings(tmp_path)

  assert wide_models(settings, ('lite', 'gemma')) == [
    ('desktop', 'gemma-lite'),
    ('desktop-lite', 'gemma'),
    ('desktop-lite', 'gemma-lite'),
  ]


PINPOINT_BODY = """
[models.gemma]
local-dir = "${LLML_MODEL_DIR}/gemma"
[models.gemma.serve.llama-server]
model = "${local-dir}/gemma.gguf"
[models.gemma-lite]
local-dir = "${LLML_MODEL_DIR}/gemma-lite"
[models.gemma-lite.serve.llama-server]
model = "${local-dir}/gemma-lite.gguf"
[models.qwen]
local-dir = "${LLML_MODEL_DIR}/qwen"
[models.qwen.serve.llama-server]
model = "${local-dir}/qwen.gguf"
"""


def make_pinpoint_settings(tmp_path: Path) -> Settings:
  settings = make_settings(tmp_path, PINPOINT_BODY, instances={'desktop-lite': PINPOINT_BODY})
  install(settings, 'gemma/gemma.gguf', 'gemma-lite/gemma-lite.gguf', 'qwen/qwen.gguf')
  return settings


def test_pinpoint_model_prefers_exact_over_substring(tmp_path: Path) -> None:
  settings = make_pinpoint_settings(tmp_path)

  assert pinpoint_installed_model(settings, ('gemma',)) == ('desktop', 'gemma')


def test_pinpoint_model_no_exact_takes_first_by_stable_sort(tmp_path: Path) -> None:
  settings = make_pinpoint_settings(tmp_path)

  assert pinpoint_installed_model(settings, ('gem',)) == ('desktop', 'gemma')


def test_pinpoint_model_ands_terms_then_pinpoints(tmp_path: Path) -> None:
  settings = make_pinpoint_settings(tmp_path)

  # 'lite' + 'gemma' narrows to lite-instance pairs; exact model 'gemma' then wins.
  assert pinpoint_installed_model(settings, ('lite', 'gemma')) == ('desktop-lite', 'gemma')


def test_pinpoint_model_skips_uninstalled(tmp_path: Path) -> None:
  settings = make_settings(tmp_path, PINPOINT_BODY)
  install(settings, 'qwen/qwen.gguf')

  with pytest.raises(CliError, match='no installed model matches'):
    pinpoint_installed_model(settings, ('gemma',))


def test_pinpoint_instance_prefers_exact(tmp_path: Path) -> None:
  settings = make_wide_settings(tmp_path)

  assert pinpoint_instance(settings, ('desktop',)) == 'desktop'


def test_pinpoint_instance_no_exact_takes_first(tmp_path: Path) -> None:
  settings = make_wide_settings(tmp_path)

  assert pinpoint_instance(settings, ('desk',)) == 'desktop'


def test_pinpoint_instance_unique_substring(tmp_path: Path) -> None:
  settings = make_wide_settings(tmp_path)

  assert pinpoint_instance(settings, ('lite',)) == 'desktop-lite'


def test_pinpoint_instance_no_match_errors(tmp_path: Path) -> None:
  settings = make_wide_settings(tmp_path)

  with pytest.raises(CliError, match='no instance matches'):
    pinpoint_instance(settings, ('nope',))


def test_model_values_expand_shared_model_variables(tmp_path: Path) -> None:
  settings = make_settings(
    tmp_path,
    """
[models.gemma]
repo = "unsloth/gemma-GGUF"
local-dir = "${LLML_MODEL_DIR}/unsloth/gemma-GGUF"
model-file = "gemma.gguf"

[models.gemma.sync.hf]
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

[models.present.sync.hf]
arguments = []

[models.present.serve.llama-server]
model = "${local-dir}/${model-file}"
temperature = 1.0

[models.missing]
local-dir = "${LLML_MODEL_DIR}/missing"
model-file = "missing.gguf"

[models.missing.sync.hf]
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
