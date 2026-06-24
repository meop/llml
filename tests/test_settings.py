from llml.settings import load_settings


def test_load_settings_uses_new_env_names(tmp_path, monkeypatch) -> None:
  monkeypatch.setenv('LLML_CONFIG_URI', 'https://example.test/config.git')
  monkeypatch.setenv('LLML_CONFIG_DIR', str(tmp_path / 'config'))
  monkeypatch.setenv('LLML_MODEL_DIR', str(tmp_path / 'models'))
  monkeypatch.setenv('LLML_HF_TOKEN', 'hf_test')

  settings = load_settings(tmp_path / 'missing.toml')

  assert settings.config_uri == 'https://example.test/config.git'
  assert settings.config_dir == tmp_path / 'config'
  assert settings.model_dir == tmp_path / 'models'
  assert settings.hf_token == 'hf_test'
