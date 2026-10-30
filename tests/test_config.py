"""Model configuration, isolated from real env files and secrets."""

import pytest

from fusion_memory import config


@pytest.fixture
def isolated_config_env(monkeypatch):
    # Do not load .env or read the process's actual credential environment.
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: False)
    environment = {}
    monkeypatch.setattr(config.os, "environ", environment)
    return environment


def test_embedding_dimensions_default_to_1024(isolated_config_env):
    assert config.Settings.from_env().embedding_dimensions == 1024


@pytest.mark.parametrize("value", ["1024", "1536"])
def test_embedding_dimensions_read_openai_environment_variable(isolated_config_env, value):
    isolated_config_env["OPENAI_EMBEDDING_DIMENSIONS"] = value
    assert config.Settings.from_env().embedding_dimensions == int(value)


@pytest.mark.parametrize("value", ["0", "-1024"])
def test_nonpositive_embedding_dimensions_reject_configuration(isolated_config_env, value):
    isolated_config_env["OPENAI_EMBEDDING_DIMENSIONS"] = value
    with pytest.raises(ValueError, match="OPENAI_EMBEDDING_DIMENSIONS.*positive"):
        config.Settings.from_env()


def test_noninteger_embedding_dimensions_reject_configuration(isolated_config_env):
    isolated_config_env["OPENAI_EMBEDDING_DIMENSIONS"] = "not-an-integer"
    with pytest.raises(ValueError):
        config.Settings.from_env()


def test_extraction_output_defaults_to_auto(isolated_config_env):
    assert config.Settings().extraction_output_mode == "auto"
    assert config.Settings.from_env().extraction_output_mode == "auto"


@pytest.mark.parametrize("value", ["auto", "strict", "json", "AUTO", "STRICT", "Json"])
def test_extraction_output_mode_from_environment(isolated_config_env, value):
    isolated_config_env["EXTRACTION_OUTPUT_MODE"] = value
    assert config.Settings.from_env().extraction_output_mode == value.lower()


@pytest.mark.parametrize("mode", ["auto", "strict", "json"])
def test_extraction_output_mode_accepts_explicit_supported_configuration(mode):
    assert config.Settings(extraction_output_mode=mode).extraction_output_mode == mode


@pytest.mark.parametrize("value", ["", "legacy", "json_schema", "PRIVATE-INVALID-CONFIG-VALUE"])
def test_invalid_extraction_output_mode_is_rejected_without_echoing_value(isolated_config_env, value):
    isolated_config_env["EXTRACTION_OUTPUT_MODE"] = value
    with pytest.raises(ValueError, match="EXTRACTION_OUTPUT_MODE must be auto, strict, or json") as caught:
        config.Settings.from_env()
    if value:
        assert value not in str(caught.value)
