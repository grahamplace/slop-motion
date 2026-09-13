"""Configuration compatibility through the CLI and persisted projects."""

import json

import pytest

from stop_motion.cli import main
from stop_motion.project import Project
from stop_motion.settings import resolve_settings
from stop_motion.storage import HarnessError, write_json


def test_cli_accepts_provider_options_and_legacy_aliases(tmp_path, capsys):
    brief = tmp_path / "brief.txt"
    brief.write_text("A robot waves.")
    options = tmp_path / "options.json"
    write_json(options, {"backend": "images", "quality": "high"})
    args = ["init", str(tmp_path / "build"), "--brief-file", str(brief), "--provider", "openai"]
    assert main([*args, "--provider-options", str(options), "--quality", "high"]) == 0
    settings = json.loads(capsys.readouterr().out)["settings"]
    assert settings["provider"] == "openai"
    assert settings["provider_options"]["backend"] == "images"
    assert settings["provider_options"]["quality"] == "high"
    assert "quality" not in settings


@pytest.mark.parametrize(
    "settings",
    [
        {"provider": "unknown"},
        {"provider": []},
        {"provider_options": []},
        {"provider_options": {"api_key": "never-store-this"}},
        {"quality": "low", "provider_options": {"quality": "high"}},
        {"provider_options": {"quality": []}},
        {"provider_options": {"backend": None}},
        {"provider_options": {"driver_model": ""}},
        {"model": []},
    ],
)
def test_invalid_provider_settings_fail_before_project_creation(tmp_path, settings):
    root = tmp_path / "build"
    with pytest.raises(HarnessError):
        Project.create(root, "A robot", **settings)
    assert not root.exists()


def test_defaults_and_aliases_normalize_without_mutating_input():
    original = {"quality": "high", "backend": "responses"}
    normalized = resolve_settings(original)
    assert original == {"quality": "high", "backend": "responses"}
    assert normalized == resolve_settings({"provider": "openai", "provider_options": original})
    assert normalized["provider_options"]["driver_model"] == "gpt-5.5"
    assert resolve_settings({})["provider_options"]["backend"] == "responses"
    assert resolve_settings({}, legacy_manifest=True)["provider_options"]["backend"] == "images"
