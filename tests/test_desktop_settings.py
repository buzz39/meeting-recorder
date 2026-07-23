import json

from config import Config
from desktop_settings import (
    apply_desktop_settings,
    has_completed_setup,
    load_settings,
    mark_setup_complete,
    save_settings,
)


def test_desktop_settings_round_trip_without_secrets(tmp_path):
    path = tmp_path / "settings.json"
    config = Config(
        output_dir="D:/Meetings",
        model_size="base",
        transcription_provider="openai",
        transcription_api_key="do-not-persist",
        speaker_count=2,
        device_index=7,
        device_name="Speakers (loopback)",
    )

    save_settings(config, path)
    loaded = load_settings(path)

    assert loaded["output_dir"] == "D:/Meetings"
    assert loaded["speaker_count"] == 2
    assert loaded["device_index"] == 7
    assert loaded["device_name"] == "Speakers (loopback)"
    assert "transcription_api_key" not in loaded
    assert "do-not-persist" not in path.read_text(encoding="utf-8")


def test_apply_desktop_settings_ignores_unknown_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "settings": {
                    "model_size": "tiny",
                    "unknown": "ignored",
                },
            }
        ),
        encoding="utf-8",
    )

    config = apply_desktop_settings(Config(), path)

    assert config.model_size == "tiny"
    assert config.output_dir == str(tmp_path / "Documents" / "Meeting Recorder")
    assert config.output_format == "all"
    assert not hasattr(config, "unknown")


def test_invalid_settings_are_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")

    assert load_settings(path) == {}


def test_setup_marker_round_trip(tmp_path):
    marker = tmp_path / "setup-complete"

    assert not has_completed_setup(marker)
    mark_setup_complete(marker)
    assert has_completed_setup(marker)
