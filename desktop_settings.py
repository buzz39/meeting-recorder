"""Persistent settings for the Windows desktop application."""

import json
import os
from pathlib import Path
from typing import Any

from config import Config

SETTINGS_VERSION = 1
PERSISTED_FIELDS = (
    "model_size",
    "output_dir",
    "output_format",
    "transcription_provider",
    "transcription_model",
    "transcription_base_url",
    "language",
    "speaker_count",
    "max_speakers",
    "device_index",
    "include_microphone",
    "microphone_device_index",
    "microphone_gain",
)


def settings_dir() -> Path:
    """Return the per-user configuration directory."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "Meeting Recorder"
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "meeting-recorder"


def default_recordings_dir() -> Path:
    """Return a writable, user-facing default recording directory."""
    documents = Path(os.environ.get("USERPROFILE", Path.home())) / "Documents"
    return documents / "Meeting Recorder"


def settings_path() -> Path:
    return settings_dir() / "settings.json"


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Load valid settings, returning an empty mapping for missing/corrupt files."""
    path = path or settings_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != SETTINGS_VERSION:
        return {}
    values = payload.get("settings")
    if not isinstance(values, dict):
        return {}
    return {key: values[key] for key in PERSISTED_FIELDS if key in values}


def save_settings(config: Config, path: Path | None = None) -> None:
    """Atomically persist non-secret desktop settings."""
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {field: getattr(config, field) for field in PERSISTED_FIELDS}
    payload = {"version": SETTINGS_VERSION, "settings": values}
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def apply_desktop_settings(config: Config, path: Path | None = None) -> Config:
    """Apply saved settings and desktop-safe defaults to a config object."""
    config.output_dir = str(default_recordings_dir())
    config.output_format = "all"
    for field, value in load_settings(path).items():
        setattr(config, field, value)
    return config


def has_completed_setup(path: Path | None = None) -> bool:
    path = path or settings_dir() / "setup-complete"
    return path.is_file()


def mark_setup_complete(path: Path | None = None) -> None:
    path = path or settings_dir() / "setup-complete"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Recording consent notice accepted.\n", encoding="utf-8")
