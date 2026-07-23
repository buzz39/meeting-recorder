import pytest

from config import Config
from desktop_ui import apply_form_values


def form_values(**overrides):
    values = {
        "output_dir": "D:/Meetings",
        "output_format": "all",
        "model_size": "base",
        "language": "en",
        "speaker_count": "2",
        "max_speakers": "4",
        "device_index": 3,
        "include_microphone": False,
        "microphone_device_index": 5,
        "microphone_gain": "1.5",
        "transcription_provider": "local",
        "transcription_model": "whisper-1",
        "transcription_base_url": "",
    }
    values.update(overrides)
    return values


def test_apply_form_values_updates_config():
    config = Config()

    apply_form_values(config, form_values())

    assert config.output_dir == "D:/Meetings"
    assert config.model_size == "base"
    assert config.speaker_count == 2
    assert config.max_speakers == 4
    assert config.device_index == 3
    assert config.include_microphone is False
    assert config.microphone_gain == 1.5


def test_apply_form_values_allows_automatic_language_and_speakers():
    config = Config()

    apply_form_values(config, form_values(language="", speaker_count=""))

    assert config.language is None
    assert config.speaker_count is None


def test_apply_form_values_expands_maximum_for_exact_speaker_count():
    config = Config()

    apply_form_values(config, form_values(speaker_count="6", max_speakers="4"))

    assert config.speaker_count == 6
    assert config.max_speakers == 6


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"output_dir": ""}, "recordings folder"),
        ({"microphone_gain": "0"}, "Microphone gain"),
        ({"speaker_count": "-1"}, "Speaker count"),
        ({"max_speakers": ""}, "Maximum speakers"),
    ],
)
def test_apply_form_values_rejects_invalid_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        apply_form_values(Config(), form_values(**overrides))
