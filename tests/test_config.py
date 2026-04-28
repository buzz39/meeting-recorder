from config import get_first_env_var


def test_get_first_env_var_skips_empty_values(monkeypatch):
    monkeypatch.setenv("TRANSCRIPTION_MODEL", "")
    monkeypatch.setenv("OPENAI_TRANSCRIBE_MODEL", "fallback-model")

    assert get_first_env_var("TRANSCRIPTION_MODEL", "OPENAI_TRANSCRIBE_MODEL") == "fallback-model"
