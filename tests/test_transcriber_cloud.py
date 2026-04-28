import json

import numpy as np

from config import Config
from transcriber import Transcriber, _multipart_form_data


def test_openai_provider_load_model_does_not_require_local_whisper():
    cfg = Config()
    cfg.transcription_provider = "openai"
    cfg.openai_api_key = "test-key"
    transcriber = Transcriber(cfg)

    transcriber.load_model()

    assert transcriber.model is None


def test_multipart_form_data_contains_fields_and_file(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")

    body, content_type = _multipart_form_data({"model": "whisper-1"}, "file", str(audio))

    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="model"' in body
    assert b"whisper-1" in body
    assert b'name="file"; filename="audio.wav"' in body
    assert b"RIFF" in body


def test_openai_audio_chunk_offsets_segments(monkeypatch):
    cfg = Config()
    cfg.transcription_provider = "openai"
    cfg.openai_api_key = "test-key"
    transcriber = Transcriber(cfg)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({
                "segments": [{"start": 0.25, "end": 0.75, "text": "hello"}],
            }).encode("utf-8")

    monkeypatch.setattr("transcriber.request.urlopen", lambda req, timeout: Response())

    segments = transcriber.transcribe(np.zeros(16000, dtype=np.float32), chunk_offset=10.0)

    assert segments == [{"start": 10.25, "end": 10.75, "text": "hello"}]
