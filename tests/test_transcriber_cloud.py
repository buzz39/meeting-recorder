import json

import numpy as np

from config import Config
from transcriber import Transcriber, _multipart_form_data, _transcription_endpoint


def test_openai_provider_load_model_does_not_require_local_whisper():
    cfg = Config()
    cfg.transcription_provider = "openai"
    cfg.openai_api_key = "test-key"
    transcriber = Transcriber(cfg)

    transcriber.load_model()

    assert transcriber.model is None


def test_vercel_provider_load_model_does_not_require_local_whisper():
    cfg = Config()
    cfg.transcription_provider = "vercel"
    cfg.transcription_api_key = "test-key"
    cfg.transcription_model = "openai/whisper-1"
    transcriber = Transcriber(cfg)

    transcriber.load_model()

    assert transcriber.model is None


def test_legacy_openai_model_is_used_when_new_model_is_default():
    cfg = Config()
    cfg.transcription_provider = "openai"
    cfg.transcription_api_key = "test-key"
    cfg.openai_model = "legacy-whisper"
    transcriber = Transcriber(cfg)

    assert transcriber._cloud_model() == "legacy-whisper"


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


def test_vercel_provider_uses_gateway_endpoint_and_model(monkeypatch):
    cfg = Config()
    cfg.transcription_provider = "vercel"
    cfg.transcription_api_key = "vercel-key"
    cfg.transcription_model = "provider/transcribe-model"
    transcriber = Transcriber(cfg)
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"text": "hello"}).encode("utf-8")

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = req.data
        return Response()

    monkeypatch.setattr("transcriber.request.urlopen", fake_urlopen)

    segments = transcriber.transcribe(np.zeros(16000, dtype=np.float32))

    assert seen["url"] == "https://ai-gateway.vercel.sh/v1/audio/transcriptions"
    assert seen["auth"] == "Bearer vercel-key"
    assert b"provider/transcribe-model" in seen["body"]
    assert segments == [{"start": 0.0, "end": 1.0, "text": "hello"}]


def test_transcription_endpoint_accepts_full_endpoint():
    endpoint = "https://example.com/v1/audio/transcriptions"

    assert _transcription_endpoint(endpoint) == endpoint
