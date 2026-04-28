"""
Transcription using either local faster-whisper (CTranslate2) or a cloud API.

faster-whisper is 5-10x faster than openai-whisper and uses less memory.
Models are downloaded automatically on first use.
"""

import json
import mimetypes
import os
import tempfile
import uuid
import wave
from urllib import request
from urllib.error import HTTPError, URLError

import numpy as np

from config import DEFAULT_TRANSCRIPTION_MODEL, Config, is_cloud_transcription_provider


def _detect_device_and_compute(preferred_compute: str) -> tuple[str, str]:
    """Pick the best available device for faster-whisper.

    Returns (device, compute_type). CUDA is preferred when available; otherwise
    CPU is used. faster-whisper does not currently support Apple MPS, so on
    macOS we stay on CPU. ``preferred_compute`` is honoured when compatible
    with the chosen device, with sane fallbacks otherwise (CTranslate2 only
    supports float16/int8_float16 on CUDA, not on CPU).
    """
    try:
        import torch  # noqa: F401 — only imported to probe CUDA availability
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if cuda_available:
        # On GPU, float16 is the typical fast path; honour explicit overrides.
        compute = preferred_compute if preferred_compute in (
            "float16", "int8_float16", "int8", "float32"
        ) else "float16"
        return "cuda", compute

    # CPU path — float16 is not supported by CTranslate2 on CPU.
    compute = preferred_compute if preferred_compute in (
        "int8", "int8_float32", "float32"
    ) else "int8"
    return "cpu", compute


class Transcriber:
    """Wraps local faster-whisper or cloud speech-to-text."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.provider = (config.transcription_provider or "local").lower()

    def load_model(self):
        """Load the Whisper model. Call once at startup."""
        if self._is_cloud_provider():
            if not self._cloud_api_key():
                raise RuntimeError(
                    "A transcription API key is required for cloud providers. "
                    "Set TRANSCRIPTION_API_KEY, AI_GATEWAY_API_KEY, or OPENAI_API_KEY."
                )
            if self.provider == "compatible" and not self.config.transcription_base_url:
                raise RuntimeError("--transcription-base-url or TRANSCRIPTION_BASE_URL is required for compatible provider")
            print(
                f"☁️  Using {self.provider} transcription API "
                f"({self._cloud_model()}); no local model to load."
            )
            return
        if self.provider != "local":
            raise RuntimeError(f"Unsupported transcription provider: {self.provider}")

        # Imported lazily so simply importing this module (e.g. for unit tests
        # of CLI helpers) does not require the heavy faster-whisper dependency.
        from faster_whisper import WhisperModel

        device, compute_type = _detect_device_and_compute(self.config.compute_type)
        print(
            f"📦 Loading Whisper model '{self.config.model_size}' "
            f"(device: {device}, compute: {compute_type})..."
        )
        self.model = WhisperModel(
            self.config.model_size,
            device=device,
            compute_type=compute_type,
        )
        print("✅ Model loaded.")

    def transcribe(self, audio_chunk, chunk_offset: float = 0.0) -> list[dict]:
        """Transcribe a numpy float32 audio array.

        Args:
            audio_chunk: numpy array of float32 audio at 16kHz
            chunk_offset: time offset (seconds) to add to segment timestamps

        Returns:
            List of segments: [{"start": float, "end": float, "text": str}, ...]
        """
        if self.model is None:
            self.load_model()

        if self._is_cloud_provider():
            return self._transcribe_audio_cloud(audio_chunk, chunk_offset)

        segments, info = self.model.transcribe(
            audio_chunk,
            beam_size=5,
            language=self.config.language,
            vad_filter=True,  # Filter out non-speech
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        results = []
        for seg in segments:
            results.append({
                "start": seg.start + chunk_offset,
                "end": seg.end + chunk_offset,
                "text": seg.text.strip(),
            })
        return results

    def transcribe_file(self, filepath: str) -> list[dict]:
        """Transcribe an audio file from disk."""
        if self.model is None:
            self.load_model()

        if self._is_cloud_provider():
            return self._transcribe_file_cloud(filepath)

        segments, info = self.model.transcribe(
            filepath,
            beam_size=5,
            language=self.config.language,
            vad_filter=True,
        )

        results = []
        for seg in segments:
            results.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })
        return results

    def _is_cloud_provider(self) -> bool:
        return is_cloud_transcription_provider(self.provider)

    def _cloud_api_key(self) -> str | None:
        # Config.transcription_api_key already applies the env fallback order
        # TRANSCRIPTION_API_KEY > AI_GATEWAY_API_KEY > OPENAI_API_KEY. The
        # openai_api_key fallback is kept for legacy direct Config mutation.
        return self.config.transcription_api_key or self.config.openai_api_key

    def _cloud_model(self) -> str:
        # Backwards compatibility: code written before the generic cloud
        # provider setting may still set only Config.openai_model directly.
        if (
            self.config.transcription_model == DEFAULT_TRANSCRIPTION_MODEL
            and self.config.openai_model != DEFAULT_TRANSCRIPTION_MODEL
        ):
            return self.config.openai_model
        # Keep a final default in case callers mutate Config fields to None
        # after dataclass initialization.
        return self.config.transcription_model or self.config.openai_model or DEFAULT_TRANSCRIPTION_MODEL

    def _cloud_base_url(self) -> str:
        if self.config.transcription_base_url:
            return self.config.transcription_base_url
        if self.provider == "vercel":
            return "https://ai-gateway.vercel.sh/v1"
        if self.provider == "compatible":
            raise RuntimeError("--transcription-base-url or TRANSCRIPTION_BASE_URL is required for compatible provider")
        return "https://api.openai.com/v1"

    def _transcribe_audio_cloud(self, audio_chunk: np.ndarray, chunk_offset: float) -> list[dict]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _write_mono_wav(tmp_path, audio_chunk, self.config.sample_rate)
            segments = self._transcribe_file_cloud(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        for seg in segments:
            seg["start"] += chunk_offset
            seg["end"] += chunk_offset
        return segments

    def _transcribe_file_cloud(self, filepath: str) -> list[dict]:
        payload = self._cloud_transcription_request(filepath)
        segments = payload.get("segments") or []
        if segments:
            return [
                {
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", seg.get("start", 0.0))),
                    "text": str(seg.get("text", "")).strip(),
                }
                for seg in segments
                if str(seg.get("text", "")).strip()
            ]

        text = str(payload.get("text", "")).strip()
        if not text:
            return []
        return [{"start": 0.0, "end": max(_wav_duration(filepath), 0.001), "text": text}]

    def _cloud_transcription_request(self, filepath: str) -> dict:
        fields = {
            "model": self._cloud_model(),
            "response_format": "verbose_json",
        }
        if self.config.language:
            fields["language"] = self.config.language

        body, content_type = _multipart_form_data(fields, "file", filepath)
        req = request.Request(
            _transcription_endpoint(self._cloud_base_url()),
            data=body,
            headers={
                "Authorization": f"Bearer {self._cloud_api_key()}",
                "Content-Type": content_type,
            },
            method="POST",
        )
        try:
            # Cloud transcription of longer recorded files can legitimately take
            # longer than normal API calls, so use a generous fixed timeout.
            with request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{self.provider} transcription failed ({e.code}): {detail}") from e
        except URLError as e:
            raise RuntimeError(f"{self.provider} transcription request failed: {e}") from e


def _write_mono_wav(filepath: str, audio: np.ndarray, sample_rate: int):
    audio = np.asarray(audio, dtype=np.float32)
    audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.tobytes())


def _wav_duration(filepath: str) -> float:
    try:
        with wave.open(filepath, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except (wave.Error, OSError, EOFError):
        return 0.0


def _transcription_endpoint(base_url: str) -> str:
    """Build the transcription endpoint from either a base URL or full endpoint.

    If the normalized URL already ends exactly at ``/audio/transcriptions``,
    it is treated as a complete endpoint and returned unchanged.
    """
    normalized = base_url.rstrip("/")
    if normalized.endswith("/audio/transcriptions"):
        return normalized
    return f"{normalized}/audio/transcriptions"


def _multipart_form_data(fields: dict[str, str], file_field: str, filepath: str) -> tuple[bytes, str]:
    boundary = f"----form-boundary-{uuid.uuid4().hex}"
    parts = []
    for name, value in fields.items():
        parts.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            str(value).encode("utf-8"),
            b"\r\n",
        ])

    filename = os.path.basename(filepath)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    parts.extend([
        f"--{boundary}\r\n".encode(),
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\n'
        ).encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
