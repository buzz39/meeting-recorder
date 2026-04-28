"""Configuration defaults for Meeting Recorder."""

import os
from dataclasses import dataclass, field

DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"
CLOUD_TRANSCRIPTION_PROVIDERS = ("openai", "vercel", "compatible")


def is_cloud_transcription_provider(provider: str) -> bool:
    return provider in CLOUD_TRANSCRIPTION_PROVIDERS


def _env_first(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment value, or default if none exist.

    Empty strings are treated as unset so accidental blank environment variables
    do not block later fallbacks.
    """
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


@dataclass
class Config:
    # Whisper model size — tradeoffs:
    #   tiny   (~75MB)  — fastest, lowest accuracy, ~1GB RAM
    #   base   (~145MB) — fast, decent accuracy, ~1GB RAM
    #   small  (~488MB) — good balance of speed & accuracy, ~2GB RAM (default)
    #   medium (~1.5GB) — slow, high accuracy, ~5GB RAM
    #   large-v2 (~3GB) — slowest, best accuracy, ~10GB RAM
    model_size: str = "small"

    # Output directory for recordings
    output_dir: str = os.path.join(os.path.dirname(__file__), "recordings")

    # Output format: txt, srt, or all (both txt and srt)
    output_format: str = "txt"

    # HuggingFace token for pyannote speaker diarization
    # Set via HF_TOKEN env var or here directly
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))

    # Transcription provider: local (faster-whisper), openai, vercel, or compatible.
    # Cloud mode avoids local model downloads/cold starts on low-resource PCs.
    transcription_provider: str = field(default_factory=lambda: os.environ.get("TRANSCRIPTION_PROVIDER", "local"))
    # API key priority: TRANSCRIPTION_API_KEY > AI_GATEWAY_API_KEY > OPENAI_API_KEY.
    transcription_api_key: str | None = field(
        default_factory=lambda: _env_first("TRANSCRIPTION_API_KEY", "AI_GATEWAY_API_KEY", "OPENAI_API_KEY")
    )
    transcription_model: str = field(
        default_factory=lambda: _env_first(
            "TRANSCRIPTION_MODEL",
            "OPENAI_TRANSCRIBE_MODEL",
            default=DEFAULT_TRANSCRIPTION_MODEL,
        )
    )
    transcription_base_url: str | None = field(default_factory=lambda: os.environ.get("TRANSCRIPTION_BASE_URL"))
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: os.environ.get("OPENAI_TRANSCRIBE_MODEL", DEFAULT_TRANSCRIPTION_MODEL))

    # Audio chunk duration in seconds for processing
    chunk_duration: float = 30.0

    # Sample rate (16kHz is what Whisper expects)
    sample_rate: int = 16000

    # Number of audio channels (mono for transcription)
    channels: int = 1

    # Whisper language (None = auto-detect)
    language: str | None = None

    # Minimum silence duration (seconds) to detect speaker change
    speaker_change_silence: float = 0.5

    # Optional known/exact speaker count and maximum speaker cap for diarization.
    # Use speaker_count=2 for two-person calls to prevent over-splitting.
    speaker_count: int | None = None
    max_speakers: int = 10

    # Energy threshold (RMS) below which audio is considered silence
    silence_threshold: float = 0.01

    # Minimum segment energy to consider speech
    min_speech_energy: float = 0.02

    # Device index for audio capture (None = default loopback)
    device_index: int | None = None

    # Optional microphone mix. WASAPI loopback captures the other side of a call;
    # enabling this also records/transcribes your local microphone.
    include_microphone: bool = False
    microphone_device_index: int | None = None
    microphone_gain: float = 1.0

    # Compute type for faster-whisper: int8, float16, float32
    compute_type: str = "int8"
