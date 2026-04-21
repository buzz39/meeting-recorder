"""
Local transcription using faster-whisper (CTranslate2).

faster-whisper is 5-10x faster than openai-whisper and uses less memory.
Models are downloaded automatically on first use.
"""

from config import Config


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
    """Wraps faster-whisper for local speech-to-text."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def load_model(self):
        """Load the Whisper model. Call once at startup."""
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
