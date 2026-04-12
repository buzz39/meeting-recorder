"""
Local transcription using faster-whisper (CTranslate2).

faster-whisper is 5-10x faster than openai-whisper and uses less memory.
Models are downloaded automatically on first use.
"""

from faster_whisper import WhisperModel
from config import Config


class Transcriber:
    """Wraps faster-whisper for local speech-to-text."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def load_model(self):
        """Load the Whisper model. Call once at startup."""
        print(f"📦 Loading Whisper model '{self.config.model_size}' (compute: {self.config.compute_type})...")
        self.model = WhisperModel(
            self.config.model_size,
            device="cpu",  # Use "cuda" if GPU available
            compute_type=self.config.compute_type,
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
