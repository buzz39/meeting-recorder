"""
Speaker diarization with pyannote-audio (preferred) or energy-based fallback.

If pyannote-audio is installed and HF_TOKEN is available, uses the
pyannote/speaker-diarization-3.1 pipeline for accurate speaker identification.
Otherwise, falls back to lightweight energy-based speaker change detection.
"""

import numpy as np

from config import Config

# Pyannote / torch imports are deferred — importing ``pyannote.audio`` is very
# expensive (it transitively pulls in torch, torchaudio, lightning,
# speechbrain, etc.) and can take minutes on a cold start. Only pay this cost
# when the pyannote backend is actually selected.
_PYANNOTE_AVAILABLE: bool | None = None  # None = not yet probed
torch = None  # populated by _ensure_pyannote_imports() when available
PyannotePipeline = None  # populated by _ensure_pyannote_imports() when available


def _ensure_pyannote_imports() -> bool:
    """Lazily import torch and pyannote.audio. Returns True if both are available."""
    global _PYANNOTE_AVAILABLE, torch, PyannotePipeline
    if _PYANNOTE_AVAILABLE is None:
        try:
            import torch as _torch
            from pyannote.audio import Pipeline as _PyannotePipeline
            torch = _torch
            PyannotePipeline = _PyannotePipeline
            _PYANNOTE_AVAILABLE = True
        except ImportError:
            _PYANNOTE_AVAILABLE = False
    return _PYANNOTE_AVAILABLE


class EnergyDiarizer:
    """Audio-feature-based speaker change detection (lightweight fallback).

    Detects speaker changes by analyzing silence gaps and audio features
    including RMS energy, spectral centroid, and zero-crossing rate.
    Effective for meetings where people take turns with natural pauses.
    """

    # Weights for [energy, spectral_centroid, zero_crossing_rate].
    # Spectral centroid is weighted highest (3.0) because it captures pitch/timbre
    # differences between speakers. ZCR (2.0) reflects voicing characteristics.
    # Energy (1.0) is weighted lowest as it varies more within a single speaker.
    _FEATURE_WEIGHTS = np.array([1.0, 3.0, 2.0])

    def __init__(self, config: Config):
        self.config = config
        self._current_speaker = 1
        self._speaker_profiles: dict[int, np.ndarray] = {}
        self._max_speakers = 10

    def reset(self):
        self._current_speaker = 1
        self._speaker_profiles = {}

    def assign_speakers(self, audio_chunk: np.ndarray, segments: list[dict]) -> list[dict]:
        """Assign speaker labels based on audio feature analysis."""
        if not segments:
            return segments

        sample_rate = self.config.sample_rate
        prev_end = None

        for seg in segments:
            start_sample = max(0, min(int(seg["start"] * sample_rate), len(audio_chunk) - 1))
            end_sample = max(start_sample + 1, min(int(seg["end"] * sample_rate), len(audio_chunk)))
            segment_audio = audio_chunk[start_sample:end_sample]
            features = self._compute_features(segment_audio)

            if prev_end is not None:
                gap = seg["start"] - prev_end
                if gap >= self.config.speaker_change_silence:
                    best_speaker = self._match_speaker(features)
                    self._current_speaker = best_speaker
            else:
                # First segment: try to match against known profiles
                best_speaker = self._match_speaker(features)
                self._current_speaker = best_speaker

            self._update_profile(self._current_speaker, features)
            seg["speaker"] = f"Speaker {self._current_speaker}"
            prev_end = seg["end"]

        return segments

    def _compute_features(self, audio: np.ndarray) -> np.ndarray:
        """Compute a feature vector: [energy, spectral_centroid, zero_crossing_rate]."""
        if len(audio) == 0:
            return np.zeros(3)

        # RMS energy
        energy = float(np.sqrt(np.mean(audio ** 2)))

        # Spectral centroid (weighted mean of frequencies)
        fft_mag = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / self.config.sample_rate)
        mag_sum = fft_mag.sum()
        if mag_sum > 0:
            spectral_centroid = float(np.sum(freqs * fft_mag) / mag_sum)
        else:
            spectral_centroid = 0.0
        # Normalize centroid to [0, 1] range relative to Nyquist
        spectral_centroid /= (self.config.sample_rate / 2)

        # Zero-crossing rate
        signs = np.sign(audio)
        # Avoid counting zeros as crossings
        signs[signs == 0] = 1
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        zcr = float(crossings) / max(len(audio) - 1, 1)

        return np.array([energy, spectral_centroid, zcr])

    def _match_speaker(self, features: np.ndarray) -> int:
        if not self._speaker_profiles:
            self._speaker_profiles[1] = features.copy()
            return 1

        weights = self._FEATURE_WEIGHTS

        best_id = self._current_speaker
        best_dist = float("inf")
        for spk_id, profile in self._speaker_profiles.items():
            diff = np.abs(features - profile) * weights
            dist = float(np.sum(diff))
            if dist < best_dist:
                best_dist = dist
                best_id = spk_id

        # Determine if this is likely a new speaker.
        # Use a relative threshold based on the average profile magnitude
        # so it adapts to different recording conditions.
        # 0.05 = absolute minimum distance floor (prevents over-splitting in quiet audio)
        # 0.15 = relative sensitivity factor (15% of avg weighted feature magnitude)
        avg_profile_mag = np.mean([float(np.sum(np.abs(p) * weights))
                                   for p in self._speaker_profiles.values()])
        threshold = max(0.05, avg_profile_mag * 0.15)

        if best_dist > threshold and len(self._speaker_profiles) < self._max_speakers:
            new_id = max(self._speaker_profiles.keys()) + 1
            self._speaker_profiles[new_id] = features.copy()
            return new_id

        return best_id

    def _update_profile(self, speaker_id: int, features: np.ndarray):
        if speaker_id in self._speaker_profiles:
            self._speaker_profiles[speaker_id] = (
                0.7 * self._speaker_profiles[speaker_id] + 0.3 * features
            )
        else:
            self._speaker_profiles[speaker_id] = features.copy()


class PyannoteDiarizer:
    """Speaker diarization using pyannote-audio pipeline.

    Uses pyannote/speaker-diarization-3.1 for accurate, neural-network-based
    speaker identification. Requires HF_TOKEN with access to the model.
    """

    def __init__(self, config: Config):
        self.config = config
        self._pipeline = None
        self._speaker_map: dict[str, str] = {}  # pyannote label -> "Speaker N"
        self._next_speaker_id = 1

    def _load_pipeline(self):
        """Load the pyannote pipeline (lazy, first call only)."""
        if self._pipeline is not None:
            return
        token = self.config.hf_token
        if not token:
            raise RuntimeError("HF_TOKEN required for pyannote diarization")
        if not _ensure_pyannote_imports():
            raise RuntimeError("pyannote-audio is not installed")
        print("📦 Loading pyannote speaker-diarization-3.1 pipeline...")
        self._pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
        print("✅ Pyannote pipeline loaded.")

    def reset(self):
        self._speaker_map = {}
        self._next_speaker_id = 1

    def assign_speakers(self, audio_chunk: np.ndarray, segments: list[dict]) -> list[dict]:
        """Assign speakers using pyannote diarization on the audio chunk."""
        if not segments:
            return segments

        self._load_pipeline()

        # pyannote expects a dict with "waveform" (torch tensor) and "sample_rate"
        waveform = torch.from_numpy(audio_chunk).unsqueeze(0)  # (1, samples)
        audio_input = {"waveform": waveform, "sample_rate": self.config.sample_rate}

        diarization = self._pipeline(audio_input)

        # Build a timeline: list of (start, end, pyannote_label)
        dia_segments = []
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            dia_segments.append((turn.start, turn.end, speaker_label))

        # Map each transcription segment to the best-matching diarization speaker
        for seg in segments:
            seg_mid = (seg["start"] + seg["end"]) / 2
            best_label = None
            best_overlap = 0.0

            for d_start, d_end, d_label in dia_segments:
                # Compute overlap between segment and diarization turn
                overlap_start = max(seg["start"], d_start)
                overlap_end = min(seg["end"], d_end)
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = d_label

            if best_label is None:
                # No diarization match — use midpoint proximity
                min_dist = float("inf")
                for d_start, d_end, d_label in dia_segments:
                    dist = min(abs(seg_mid - d_start), abs(seg_mid - d_end))
                    if dist < min_dist:
                        min_dist = dist
                        best_label = d_label

            # Map pyannote label to consistent "Speaker N"
            if best_label and best_label not in self._speaker_map:
                self._speaker_map[best_label] = f"Speaker {self._next_speaker_id}"
                self._next_speaker_id += 1

            seg["speaker"] = self._speaker_map.get(best_label, "Speaker 1")

        return segments


class Diarizer:
    """Speaker diarization facade.

    Automatically selects pyannote (if available + HF_TOKEN set) or
    falls back to energy-based detection.
    """

    def __init__(self, config: Config):
        self.config = config
        self._backend = None
        self._init_backend()

    def _init_backend(self):
        if self.config.hf_token and _ensure_pyannote_imports():
            print("🔊 Using pyannote-audio for speaker diarization")
            self._backend = PyannoteDiarizer(self.config)
        else:
            if not self.config.hf_token:
                reason = "HF_TOKEN not set"
            else:
                reason = "pyannote-audio not installed"
            print(f"🔊 Using energy-based speaker diarization ({reason})")
            self._backend = EnergyDiarizer(self.config)

    def reset(self):
        self._backend.reset()

    def assign_speakers(self, audio_chunk: np.ndarray, segments: list[dict]) -> list[dict]:
        return self._backend.assign_speakers(audio_chunk, segments)
