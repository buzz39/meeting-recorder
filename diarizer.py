"""
Speaker diarization with pyannote-audio (preferred) or energy-based fallback.

If pyannote-audio is installed and HF_TOKEN is available, uses the
pyannote/speaker-diarization-3.1 pipeline for accurate speaker identification.
Otherwise, falls back to lightweight energy-based speaker change detection.
"""

import numpy as np
from config import Config

# Try importing pyannote — graceful fallback if unavailable
_PYANNOTE_AVAILABLE = False
try:
    import torch
    from pyannote.audio import Pipeline as PyannotePipeline
    _PYANNOTE_AVAILABLE = True
except ImportError:
    pass


class EnergyDiarizer:
    """Energy-based speaker change detection (lightweight fallback).
    
    Detects speaker changes by analyzing silence gaps and energy patterns.
    Simple but effective for meetings where people take turns with natural pauses.
    """

    def __init__(self, config: Config):
        self.config = config
        self._current_speaker = 1
        self._speaker_profiles: dict[int, float] = {}
        self._max_speakers = 10

    def reset(self):
        self._current_speaker = 1
        self._speaker_profiles = {}

    def assign_speakers(self, audio_chunk: np.ndarray, segments: list[dict]) -> list[dict]:
        """Assign speaker labels based on energy analysis."""
        if not segments:
            return segments

        sample_rate = self.config.sample_rate
        prev_end = None

        for seg in segments:
            start_sample = max(0, min(int(seg["start"] * sample_rate), len(audio_chunk) - 1))
            end_sample = max(start_sample + 1, min(int(seg["end"] * sample_rate), len(audio_chunk)))
            segment_audio = audio_chunk[start_sample:end_sample]
            energy = self._compute_energy(segment_audio)

            if prev_end is not None:
                gap = seg["start"] - prev_end
                if gap >= self.config.speaker_change_silence:
                    best_speaker = self._match_speaker(energy)
                    if best_speaker != self._current_speaker:
                        self._current_speaker = best_speaker

            self._update_profile(self._current_speaker, energy)
            seg["speaker"] = f"Speaker {self._current_speaker}"
            prev_end = seg["end"]

        return segments

    def _compute_energy(self, audio: np.ndarray) -> float:
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))

    def _match_speaker(self, energy: float) -> int:
        if not self._speaker_profiles:
            self._speaker_profiles[1] = energy
            return 1

        best_id = self._current_speaker
        best_diff = float("inf")
        for spk_id, avg_energy in self._speaker_profiles.items():
            diff = abs(energy - avg_energy)
            if diff < best_diff:
                best_diff = diff
                best_id = spk_id

        threshold = self.config.min_speech_energy * 2
        if best_diff > threshold and len(self._speaker_profiles) < self._max_speakers:
            new_id = max(self._speaker_profiles.keys()) + 1
            self._speaker_profiles[new_id] = energy
            return new_id

        return best_id

    def _update_profile(self, speaker_id: int, energy: float):
        if speaker_id in self._speaker_profiles:
            self._speaker_profiles[speaker_id] = (
                0.7 * self._speaker_profiles[speaker_id] + 0.3 * energy
            )
        else:
            self._speaker_profiles[speaker_id] = energy


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
        if _PYANNOTE_AVAILABLE and self.config.hf_token:
            print("🔊 Using pyannote-audio for speaker diarization")
            self._backend = PyannoteDiarizer(self.config)
        else:
            if not _PYANNOTE_AVAILABLE:
                reason = "pyannote-audio not installed"
            else:
                reason = "HF_TOKEN not set"
            print(f"🔊 Using energy-based speaker diarization ({reason})")
            self._backend = EnergyDiarizer(self.config)

    def reset(self):
        self._backend.reset()

    def assign_speakers(self, audio_chunk: np.ndarray, segments: list[dict]) -> list[dict]:
        return self._backend.assign_speakers(audio_chunk, segments)
