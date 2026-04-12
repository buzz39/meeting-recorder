"""
Speaker diarization using energy-based speaker change detection.

This is a lightweight alternative to pyannote-audio that doesn't require
a HuggingFace token or heavy model downloads. It detects speaker changes
by analyzing silence gaps and energy patterns in the audio.

For better accuracy, you can swap this with pyannote-audio (see README).
"""

import numpy as np
from config import Config


class Diarizer:
    """Energy-based speaker change detection.
    
    Strategy: When there's a silence gap longer than `speaker_change_silence`,
    we check if the audio characteristics (energy profile) changed enough
    to suggest a different speaker. Simple but effective for meetings where
    people take turns speaking with natural pauses.
    """

    def __init__(self, config: Config):
        self.config = config
        self._current_speaker = 1
        self._speaker_profiles: dict[int, float] = {}  # speaker_id -> avg energy
        self._max_speakers = 10

    def reset(self):
        """Reset speaker tracking for a new recording."""
        self._current_speaker = 1
        self._speaker_profiles = {}

    def assign_speakers(self, audio_chunk: np.ndarray, segments: list[dict]) -> list[dict]:
        """Assign speaker labels to transcription segments.
        
        Analyzes the audio energy for each segment and detects speaker changes
        based on silence gaps and energy profile differences.
        
        Args:
            audio_chunk: Full audio chunk (float32, 16kHz) that segments came from
            segments: List of {"start": float, "end": float, "text": str}
            
        Returns:
            Same segments with added "speaker" field: "Speaker 1", "Speaker 2", etc.
        """
        if not segments:
            return segments

        sample_rate = self.config.sample_rate
        prev_end = None

        for seg in segments:
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            
            # Clamp to audio bounds
            start_sample = max(0, min(start_sample, len(audio_chunk) - 1))
            end_sample = max(start_sample + 1, min(end_sample, len(audio_chunk)))
            
            segment_audio = audio_chunk[start_sample:end_sample]
            energy = self._compute_energy(segment_audio)

            # Check for speaker change based on silence gap
            if prev_end is not None:
                gap = seg["start"] - prev_end
                if gap >= self.config.speaker_change_silence:
                    # Silence gap detected — check if energy profile suggests new speaker
                    best_speaker = self._match_speaker(energy)
                    if best_speaker != self._current_speaker:
                        self._current_speaker = best_speaker

            # Update speaker profile
            self._update_profile(self._current_speaker, energy)
            seg["speaker"] = f"Speaker {self._current_speaker}"
            prev_end = seg["end"]

        return segments

    def _compute_energy(self, audio: np.ndarray) -> float:
        """Compute RMS energy of an audio segment."""
        if len(audio) == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio ** 2)))

    def _match_speaker(self, energy: float) -> int:
        """Find the best matching speaker for given energy, or create new one."""
        if not self._speaker_profiles:
            self._speaker_profiles[1] = energy
            return 1

        # Find closest energy profile match
        best_id = self._current_speaker
        best_diff = float("inf")
        
        for spk_id, avg_energy in self._speaker_profiles.items():
            diff = abs(energy - avg_energy)
            if diff < best_diff:
                best_diff = diff
                best_id = spk_id

        # If energy is very different from all known speakers, create new one
        threshold = self.config.min_speech_energy * 2
        if best_diff > threshold and len(self._speaker_profiles) < self._max_speakers:
            new_id = max(self._speaker_profiles.keys()) + 1
            self._speaker_profiles[new_id] = energy
            return new_id

        # If it matches current speaker closely, don't change
        if best_id == self._current_speaker:
            return self._current_speaker
            
        return best_id

    def _update_profile(self, speaker_id: int, energy: float):
        """Update running average energy for a speaker."""
        if speaker_id in self._speaker_profiles:
            # Exponential moving average
            self._speaker_profiles[speaker_id] = (
                0.7 * self._speaker_profiles[speaker_id] + 0.3 * energy
            )
        else:
            self._speaker_profiles[speaker_id] = energy
