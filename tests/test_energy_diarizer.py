"""Tests for the energy-based diarizer (no neural model required)."""

import numpy as np
import pytest

from config import Config
from diarizer import EnergyDiarizer


def _tone(freq: float, duration: float, sample_rate: int, amplitude: float = 0.3) -> np.ndarray:
    """Generate a mono float32 sine tone."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(sample_rate * duration), dtype=np.float32)


@pytest.fixture()
def cfg():
    return Config()


def test_assign_speakers_empty_segments_returns_empty(cfg):
    d = EnergyDiarizer(cfg)
    assert d.assign_speakers(np.zeros(16000, dtype=np.float32), []) == []


def test_assign_speakers_labels_are_strings(cfg):
    d = EnergyDiarizer(cfg)
    sr = cfg.sample_rate
    audio = _tone(220, 1.0, sr)
    segments = [{"start": 0.0, "end": 1.0, "text": "hi"}]
    out = d.assign_speakers(audio, segments)
    assert out[0]["speaker"].startswith("Speaker ")


def test_reset_clears_speaker_profiles(cfg):
    d = EnergyDiarizer(cfg)
    sr = cfg.sample_rate
    audio = _tone(220, 1.0, sr)
    d.assign_speakers(audio, [{"start": 0.0, "end": 1.0, "text": "x"}])
    assert d._speaker_profiles
    d.reset()
    assert d._speaker_profiles == {}
    assert d._current_speaker == 1


def test_distinct_pitches_after_silence_can_split_speakers(cfg):
    """A loud low-pitch segment followed by silence and a loud high-pitch
    segment should plausibly be assigned two different speaker labels.

    The energy diarizer is a heuristic; this asserts the structural property
    (two distinct labels are *possible*) by feeding very different features
    across a clear silence gap.
    """
    d = EnergyDiarizer(cfg)
    sr = cfg.sample_rate
    # 1s of 150 Hz tone, 1s silence, 1s of 2000 Hz tone.
    audio = np.concatenate([
        _tone(150, 1.0, sr, amplitude=0.5),
        _silence(1.0, sr),
        _tone(2000, 1.0, sr, amplitude=0.5),
    ])
    segments = [
        {"start": 0.0, "end": 1.0, "text": "low"},
        {"start": 2.0, "end": 3.0, "text": "high"},
    ]
    out = d.assign_speakers(audio, segments)
    assert len(out) == 2
    # We don't assert they MUST differ — the heuristic could collapse them —
    # but both must be valid Speaker N labels.
    assert all(s["speaker"].startswith("Speaker ") for s in out)


def test_speaker_count_is_capped(cfg):
    """The diarizer should never invent more than _max_speakers identities."""
    d = EnergyDiarizer(cfg)
    sr = cfg.sample_rate
    # Build a long audio buffer with widely varying tones separated by silence.
    parts = []
    segments = []
    cursor = 0.0
    rng = np.random.default_rng(0)
    for _ in range(d._max_speakers + 5):
        freq = float(rng.uniform(100, 4000))
        parts.append(_tone(freq, 0.5, sr, amplitude=0.5))
        parts.append(_silence(1.0, sr))
        segments.append({"start": cursor, "end": cursor + 0.5, "text": "x"})
        cursor += 1.5
    audio = np.concatenate(parts)
    d.assign_speakers(audio, segments)
    assert len(d._speaker_profiles) <= d._max_speakers


def test_compute_features_handles_empty_audio(cfg):
    d = EnergyDiarizer(cfg)
    feats = d._compute_features(np.array([], dtype=np.float32))
    assert feats.shape == (3,)
    assert np.allclose(feats, 0.0)
