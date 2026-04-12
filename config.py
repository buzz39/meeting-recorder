"""Configuration defaults for Meeting Recorder."""

import os
from dataclasses import dataclass, field

@dataclass
class Config:
    # Whisper model size: tiny, base, small, medium, large-v2
    model_size: str = "base"
    
    # Output directory for recordings
    output_dir: str = os.path.join(os.path.dirname(__file__), "recordings")
    
    # Output format: txt or srt
    output_format: str = "txt"
    
    # Audio chunk duration in seconds for processing
    chunk_duration: float = 30.0
    
    # Sample rate (16kHz is what Whisper expects)
    sample_rate: int = 16000
    
    # Number of audio channels (mono for transcription)
    channels: int = 1
    
    # Whisper language (None = auto-detect)
    language: str | None = None
    
    # Minimum silence duration (seconds) to detect speaker change
    speaker_change_silence: float = 1.5
    
    # Energy threshold (RMS) below which audio is considered silence
    silence_threshold: float = 0.01
    
    # Minimum segment energy to consider speech
    min_speech_energy: float = 0.02
    
    # Device index for audio capture (None = default loopback)
    device_index: int | None = None
    
    # Compute type for faster-whisper: int8, float16, float32
    compute_type: str = "int8"
