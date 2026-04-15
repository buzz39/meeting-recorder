"""
WASAPI Loopback Audio Capture for Windows.

Captures system audio (whatever is playing through speakers/headphones)
using Windows Audio Session API (WASAPI) loopback mode.

NOTE: This module requires Windows 10/11 and pyaudiowpatch.
"""

import threading
import queue
import wave
import time
import numpy as np

# pyaudiowpatch is a fork of PyAudio with WASAPI loopback support (Windows only)
try:
    import pyaudiowpatch as pyaudio
except ImportError:
    raise ImportError(
        "pyaudiowpatch is required for WASAPI loopback capture.\n"
        "Install it: pip install pyaudiowpatch"
    )

from config import Config


class AudioCapture:
    """Captures system audio via WASAPI loopback and provides chunks for processing."""

    def __init__(self, config: Config):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._raw_frames: list[bytes] = []  # For saving full WAV
        self._lock = threading.Lock()
        self._device_info = None

    def find_loopback_device(self) -> dict:
        """Find the default WASAPI loopback device.
        
        WASAPI loopback devices mirror the output device, capturing
        whatever audio is being played through speakers/headphones.
        """
        if self.config.device_index is not None:
            return self.audio.get_device_info_by_index(self.config.device_index)

        # Get default speakers
        default_speakers = self.audio.get_default_wasapi_loopback()
        if default_speakers:
            return default_speakers

        # Fallback: search for any loopback device
        for i in range(self.audio.get_device_count()):
            dev = self.audio.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice", False):
                return dev

        raise RuntimeError(
            "No WASAPI loopback device found. Make sure you're on Windows 10/11 "
            "and have audio output devices available."
        )

    def list_devices(self) -> list[dict]:
        """List all available audio devices."""
        devices = []
        for i in range(self.audio.get_device_count()):
            dev = self.audio.get_device_info_by_index(i)
            devices.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["maxInputChannels"],
                "rate": int(dev["defaultSampleRate"]),
                "loopback": dev.get("isLoopbackDevice", False),
            })
        return devices

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Called by PyAudio for each audio buffer."""
        with self._lock:
            if self.is_recording:
                self._raw_frames.append(in_data)
                # Convert to numpy, resample to 16kHz mono float32
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                # If stereo (or more), mix down to mono
                if self._device_info and self._device_info["maxInputChannels"] > 1:
                    channels = self._device_info["maxInputChannels"]
                    audio_data = audio_data.reshape(-1, channels).mean(axis=1)
                self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def start(self):
        """Start capturing system audio."""
        self._device_info = self.find_loopback_device()
        device_channels = self._device_info["maxInputChannels"]
        device_rate = int(self._device_info["defaultSampleRate"])

        print(f"🎤 Capturing from: {self._device_info['name']}")
        print(f"   Channels: {device_channels}, Rate: {device_rate}Hz")

        self.is_recording = True
        self._raw_frames = []

        # Open stream using the loopback device's native format
        # We capture at device rate and convert later for Whisper
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=device_channels,
            rate=device_rate,
            input=True,
            input_device_index=self._device_info["index"],
            frames_per_buffer=1024,
            stream_callback=self._audio_callback,
        )
        self.stream.start_stream()

    def stop(self) -> list[bytes]:
        """Stop capturing and return raw frames."""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        with self._lock:
            frames = list(self._raw_frames)
            self._raw_frames = []
        return frames

    def save_wav(self, filepath: str, frames: list[bytes]):
        """Save captured audio frames to a WAV file."""
        if not frames or not self._device_info:
            return
        channels = self._device_info["maxInputChannels"]
        rate = int(self._device_info["defaultSampleRate"])
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(4)  # float32 = 4 bytes
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))

    def get_chunk(self, duration: float, stop_event: threading.Event | None = None) -> np.ndarray | None:
        """Collect audio samples for `duration` seconds, return as float32 array.
        
        Args:
            duration: Number of seconds of audio to collect.
            stop_event: Optional threading.Event; when set, collection stops
                        immediately so callers are not blocked for the full duration.

        Returns None if recording stopped before enough data collected.
        """
        if not self._device_info:
            return None
        device_rate = int(self._device_info["defaultSampleRate"])
        target_samples = int(duration * device_rate)
        # Account for stereo→mono conversion
        if self._device_info["maxInputChannels"] > 1:
            # After mono mixdown, each callback gives frame_count samples
            pass  # audio_callback already mixes to mono

        collected = []
        total = 0
        while total < target_samples and self.is_recording:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                collected.append(chunk)
                total += len(chunk)
            except queue.Empty:
                if not self.is_recording:
                    break

        if not collected:
            return None

        audio = np.concatenate(collected)

        # Resample from device rate to 16kHz for Whisper if needed
        if device_rate != 16000:
            # Simple resampling via linear interpolation
            ratio = 16000 / device_rate
            indices = np.arange(0, len(audio), 1 / ratio).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]

        return audio.astype(np.float32)

    def cleanup(self):
        """Release PyAudio resources."""
        self.audio.terminate()
