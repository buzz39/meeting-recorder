"""
WASAPI Loopback Audio Capture for Windows.

Captures system audio (whatever is playing through speakers/headphones)
using Windows Audio Session API (WASAPI) loopback mode.

NOTE: This module requires Windows 10/11 and pyaudiowpatch.
"""

import queue
import threading
import wave

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

# scipy is available as a transitive dependency of faster-whisper; use its
# polyphase resampler when present for anti-aliased downsampling.
try:
    from math import gcd as _gcd

    from scipy.signal import resample_poly as _resample_poly
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def _resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono float32 audio from src_rate to dst_rate.

    Uses scipy's polyphase resampler (with built-in anti-aliasing low-pass
    filter) when available. Falls back to linear interpolation otherwise —
    still aliased, but considerably better than nearest-neighbor decimation.
    """
    if src_rate == dst_rate or len(audio) == 0:
        return audio
    if _SCIPY_AVAILABLE:
        g = _gcd(src_rate, dst_rate)
        up = dst_rate // g
        down = src_rate // g
        return _resample_poly(audio, up, down).astype(np.float32)
    # Fallback: linear interpolation.
    n_out = int(round(len(audio) * dst_rate / src_rate))
    if n_out <= 0:
        return audio[:0]
    src_idx = np.linspace(0, len(audio) - 1, num=n_out, dtype=np.float64)
    return np.interp(src_idx, np.arange(len(audio)), audio).astype(np.float32)


class AudioCapture:
    """Captures system audio via WASAPI loopback and provides chunks for processing."""

    def __init__(self, config: Config):
        self.config = config
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.mic_stream = None
        self.is_recording = False
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.mic_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._raw_frames: list[bytes] = []  # For saving full WAV
        self._processed_chunks: list[np.ndarray] = []  # Mixed mono 16kHz chunks when mic is enabled
        self._lock = threading.Lock()
        self._device_info = None
        self._mic_device_info = None

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

    def _mic_callback(self, in_data, frame_count, time_info, status):
        """Called by PyAudio for microphone buffers when mic mixing is enabled."""
        with self._lock:
            if self.is_recording:
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                if self._mic_device_info and self._mic_device_info["maxInputChannels"] > 1:
                    channels = self._mic_device_info["maxInputChannels"]
                    audio_data = audio_data.reshape(-1, channels).mean(axis=1)
                if self.config.microphone_gain != 1.0:
                    audio_data = audio_data * float(self.config.microphone_gain)
                self.mic_queue.put(audio_data)
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
        self._processed_chunks = []

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

        if self.config.include_microphone:
            self._mic_device_info = (
                self.audio.get_device_info_by_index(self.config.microphone_device_index)
                if self.config.microphone_device_index is not None
                else self.audio.get_default_input_device_info()
            )
            mic_channels = int(self._mic_device_info["maxInputChannels"])
            mic_rate = int(self._mic_device_info["defaultSampleRate"])
            if mic_channels <= 0:
                raise RuntimeError("Selected microphone device has no input channels")
            print(f"🎙️  Mixing microphone: {self._mic_device_info['name']}")
            print(f"   Channels: {mic_channels}, Rate: {mic_rate}Hz, Gain: {self.config.microphone_gain:g}x")
            self.mic_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=mic_channels,
                rate=mic_rate,
                input=True,
                input_device_index=self._mic_device_info["index"],
                frames_per_buffer=1024,
                stream_callback=self._mic_callback,
            )
            self.mic_stream.start_stream()

    def stop(self) -> list[bytes]:
        """Stop capturing and return raw frames."""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.mic_stream:
            self.mic_stream.stop_stream()
            self.mic_stream.close()
            self.mic_stream = None
        with self._lock:
            frames = list(self._raw_frames)
            self._raw_frames = []
        return frames

    def save_wav(self, filepath: str, frames: list[bytes]):
        """Save captured audio frames to a WAV file.

        Frames are float32 samples from the device. We convert to int16 PCM
        before writing so the file is universally playable — the stdlib `wave`
        module always writes a PCM (WAVE_FORMAT_PCM) header, so writing raw
        float32 bytes with `setsampwidth(4)` produces a file that most players
        misinterpret as int32 and render as static.
        """
        if self.config.include_microphone and self._processed_chunks:
            audio_f32 = np.concatenate(self._processed_chunks)
            audio_i16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(audio_i16.tobytes())
            return

        if not frames or not self._device_info:
            return
        channels = self._device_info["maxInputChannels"]
        rate = int(self._device_info["defaultSampleRate"])

        # Concatenate all float32 frames, clip, and convert to int16 PCM.
        audio_f32 = np.frombuffer(b"".join(frames), dtype=np.float32)
        audio_clipped = np.clip(audio_f32, -1.0, 1.0)
        audio_i16 = (audio_clipped * 32767.0).astype(np.int16)

        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(rate)
            wf.writeframes(audio_i16.tobytes())

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

        # Resample from device rate to the configured transcription sample rate if needed.
        # Uses scipy.signal.resample_poly (polyphase + anti-aliasing) when
        # available, otherwise linear interpolation. The previous nearest-
        # neighbor approach aliased badly when downsampling 48kHz → 16kHz.
        if device_rate != self.config.sample_rate:
            audio = _resample_audio(audio, device_rate, self.config.sample_rate)

        if self.config.include_microphone and self._mic_device_info:
            mic_rate = int(self._mic_device_info["defaultSampleRate"])
            mic_chunks = []
            while True:
                try:
                    mic_chunks.append(self.mic_queue.get_nowait())
                except queue.Empty:
                    break
            if mic_chunks:
                mic_audio = np.concatenate(mic_chunks)
                if mic_rate != self.config.sample_rate:
                    mic_audio = _resample_audio(mic_audio, mic_rate, self.config.sample_rate)
                if len(mic_audio) < len(audio):
                    mic_audio = np.pad(mic_audio, (0, len(audio) - len(mic_audio)))
                else:
                    mic_audio = mic_audio[:len(audio)]
                # Simple additive mix keeps dependencies low. Clip to the valid
                # float32 audio range; very loud simultaneous sources may distort,
                # so users can lower --mic-gain if needed.
                audio = np.clip(audio + mic_audio, -1.0, 1.0)
            self._processed_chunks.append(audio.copy())

        return audio.astype(np.float32)

    def cleanup(self):
        """Release PyAudio resources."""
        self.audio.terminate()
