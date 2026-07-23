import sys
import tempfile
import types
import wave

import numpy as np

sys.modules.setdefault("pyaudiowpatch", types.SimpleNamespace())

from audio_capture import AudioCapture
from config import Config


class FakeAudio:
    def __init__(self, devices, default_loopback=0, default_microphone=1):
        self.devices = devices
        self.default_loopback = default_loopback
        self.default_microphone = default_microphone

    def get_device_info_by_index(self, index):
        return self.devices[index]

    def get_device_count(self):
        return len(self.devices)

    def get_default_wasapi_loopback(self):
        return self.devices[self.default_loopback]

    def get_default_input_device_info(self):
        return self.devices[self.default_microphone]


def make_capture(config, devices):
    capture = AudioCapture.__new__(AudioCapture)
    capture.config = config
    capture.audio = FakeAudio(devices)
    return capture


def test_float_spool_is_converted_to_pcm_in_blocks(tmp_path):
    samples = np.array([-1.5, -1.0, -0.25, 0.0, 0.25, 1.0, 1.5], dtype=np.float32)
    source = tempfile.TemporaryFile()
    source.write(samples.tobytes())
    output = tmp_path / "audio.wav"

    with wave.open(str(output), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(16000)
        AudioCapture._write_float_file_as_pcm(source, destination)

    with wave.open(str(output), "rb") as result:
        pcm = np.frombuffer(result.readframes(result.getnframes()), dtype=np.int16)

    source.close()
    assert pcm.tolist() == [-32767, -32767, -8191, 0, 8191, 32767, 32767]


def test_saved_loopback_name_is_resolved_when_index_changes():
    devices = [
        {"index": 0, "name": "Default output", "maxInputChannels": 2, "isLoopbackDevice": True},
        {"index": 1, "name": "Microphone", "maxInputChannels": 1, "isLoopbackDevice": False},
        {"index": 2, "name": "Headset output", "maxInputChannels": 2, "isLoopbackDevice": True},
    ]
    capture = make_capture(Config(device_index=1, device_name="Headset output"), devices)

    assert capture.find_loopback_device()["index"] == 2


def test_saved_microphone_name_is_resolved_when_index_changes():
    devices = [
        {"index": 0, "name": "Default output", "maxInputChannels": 2, "isLoopbackDevice": True},
        {"index": 1, "name": "Default microphone", "maxInputChannels": 1, "isLoopbackDevice": False},
        {"index": 2, "name": "USB microphone", "maxInputChannels": 1, "isLoopbackDevice": False},
    ]
    config = Config(microphone_device_index=0, microphone_device_name="USB microphone")
    capture = make_capture(config, devices)

    assert capture.find_microphone_device()["index"] == 2
