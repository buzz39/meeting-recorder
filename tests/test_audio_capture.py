import sys
import tempfile
import types
import wave

import numpy as np

sys.modules.setdefault("pyaudiowpatch", types.SimpleNamespace())

from audio_capture import AudioCapture


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
