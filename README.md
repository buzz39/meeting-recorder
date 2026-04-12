# Meeting Recorder 🎙️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A lightweight Windows CLI tool that captures system audio (WASAPI loopback), transcribes it locally using faster-whisper, and identifies different speakers — all running offline after initial setup.

## Prerequisites

- **Windows 10/11** (WASAPI loopback is Windows-only)
- **Python 3.10+**
- Audio output device (speakers or headphones must be active)

## Installation

```bash
# Clone or copy the project
cd meeting-recorder

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

The Whisper model downloads automatically on first run (~150MB for `base`).

## Usage

### Record a meeting
```bash
# Start recording with defaults (base model, txt output)
python recorder.py start

# Use a larger model for better accuracy
python recorder.py start --model small

# Save as SRT subtitles
python recorder.py start --format srt

# Custom output directory
python recorder.py start --output C:\Users\you\meetings

# Specify language (skip auto-detection)
python recorder.py start --language en

# All options
python recorder.py start --model small --format srt --output ./my_meetings --language en --chunk 20
```

Press **Ctrl+C** to stop — recording and transcript are saved automatically.

### Transcribe an existing file
```bash
python recorder.py transcribe path\to\meeting.wav
python recorder.py transcribe meeting.wav --model small --format srt
```

### List past recordings
```bash
python recorder.py list
```

### List audio devices
```bash
python recorder.py devices
```

Use `--device <index>` with `start` to pick a specific loopback device.

## Output

Each recording creates two files in the output directory:
- `meeting_YYYYMMDD_HHMMSS.wav` — full audio
- `meeting_YYYYMMDD_HHMMSS.txt` — transcript with timestamps and speaker labels

### Sample transcript output
```
[00:00:02.340] Speaker 1: Welcome everyone to the standup.
[00:00:05.120] Speaker 1: Let's start with updates from the backend team.
[00:00:08.900] Speaker 2: Sure, we shipped the API changes yesterday.
[00:00:15.400] Speaker 3: On the frontend side, we're still blocked on the design review.
```

## Model Sizes vs Speed

| Model    | Size   | Speed  | Accuracy | RAM   |
|----------|--------|--------|----------|-------|
| `tiny`   | 75 MB  | ⚡⚡⚡⚡ | ★★☆☆☆   | ~1 GB |
| `base`   | 145 MB | ⚡⚡⚡  | ★★★☆☆   | ~1 GB |
| `small`  | 488 MB | ⚡⚡   | ★★★★☆   | ~2 GB |
| `medium` | 1.5 GB | ⚡     | ★★★★★   | ~5 GB |

**Recommendation:** Start with `base` for real-time. Use `small` if you have 8GB+ RAM and don't mind slight delay.

## Troubleshooting

### "No WASAPI loopback device found"
- Make sure you have an audio output device (speakers/headphones) active
- Run `python recorder.py devices` to see available devices
- Try selecting a specific device: `--device <index>`

### No audio captured / empty transcript
- Ensure something is actually playing through your speakers during recording
- Check that your meeting app audio is going to the default output device
- Virtual audio cables may need separate configuration

### Transcription is slow
- Use a smaller model: `--model tiny`
- Reduce chunk size: `--chunk 15`
- Close other CPU-heavy applications

### Speaker detection is inaccurate
The built-in diarization uses energy-based detection (lightweight but approximate). For better accuracy, consider integrating [pyannote-audio](https://github.com/pyannote/pyannote-audio):

```bash
pip install pyannote-audio
```
Then modify `diarizer.py` to use pyannote's `Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")`. Note: requires a HuggingFace token.

## How It Works

1. **Audio Capture** — Uses `pyaudiowpatch` to open a WASAPI loopback stream that mirrors your default audio output
2. **Chunked Processing** — Audio is buffered in 30-second chunks (configurable) and sent to faster-whisper
3. **Transcription** — faster-whisper (CTranslate2) runs locally on CPU, converting speech to text with timestamps
4. **Speaker Detection** — Energy-based analysis detects silence gaps and audio profile changes to assign speaker labels
5. **Output** — Results stream to console in real-time and are saved to file on stop

## Contributing

Contributions are welcome! Feel free to:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/awesome`)
3. Commit your changes (`git commit -m 'Add awesome feature'`)
4. Push to the branch (`git push origin feature/awesome`)
5. Open a Pull Request

### Ideas for contributions
- Linux/macOS audio capture support (PulseAudio/CoreAudio)
- pyannote-audio integration for better speaker diarization
- Real-time subtitle overlay
- Meeting summary generation with LLMs

## License

MIT — see [LICENSE](LICENSE) for details.
