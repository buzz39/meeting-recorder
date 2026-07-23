# Meeting Recorder 🎙️

[![CI](https://github.com/buzz39/meeting-recorder/actions/workflows/ci.yml/badge.svg)](https://github.com/buzz39/meeting-recorder/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-windows-lightgrey.svg)](#prerequisites)

A lightweight Windows CLI tool that captures system audio (WASAPI loopback), transcribes it locally using faster-whisper, and identifies different speakers — all running offline after initial setup.

> [!IMPORTANT]
> Only record people with their knowledge and consent. Recording laws vary by
> location and meeting context; you are responsible for complying with them.

## Features

- **Local or cloud transcription** — faster-whisper runs offline, or use OpenAI's transcription API on low-RAM/CPU systems
- **Speaker diarization** — pyannote-audio (accurate, neural) or energy-based (lightweight) fallback
- **Speaker-count controls** — pass a known speaker count (for example `--speakers 2`) to avoid over-splitting voices
- **Microphone mix by default** — capture your mic alongside WASAPI loopback so your own voice is clear in recordings
- **Multiple output formats** — TXT timestamps, SRT subtitles, JSON, or all three
- **System tray mode** — minimize to tray, start/stop recording from right-click menu (Windows)
- **Real-time streaming** — see transcription as it happens during recording

## Why this vs. alternatives?

| Tool | Loopback capture | Local transcription | Diarization | Tray mode | Output formats |
|------|------------------|---------------------|-------------|-----------|----------------|
| **Meeting Recorder** | ✅ WASAPI (Windows) | ✅ faster-whisper | ✅ pyannote / energy | ✅ Windows | TXT, SRT, JSON |
| `whisper.cpp` examples | ❌ | ✅ | ❌ | ❌ | TXT, SRT, VTT |
| `chidiwilliams/buzz` | ❌ (mic only) | ✅ | ❌ | ❌ (GUI app) | TXT, SRT, VTT |
| `WhisperX` | ❌ | ✅ | ✅ | ❌ | JSON, SRT |
| Cloud (Otter, Fireflies, …) | ✅ | ❌ (cloud) | ✅ | ✅ | Many |

This project's niche is the combination of **system-audio loopback** (capture the other side of any meeting app without configuring a virtual cable), **fully local** transcription + diarization, and an **always-on tray icon** so you can hit "record" the moment a meeting starts.

## Prerequisites

- **Windows 10/11** (WASAPI loopback is Windows-only)
- **Python 3.10+**
- Audio output device (speakers or headphones must be active)

## Installation

### Windows desktop installer

For normal desktop use, install a signed `MeetingRecorder-*-Setup.exe` release.
The installer creates a Start menu shortcut and can launch Meeting Recorder
automatically when you sign in. On first launch, the setup window:

1. explains recording consent, local storage, microphone capture, and cloud uploads;
2. lets you choose audio devices, recordings folder, model, language, speaker
   count, transcript format, and local or cloud transcription; and
3. saves non-secret preferences under `%APPDATA%\Meeting Recorder`.

The application then lives in the Windows notification area. Right-click its
icon to start or stop recording, open the recordings folder, or change Settings.
Recordings default to `Documents\Meeting Recorder`, so application upgrades do
not remove them. API keys and HuggingFace tokens are never stored in the settings
file; continue to provide them through environment variables.

The desktop installer contains the lightweight energy-based diarizer. Local
Whisper models download automatically on the first recording. Developers who
need pyannote's more accurate diarization should use the full Python installation
below and set `HF_TOKEN`.

### Full install (with pyannote speaker diarization)

```bash
cd meeting-recorder
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ `torch` is ~2GB. Full install is recommended if you have the disk space.

### Lite install (energy-based diarization, smaller download)

```bash
pip install -r requirements-lite.txt
```

The Whisper model downloads automatically on first run (~488MB for `small`).

### pyannote setup (for full install)

pyannote-audio requires a HuggingFace token with access to the model:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Accept the terms for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
5. Set the token:

```bash
set HF_TOKEN=hf_your_token_here       # Windows CMD
$env:HF_TOKEN = "hf_your_token_here"  # PowerShell
export HF_TOKEN=hf_your_token_here     # Linux/macOS
```

If `HF_TOKEN` is not set or pyannote is not installed, the recorder automatically falls back to energy-based diarization.

## Usage

### Quick start with speaker diarization

```bash
# Step 1: Set your HuggingFace token (required for pyannote speaker detection)
set HF_TOKEN=hf_your_token_here       # Windows CMD
$env:HF_TOKEN = "hf_your_token_here"  # PowerShell

# Step 2: Start recording — that's it!
python recorder.py start
```

> 💡 **Tip:** Add `HF_TOKEN` to your Windows environment variables permanently so you don't have to set it every time:
> Settings → System → About → Advanced system settings → Environment Variables → New → Name: `HF_TOKEN`, Value: `hf_your_token_here`

### Record a meeting

```bash
# Start recording (small model, txt output)
python recorder.py start

# Use a different model
python recorder.py start --model base

# Save as SRT subtitles
python recorder.py start --format srt

# Save as JSON (machine-readable, ideal for piping into other tools)
python recorder.py start --format json

# Save TXT, SRT, and JSON
python recorder.py start --format all

# Custom output directory + language
python recorder.py start --output C:\Users\you\meetings --language en

# Two-person call: prevent extra Speaker 3/4 labels
python recorder.py start --speaker-count 2

# Your microphone is mixed in by default; adjust gain if needed
python recorder.py start --mic-gain 1.5

# Use cloud transcription to avoid local Whisper downloads/cold start
set OPENAI_API_KEY=sk_your_key_here
python recorder.py start --provider openai --transcription-model whisper-1

# Use Vercel AI Gateway / compatible APIs by changing the provider and model
set AI_GATEWAY_API_KEY=your_gateway_key_here
python recorder.py start --provider vercel --transcription-model openai/whisper-1

# Record loopback audio only
python recorder.py start --no-include-mic

# All common options
python recorder.py start --model small --format all --output ./my_meetings --language en --chunk 20 --speaker-count 2 --mic-gain 1.5
```

Press **Ctrl+C** to stop — recording and transcript are saved automatically.

### System tray mode (Windows)

**Easiest way — double-click:**
- `start_tray.bat` — launches tray with a console window (shows logs)
- `start_tray.vbs` — launches tray **silently** (no console window, fully invisible)

> 💡 **Pro tip:** Create a shortcut to `start_tray.vbs` and put it in your Startup folder (`Win+R` → `shell:startup`) to auto-launch on boot!

**From CLI:**
```bash
python recorder.py tray
```

This minimizes to the system tray with:
- 🔴 Red circle icon when recording
- ⚫ Gray circle icon when idle
- Right-click menu: Start Recording, Stop Recording, Open Recordings Folder, Quit
- Native first-run setup and a Settings window
- Tooltip shows elapsed recording time

### Transcribe an existing file

```bash
python recorder.py transcribe path\to\meeting.wav
python recorder.py transcribe meeting.wav --model small --format srt
python recorder.py transcribe meeting.wav --format all
python recorder.py transcribe meeting.wav --provider openai
python recorder.py transcribe meeting.wav --provider vercel --transcription-model openai/whisper-1
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
Use `--mic-device <index>` to pick a specific microphone.

### Improving speaker labels

If you know the meeting has two speakers, pass:

```bash
python recorder.py start --speaker-count 2
```

This forwards the exact count to pyannote when available and caps the lightweight
energy diarizer at two labels, preventing accidental `Speaker 3`, `Speaker 4`,
etc. For larger meetings where the exact count is unknown, use `--max-speakers N`.

### Making your own voice clear

WASAPI loopback records the audio playing through speakers/headphones. Most
meeting apps do **not** play your own microphone back to you, so loopback alone
captures the other participants clearly but may miss or weaken your voice.

Microphone mixing is enabled by default:

```bash
python recorder.py devices
python recorder.py start --mic-gain 1.5
python recorder.py start --mic-device 3
```

The saved WAV is a mono 16 kHz mix of the loopback audio and your microphone.
If you only want system audio, pass `--no-include-mic`.

### Cloud transcription for low-resource systems

Cloud transcription avoids local Whisper model downloads, high RAM/CPU use, and
local model cold start. It uploads the selected audio to the configured provider,
where that provider's privacy, retention, and billing terms apply. Set your API
key and select a cloud provider:

```bash
set OPENAI_API_KEY=sk_your_key_here       # Windows CMD
$env:OPENAI_API_KEY = "sk_your_key_here"  # PowerShell
python recorder.py start --provider openai --transcription-model whisper-1
```

For Vercel AI Gateway or other OpenAI-compatible transcription endpoints:

```bash
set AI_GATEWAY_API_KEY=your_gateway_key_here
python recorder.py start --provider vercel --transcription-model openai/whisper-1

set TRANSCRIPTION_API_KEY=your_key_here
python recorder.py start --provider compatible --transcription-base-url https://example.com/v1 --transcription-model provider/model
```

You can also set `TRANSCRIPTION_PROVIDER`, `TRANSCRIPTION_MODEL`,
`TRANSCRIPTION_API_KEY`, and `TRANSCRIPTION_BASE_URL` as environment variables.
The older `OPENAI_API_KEY` and `OPENAI_TRANSCRIBE_MODEL` variables still work for
OpenAI-compatible setups.

## Output

Each recording creates files in the output directory:
- `meeting_YYYYMMDD_HHMMSS.wav` — full audio (16-bit PCM)
- `meeting_YYYYMMDD_HHMMSS.txt` — transcript with timestamps and speaker labels
- `meeting_YYYYMMDD_HHMMSS.srt` — SRT subtitles (if `--format srt` or `--format all`)
- `meeting_YYYYMMDD_HHMMSS.json` — structured transcript (if `--format json` or `--format all`)
- `meeting_YYYYMMDD_HHMMSS.html` — self-contained viewer: open in any browser to play the audio and click any line in the transcript to seek to that moment. No server, no extra dependencies.

### Sample TXT output

```
[00:00:02.340] Speaker 1: Welcome everyone to the standup.
[00:00:05.120] Speaker 1: Let's start with updates from the backend team.
[00:00:08.900] Speaker 2: Sure, we shipped the API changes yesterday.
```

### Sample SRT output

```
1
00:00:02,340 --> 00:00:05,120
[Speaker 1] Welcome everyone to the standup.

2
00:00:05,120 --> 00:00:08,900
[Speaker 1] Let's start with updates from the backend team.

3
00:00:08,900 --> 00:00:15,400
[Speaker 2] Sure, we shipped the API changes yesterday.
```

### Sample JSON output

```json
{
  "version": 1,
  "segments": [
    {"start": 2.34, "end": 5.12,  "speaker": "Speaker 1", "text": "Welcome everyone to the standup."},
    {"start": 5.12, "end": 8.90,  "speaker": "Speaker 1", "text": "Let's start with updates from the backend team."},
    {"start": 8.90, "end": 15.40, "speaker": "Speaker 2", "text": "Sure, we shipped the API changes yesterday."}
  ]
}
```

## Model Sizes vs Speed

| Model    | Size   | Speed  | Accuracy | RAM   |
|----------|--------|--------|----------|-------|
| `tiny`   | 75 MB  | ⚡⚡⚡⚡ | ★★☆☆☆   | ~1 GB |
| `base`   | 145 MB | ⚡⚡⚡  | ★★★☆☆   | ~1 GB |
| `small`  | 488 MB | ⚡⚡   | ★★★★☆   | ~2 GB |
| `medium` | 1.5 GB | ⚡     | ★★★★★   | ~5 GB |

**Default:** `small` — good balance of speed and accuracy for 8GB+ RAM machines.

## Troubleshooting

### "No WASAPI loopback device found"
- Make sure you have an audio output device (speakers/headphones) active
- Run `python recorder.py devices` to see available devices
- Try selecting a specific device: `--device <index>`

### No audio captured / empty transcript
- Ensure something is actually playing through your speakers during recording
- Check that your meeting app audio is going to the default output device

### Transcription is slow
- Use a smaller model: `--model tiny` or `--model base`
- Reduce chunk size: `--chunk 15`

### Speaker detection is inaccurate
- **Best:** Install pyannote-audio (full install) and set `HF_TOKEN`
- **Fallback:** Energy-based detection works best when speakers have distinct volumes and take clear turns

## How It Works

1. **Audio Capture** — WASAPI loopback stream mirrors your default audio output
2. **Chunked Processing** — Audio buffered in 30s chunks, sent to faster-whisper
3. **Transcription** — faster-whisper (CTranslate2) runs locally on CPU
4. **Speaker Diarization** — pyannote neural pipeline (or energy-based fallback)
5. **Output** — Real-time console output + saved files on stop

## Known limitations

- **Windows only for system audio** — WASAPI loopback is the easiest way to
  capture "everything playing through the speakers" without a virtual cable,
  and it is Windows-only. Microphone-only and cross-platform capture are
  tracked as future work.
- **Diarization runs per chunk** — both the energy-based and pyannote
  back-ends process each ~30 s chunk in isolation, so global speaker labels
  can drift across long meetings. The `transcribe` command currently transcribes
  existing files but does not add speaker labels.
- **Energy-based diarizer is a heuristic** — it leans on RMS, spectral
  centroid, and zero-crossing rate. It works well when speakers have
  distinct pitch/timbre and take clear turns; it struggles with overlapping
  speech and speakers with similar voices.
- **First-run model download** — the Whisper model (~488 MB for `small`) is
  fetched on first use and cached. Plan accordingly on metered connections.

## Privacy and data handling

- Local transcription and diarization run on your computer after their initial
  model downloads. The project contains no telemetry or analytics.
- Cloud transcription uploads audio to the configured API endpoint. Do not use
  cloud mode for material that the provider is not authorized to process.
- Recordings and transcripts are stored unencrypted in the output directory.
  Protect, retain, share, and delete these files according to your requirements.
- API tokens are read from environment variables; do not put real tokens in
  source files, command examples, bug reports, or committed configuration.

See [SECURITY.md](SECURITY.md) for reporting security issues.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for local
setup, the test command, and lint configuration. The change history lives in
[CHANGELOG.md](CHANGELOG.md).

### Building the Windows installer

Install Python 3.10+, the lite runtime requirements, the pinned packaging tools,
Inno Setup 6, and the Windows SDK (for `signtool.exe`):

```powershell
python -m pip install -r requirements-lite.txt
python -m pip install -r requirements-build.txt
$env:SIGN_CERT_SHA1 = "certificate thumbprint from the Windows certificate store"
.\installer\build.ps1
```

The build produces a PyInstaller onedir application and
`dist\MeetingRecorder-0.1.0-Setup.exe`. Both executables are SHA-256 signed and
timestamped. For a local packaging test without a certificate, use
`.\installer\build.ps1 -AllowUnsigned`; do not distribute unsigned builds.

Before publishing, install the generated package on clean Windows 10 and 11
machines and verify first launch, recording start/stop, settings persistence,
startup-on-login, upgrade over the previous version, and uninstall. Upgrades
retain `%APPDATA%\Meeting Recorder` settings and recordings stored outside the
installation directory.

## License

MIT — see [LICENSE](LICENSE) for details.
