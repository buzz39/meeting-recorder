# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- JSON transcript output (`--format json`, also included in `--format all`).
- Auto-detect CUDA for transcription; CPU stays the default fallback.
- `pyproject.toml` with a `meeting-recorder` console script entry point and
  optional dependency groups (`windows`, `pyannote`, `dev`).
- `tests/` covering timestamp formatters, transcript writers (txt/srt/json),
  output-path selection, and the energy-based diarizer.
- GitHub Actions CI workflow: ruff lint + pytest on
  Linux/macOS/Windows × Python 3.10/3.11/3.12.
- `CONTRIBUTING.md` describing local setup, tests, and lint.
- README: badges, "vs alternatives" comparison, "Known limitations" section,
  and a JSON output example.

### Changed
- `audio_capture.save_wav` now writes int16 PCM WAVs (universally playable);
  previous behaviour wrote raw float32 with a PCM header that most players
  rendered as static.
- Resampling to 16 kHz now uses `scipy.signal.resample_poly` (polyphase with
  anti-aliasing) when available, falling back to linear interpolation. The
  previous nearest-neighbour decimation aliased badly at 48 kHz → 16 kHz and
  measurably hurt transcription accuracy.
- Heavy imports (`faster_whisper`) are now lazy, so importing CLI helpers no
  longer requires the full ML stack.
- Dependencies have explicit upper bounds (notably `pyannote.audio<4` and
  `torch<3`) to insulate users from breaking releases.

### Fixed
- SIGINT handler installed by `start_recording` is now restored on exit, so
  subsequent recordings (e.g. successive runs from the tray app) and the
  parent shell are no longer affected.
