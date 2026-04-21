# Contributing to Meeting Recorder

Thanks for your interest in improving Meeting Recorder! This project is small,
so the contribution loop is intentionally lightweight.

## Quick start

```bash
git clone https://github.com/buzz39/meeting-recorder.git
cd meeting-recorder
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate

# Install dev tooling + lightweight runtime deps used by tests.
pip install -e ".[dev]"
pip install numpy
```

The heavy runtime dependencies (`faster-whisper`, `torch`, `pyannote.audio`,
`pyaudiowpatch`) are imported lazily, so you do not need them installed to run
the unit tests.

## Running the tests

```bash
pytest
```

CI runs the same suite on Linux, macOS, and Windows against Python 3.10–3.12.

## Linting

```bash
ruff check .
# Auto-fix what's safe:
ruff check --fix .
```

The repository is configured for `ruff` (lint + import sort) via
`pyproject.toml`. Please make sure `ruff check .` passes before opening a PR.

## Pull requests

- Keep PRs small and focused. One change per PR is ideal.
- Add or update tests for any behaviour change in pure-Python code.
- Update `CHANGELOG.md` under the `[Unreleased]` heading.
- For changes that touch audio capture, the system tray, or the neural
  diarizer, please describe the platform you tested on (Windows version,
  Python version, GPU/CPU). CI cannot exercise these paths.

## Reporting bugs

Open an issue with:
- OS + Python version
- Output of `python recorder.py devices`
- The full command you ran and the error/log output

## Code style notes

- Prefer small, composable functions; avoid adding more responsibilities to
  `recorder.py` (it should orchestrate, not format).
- New transcript output formats belong in `_save_transcript` plus
  `_get_output_paths` (and the argparse `choices` lists).
- New configuration knobs should live on `Config` with a sensible default and
  a comment explaining the trade-off.
