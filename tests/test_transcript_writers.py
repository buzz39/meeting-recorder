"""Tests for transcript writers (txt / srt / json) and output path selection."""

import json
import os

import pytest

from config import Config
from recorder import Recorder


@pytest.fixture()
def segments():
    return [
        {"start": 0.0, "end": 2.5, "text": "Hello world.", "speaker": "Speaker 1"},
        {"start": 2.5, "end": 5.0, "text": "Second line.", "speaker": "Speaker 2"},
    ]


@pytest.fixture()
def recorder(tmp_path):
    cfg = Config()
    cfg.output_dir = str(tmp_path)
    # Bypass __init__ side effects (model/diarizer load) — we only need the
    # pure-Python writer methods under test.
    rec = Recorder.__new__(Recorder)
    rec.config = cfg
    return rec


def test_get_output_paths_txt(recorder):
    recorder.config.output_format = "txt"
    paths = recorder._get_output_paths(recorder.config.output_dir, "session")
    assert set(paths) == {"txt"}
    assert paths["txt"].endswith("session.txt")


def test_get_output_paths_srt(recorder):
    recorder.config.output_format = "srt"
    paths = recorder._get_output_paths(recorder.config.output_dir, "session")
    assert set(paths) == {"srt"}


def test_get_output_paths_json(recorder):
    recorder.config.output_format = "json"
    paths = recorder._get_output_paths(recorder.config.output_dir, "session")
    assert set(paths) == {"json"}
    assert paths["json"].endswith("session.json")


def test_get_output_paths_all(recorder):
    recorder.config.output_format = "all"
    paths = recorder._get_output_paths(recorder.config.output_dir, "session")
    assert set(paths) == {"txt", "srt", "json"}


def test_save_transcript_txt(recorder, segments, tmp_path):
    path = tmp_path / "out.txt"
    recorder._save_transcript(str(path), segments)
    content = path.read_text(encoding="utf-8")
    assert "[00:00:00.000] Speaker 1: Hello world." in content
    assert "[00:00:02.500] Speaker 2: Second line." in content


def test_save_transcript_srt(recorder, segments, tmp_path):
    path = tmp_path / "out.srt"
    recorder._save_transcript(str(path), segments)
    content = path.read_text(encoding="utf-8")
    assert content.startswith("1\n")
    assert "00:00:00,000 --> 00:00:02,500" in content
    assert "[Speaker 1] Hello world." in content
    assert "2\n" in content
    assert "[Speaker 2] Second line." in content


def test_save_transcript_json(recorder, segments, tmp_path):
    path = tmp_path / "out.json"
    recorder._save_transcript(str(path), segments)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert len(payload["segments"]) == 2
    assert payload["segments"][0] == {
        "start": 0.0,
        "end": 2.5,
        "speaker": "Speaker 1",
        "text": "Hello world.",
    }


def test_save_transcript_format_inferred_from_extension(recorder, segments, tmp_path):
    """When ``fmt`` is omitted, the writer should infer format from the path."""
    json_path = tmp_path / "out.json"
    recorder._save_transcript(str(json_path), segments)
    json.loads(json_path.read_text(encoding="utf-8"))  # parses successfully

    srt_path = tmp_path / "out.srt"
    recorder._save_transcript(str(srt_path), segments)
    assert "-->" in srt_path.read_text(encoding="utf-8")


def test_save_transcript_handles_missing_speaker(recorder, tmp_path):
    segs = [{"start": 0.0, "end": 1.0, "text": "Anonymous"}]
    path = tmp_path / "out.txt"
    recorder._save_transcript(str(path), segs)
    assert "Unknown: Anonymous" in path.read_text(encoding="utf-8")


def test_save_html_viewer(recorder, segments, tmp_path):
    path = tmp_path / "session.html"
    recorder._save_html(str(path), segments, "session.wav", "session")
    content = path.read_text(encoding="utf-8")
    # References the audio file via a relative src
    assert 'src="session.wav"' in content
    # Title is rendered
    assert "<title>session" in content
    # Each segment is present with its start time as data-start, and a seek button
    assert 'data-start="0.000"' in content
    assert 'data-start="2.500"' in content
    assert "Hello world." in content
    assert "Second line." in content
    assert "Speaker 1" in content and "Speaker 2" in content
    # Has the click-to-seek script wiring
    assert "currentTime" in content


def test_save_html_escapes_html_in_segments(recorder, tmp_path):
    segs = [{"start": 0.0, "end": 1.0, "text": "<script>alert(1)</script>", "speaker": "A&B"}]
    path = tmp_path / "out.html"
    recorder._save_html(str(path), segs, "x.wav", "t<>&")
    content = path.read_text(encoding="utf-8")
    # Raw segment text/speaker must not appear unescaped
    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in content
    assert "A&amp;B" in content
    # Title is escaped too
    assert "t&lt;&gt;&amp;" in content


def test_save_html_handles_empty_segments(recorder, tmp_path):
    path = tmp_path / "empty.html"
    recorder._save_html(str(path), [], "x.wav", "empty")
    content = path.read_text(encoding="utf-8")
    assert "No transcript segments." in content
    assert 'src="x.wav"' in content
