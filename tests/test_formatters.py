"""Tests for timestamp formatting helpers."""

from recorder import format_srt_timestamp, format_timestamp


def test_format_timestamp_zero():
    assert format_timestamp(0) == "00:00:00.000"


def test_format_timestamp_subsecond():
    assert format_timestamp(0.345) == "00:00:00.345"


def test_format_timestamp_minutes_seconds():
    assert format_timestamp(75.5) == "00:01:15.500"


def test_format_timestamp_hours():
    # 1h 2m 3.456s
    assert format_timestamp(3723.456) == "01:02:03.456"


def test_format_srt_timestamp_uses_comma_separator():
    # SRT requires comma between seconds and milliseconds.
    assert format_srt_timestamp(75.5) == "00:01:15,500"


def test_format_srt_timestamp_zero():
    assert format_srt_timestamp(0) == "00:00:00,000"


def test_format_srt_timestamp_hours():
    assert format_srt_timestamp(3723.456) == "01:02:03,456"
