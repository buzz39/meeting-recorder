#!/usr/bin/env python3
"""
Meeting Recorder — CLI tool for recording and transcribing system audio.

Captures system audio via WASAPI loopback (Windows), transcribes locally
using faster-whisper, and provides speaker diarization.

Usage:
    python recorder.py start [--model base] [--output ./recordings] [--format txt]
    python recorder.py list
    python recorder.py transcribe <audio_file>
    python recorder.py devices
"""

import argparse
import os
import sys
import signal
import time
import threading
from datetime import datetime

from config import Config
from transcriber import Transcriber
from diarizer import Diarizer


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class Recorder:
    """Main recording orchestrator."""

    def __init__(self, config: Config):
        self.config = config
        self.transcriber = Transcriber(config)
        self.diarizer = Diarizer(config)
        self._stop_event = threading.Event()
        self._all_segments: list[dict] = []

    def start_recording(self):
        """Start recording system audio with real-time transcription."""
        # Import here so the module can be loaded on Linux for testing structure
        from audio_capture import AudioCapture

        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Generate session name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"meeting_{timestamp}"
        wav_path = os.path.join(self.config.output_dir, f"{session_name}.wav")

        print("=" * 60)
        print(f"🎙️  Meeting Recorder")
        print(f"   Model: {self.config.model_size} | Format: {self.config.output_format}")
        print(f"   Output: {self.config.output_dir}")
        print(f"   Press Ctrl+C to stop recording")
        print("=" * 60)

        # Determine output paths based on format
        output_paths = self._get_output_paths(self.config.output_dir, session_name)

        # Load whisper model
        self.transcriber.load_model()

        # Setup audio capture
        capture = AudioCapture(self.config)
        try:
            capture.start()
        except Exception as e:
            print(f"\n❌ Failed to start audio capture: {e}")
            capture.cleanup()
            return

        print(f"\n🔴 Recording... (saving to {session_name})")
        print("-" * 60)

        # Handle Ctrl+C (only possible from the main thread; when launched
        # from the tray app the stop_event is set directly instead).
        if threading.current_thread() is threading.main_thread():
            def signal_handler(sig, frame):
                print("\n\n⏹️  Stopping recording...")
                self._stop_event.set()

            signal.signal(signal.SIGINT, signal_handler)

        # Process audio in chunks
        chunk_count = 0
        self.diarizer.reset()
        self._all_segments = []

        try:
            while not self._stop_event.is_set():
                chunk_offset = chunk_count * self.config.chunk_duration
                audio_chunk = capture.get_chunk(self.config.chunk_duration)

                if audio_chunk is None or len(audio_chunk) == 0:
                    continue

                # Transcribe chunk
                segments = self.transcriber.transcribe(audio_chunk, chunk_offset)

                if segments:
                    # Assign speakers
                    # For diarization, we need the chunk at 16kHz (which get_chunk returns)
                    # But segment timestamps are relative to chunk_offset, so adjust for audio indexing
                    segments_local = []
                    for s in segments:
                        segments_local.append({
                            **s,
                            "start": s["start"] - chunk_offset,
                            "end": s["end"] - chunk_offset,
                        })
                    segments_local = self.diarizer.assign_speakers(audio_chunk, segments_local)

                    # Restore global timestamps and print
                    for s_local, s_global in zip(segments_local, segments):
                        s_global["speaker"] = s_local["speaker"]
                        ts = format_timestamp(s_global["start"])
                        print(f"  [{ts}] {s_global['speaker']}: {s_global['text']}")

                    self._all_segments.extend(segments)

                chunk_count += 1

        except Exception as e:
            print(f"\n❌ Error during recording: {e}")

        # Stop and save
        frames = capture.stop()

        print("-" * 60)
        print(f"\n💾 Saving audio to {wav_path}")
        capture.save_wav(wav_path, frames)

        for fmt, path in output_paths.items():
            print(f"💾 Saving {fmt} transcript to {path}")
            self._save_transcript(path, self._all_segments, fmt)

        capture.cleanup()
        print(f"\n✅ Recording saved! ({chunk_count} chunks processed)")
        print(f"   Audio: {wav_path}")
        for fmt, path in output_paths.items():
            print(f"   Transcript ({fmt}): {path}")

    def _get_output_paths(self, output_dir: str, session_name: str) -> dict[str, str]:
        """Return dict of format -> filepath based on config.output_format."""
        paths = {}
        fmt = self.config.output_format
        if fmt in ("txt", "all"):
            paths["txt"] = os.path.join(output_dir, f"{session_name}.txt")
        if fmt in ("srt", "all"):
            paths["srt"] = os.path.join(output_dir, f"{session_name}.srt")
        return paths

    def _save_transcript(self, filepath: str, segments: list[dict], fmt: str = None):
        """Save transcript to file in txt or srt format."""
        if fmt is None:
            fmt = "srt" if filepath.endswith(".srt") else "txt"
        with open(filepath, "w", encoding="utf-8") as f:
            if fmt == "srt":
                for i, seg in enumerate(segments, 1):
                    start = format_srt_timestamp(seg["start"])
                    end = format_srt_timestamp(seg["end"])
                    speaker = seg.get("speaker", "Unknown")
                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"[{speaker}] {seg['text']}\n\n")
            else:
                for seg in segments:
                    ts = format_timestamp(seg["start"])
                    speaker = seg.get("speaker", "Unknown")
                    f.write(f"[{ts}] {speaker}: {seg['text']}\n")

    def transcribe_file(self, filepath: str):
        """Transcribe an existing audio file."""
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return

        print(f"📄 Transcribing: {filepath}")
        self.transcriber.load_model()

        segments = self.transcriber.transcribe_file(filepath)

        if not segments:
            print("No speech detected in the audio file.")
            return

        print(f"\n{'=' * 60}")
        print(f"Transcript ({len(segments)} segments)")
        print(f"{'=' * 60}\n")

        for seg in segments:
            ts = format_timestamp(seg["start"])
            print(f"  [{ts}] {seg['text']}")

        # Save transcript next to audio file
        base = os.path.splitext(filepath)[0]
        fmt = self.config.output_format
        if fmt == "all":
            for f in ("txt", "srt"):
                path = f"{base}_transcript.{f}"
                self._save_transcript(path, segments, f)
                print(f"\n💾 Transcript saved to: {path}")
        else:
            path = f"{base}_transcript.{fmt}"
            self._save_transcript(path, segments, fmt)
            print(f"\n💾 Transcript saved to: {path}")

    def list_recordings(self):
        """List all past recordings."""
        if not os.path.exists(self.config.output_dir):
            print("No recordings directory found.")
            return

        files = sorted(
            f for f in os.listdir(self.config.output_dir) if f.endswith(".wav")
        )

        if not files:
            print("No recordings found.")
            return

        print(f"\n📁 Recordings in {self.config.output_dir}:\n")
        for f in files:
            path = os.path.join(self.config.output_dir, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            # Check for transcript
            base = os.path.splitext(f)[0]
            has_txt = os.path.exists(os.path.join(self.config.output_dir, f"{base}.txt"))
            has_srt = os.path.exists(os.path.join(self.config.output_dir, f"{base}.srt"))
            transcript = " 📝" if (has_txt or has_srt) else ""
            print(f"  {f}  ({size_mb:.1f} MB){transcript}")

    def list_devices(self):
        """List available audio devices."""
        from audio_capture import AudioCapture
        capture = AudioCapture(self.config)
        devices = capture.list_devices()
        capture.cleanup()

        print(f"\n🔊 Audio Devices:\n")
        for d in devices:
            loopback = " [LOOPBACK]" if d["loopback"] else ""
            print(f"  [{d['index']}] {d['name']} (ch={d['channels']}, rate={d['rate']}){loopback}")


def main():
    parser = argparse.ArgumentParser(
        description="Meeting Recorder — Record & transcribe system audio locally"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # start
    start_parser = subparsers.add_parser("start", help="Start recording")
    start_parser.add_argument("--model", default="small", help="Whisper model size (tiny/base/small/medium/large-v2)")
    start_parser.add_argument("--output", default=None, help="Output directory")
    start_parser.add_argument("--format", default="txt", choices=["txt", "srt", "all"], help="Transcript format (all = txt + srt)")
    start_parser.add_argument("--device", type=int, default=None, help="Audio device index")
    start_parser.add_argument("--language", default=None, help="Language code (e.g. en, hi)")
    start_parser.add_argument("--chunk", type=float, default=30.0, help="Chunk duration in seconds")

    # list
    subparsers.add_parser("list", help="List past recordings")

    # transcribe
    trans_parser = subparsers.add_parser("transcribe", help="Transcribe an audio file")
    trans_parser.add_argument("file", help="Path to audio file")
    trans_parser.add_argument("--model", default="small", help="Whisper model size")
    trans_parser.add_argument("--format", default="txt", choices=["txt", "srt", "all"])

    # devices
    subparsers.add_parser("devices", help="List audio devices")

    # tray (Windows system tray mode)
    tray_parser = subparsers.add_parser("tray", help="Launch in system tray mode (Windows)")
    tray_parser.add_argument("--model", default="small", help="Whisper model size")
    tray_parser.add_argument("--output", default=None, help="Output directory")
    tray_parser.add_argument("--format", default="all", choices=["txt", "srt", "all"])
    tray_parser.add_argument("--language", default=None, help="Language code")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    config = Config()

    if hasattr(args, "model"):
        config.model_size = args.model
    if hasattr(args, "format"):
        config.output_format = args.format
    if hasattr(args, "output") and args.output:
        config.output_dir = args.output
    if hasattr(args, "device") and args.device is not None:
        config.device_index = args.device
    if hasattr(args, "language") and args.language:
        config.language = args.language
    if hasattr(args, "chunk"):
        config.chunk_duration = args.chunk

    recorder = Recorder(config)

    if args.command == "start":
        recorder.start_recording()
    elif args.command == "list":
        recorder.list_recordings()
    elif args.command == "transcribe":
        recorder.transcribe_file(args.file)
    elif args.command == "devices":
        recorder.list_devices()
    elif args.command == "tray":
        from tray_app import run_tray
        run_tray(config)


if __name__ == "__main__":
    main()
