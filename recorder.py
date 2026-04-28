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
import html
import json
import os
import signal
import threading
from datetime import datetime

from config import Config

# Note: ``diarizer`` and ``transcriber`` are imported lazily inside
# ``Recorder.__init__`` because they transitively pull in heavy ML libraries
# (torch, pyannote.audio, faster-whisper). Keeping them out of module import
# means ``python recorder.py tray`` can spin up the system-tray UI quickly
# instead of paying minutes of import cost up front.


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
        # Imported lazily — see module-level note. These pull in heavy ML deps.
        from diarizer import Diarizer
        from transcriber import Transcriber

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
        print("🎙️  Meeting Recorder")
        provider = self.config.transcription_provider
        model = self.config.openai_model if provider == "openai" else self.config.model_size
        print(f"   Provider: {provider} | Model: {model} | Format: {self.config.output_format}")
        if self.config.speaker_count:
            print(f"   Speakers: fixed at {self.config.speaker_count}")
        elif self.config.max_speakers:
            print(f"   Max speakers: {self.config.max_speakers}")
        if self.config.include_microphone:
            print(f"   Microphone mix: on (gain {self.config.microphone_gain:g}x)")
        print(f"   Output: {self.config.output_dir}")
        print("   Press Ctrl+C to stop recording")
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
        previous_sigint = None
        if threading.current_thread() is threading.main_thread():
            def signal_handler(sig, frame):
                print("\n\n⏹️  Stopping recording...")
                self._stop_event.set()

            previous_sigint = signal.signal(signal.SIGINT, signal_handler)

        # Process audio in chunks
        chunk_count = 0
        self.diarizer.reset()
        self._all_segments = []

        try:
            while not self._stop_event.is_set():
                chunk_offset = chunk_count * self.config.chunk_duration
                audio_chunk = capture.get_chunk(self.config.chunk_duration, self._stop_event)

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

        # Always emit a small self-contained HTML viewer alongside the WAV
        # so users can play the audio and click any line to seek to it
        # without needing extra tooling.
        html_path = os.path.join(self.config.output_dir, f"{session_name}.html")
        print(f"💾 Saving HTML viewer to {html_path}")
        self._save_html(html_path, self._all_segments, os.path.basename(wav_path), session_name)

        capture.cleanup()
        # Restore the previous SIGINT handler so subsequent recordings (e.g.
        # in tray mode) and shell behaviour are not affected.
        if previous_sigint is not None:
            try:
                signal.signal(signal.SIGINT, previous_sigint)
            except (ValueError, TypeError):
                pass
        print(f"\n✅ Recording saved! ({chunk_count} chunks processed)")
        print(f"   Audio: {wav_path}")
        for fmt, path in output_paths.items():
            print(f"   Transcript ({fmt}): {path}")
        print(f"   Viewer (html): {html_path}")

    def _get_output_paths(self, output_dir: str, session_name: str) -> dict[str, str]:
        """Return dict of format -> filepath based on config.output_format."""
        paths = {}
        fmt = self.config.output_format
        if fmt in ("txt", "all"):
            paths["txt"] = os.path.join(output_dir, f"{session_name}.txt")
        if fmt in ("srt", "all"):
            paths["srt"] = os.path.join(output_dir, f"{session_name}.srt")
        if fmt in ("json", "all"):
            paths["json"] = os.path.join(output_dir, f"{session_name}.json")
        return paths

    def _save_transcript(self, filepath: str, segments: list[dict], fmt: str = None):
        """Save transcript to file in txt, srt, or json format."""
        if fmt is None:
            if filepath.endswith(".srt"):
                fmt = "srt"
            elif filepath.endswith(".json"):
                fmt = "json"
            else:
                fmt = "txt"
        with open(filepath, "w", encoding="utf-8") as f:
            if fmt == "srt":
                for i, seg in enumerate(segments, 1):
                    start = format_srt_timestamp(seg["start"])
                    end = format_srt_timestamp(seg["end"])
                    speaker = seg.get("speaker", "Unknown")
                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"[{speaker}] {seg['text']}\n\n")
            elif fmt == "json":
                payload = {
                    "version": 1,
                    "segments": [
                        {
                            "start": float(seg["start"]),
                            "end": float(seg["end"]),
                            "speaker": seg.get("speaker", "Unknown"),
                            "text": seg["text"],
                        }
                        for seg in segments
                    ],
                }
                json.dump(payload, f, ensure_ascii=False, indent=2)
            else:
                for seg in segments:
                    ts = format_timestamp(seg["start"])
                    speaker = seg.get("speaker", "Unknown")
                    f.write(f"[{ts}] {speaker}: {seg['text']}\n")

    def _save_html(self, filepath: str, segments: list[dict], audio_filename: str, title: str):
        """Save a self-contained HTML viewer with an audio player and a
        click-to-seek transcript.

        ``audio_filename`` should be a path relative to ``filepath`` (typically
        just the basename, since the HTML lives in the same directory as the
        WAV). No external assets or network requests — everything is inline.
        """
        # Pre-build the segment rows so the template stays readable.
        rows = []
        for seg in segments:
            start = float(seg["start"])
            ts = format_timestamp(start)
            speaker = html.escape(str(seg.get("speaker", "Unknown")))
            text = html.escape(str(seg.get("text", "")))
            rows.append(
                f'    <li data-start="{start:.3f}">'
                f'<button type="button" class="seek">[{ts}]</button> '
                f'<span class="speaker">{speaker}:</span> '
                f'<span class="text">{text}</span></li>'
            )
        rows_html = "\n".join(rows) if rows else '    <li class="empty">No transcript segments.</li>'

        safe_title = html.escape(title)
        safe_audio = html.escape(audio_filename, quote=True)

        document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{safe_title} — Meeting Recorder</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 820px; margin: 2em auto; padding: 0 1em; color: #222; }}
 h1 {{ font-size: 1.2em; margin-bottom: 0.25em; }}
 audio {{ width: 100%; margin: 0.75em 0 1em; }}
 ol {{ list-style: none; padding: 0; }}
 li {{ padding: 4px 6px; border-radius: 4px; line-height: 1.4; }}
 li.active {{ background: #fff4c2; }}
 button.seek {{ font-family: monospace; background: none; border: none; color: #0366d6; cursor: pointer; padding: 0; }}
 button.seek:hover {{ text-decoration: underline; }}
 .speaker {{ font-weight: 600; }}
 .empty {{ color: #888; font-style: italic; }}
</style>
</head>
<body>
<h1>{safe_title}</h1>
<audio id="player" controls preload="metadata" src="{safe_audio}"></audio>
<ol id="segments">
{rows_html}
</ol>
<script>
 (function () {{
  var player = document.getElementById("player");
  var items = document.querySelectorAll("#segments li[data-start]");
  items.forEach(function (li) {{
   var btn = li.querySelector("button.seek");
   if (!btn) return;
   btn.addEventListener("click", function () {{
    var t = parseFloat(li.getAttribute("data-start"));
    if (!isNaN(t)) {{ player.currentTime = t; player.play(); }}
   }});
  }});
  player.addEventListener("timeupdate", function () {{
   var t = player.currentTime;
   var current = null;
   items.forEach(function (li) {{
    var s = parseFloat(li.getAttribute("data-start"));
    if (!isNaN(s) && s <= t) current = li;
   }});
   items.forEach(function (li) {{ li.classList.remove("active"); }});
   if (current) current.classList.add("active");
  }});
 }})();
</script>
</body>
</html>
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(document)

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
        formats = ("txt", "srt", "json") if fmt == "all" else (fmt,)
        for f in formats:
            path = f"{base}_transcript.{f}"
            self._save_transcript(path, segments, f)
            print(f"\n💾 Transcript saved to: {path}")

        # Also emit a click-to-seek HTML viewer next to the audio file.
        html_path = f"{base}_transcript.html"
        self._save_html(
            html_path,
            segments,
            os.path.basename(filepath),
            os.path.basename(base),
        )
        print(f"💾 Viewer saved to: {html_path}")

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
            has_html = os.path.exists(os.path.join(self.config.output_dir, f"{base}.html"))
            transcript = " 📝" if (has_txt or has_srt) else ""
            viewer = " 🌐" if has_html else ""
            print(f"  {f}  ({size_mb:.1f} MB){transcript}{viewer}")

    def list_devices(self):
        """List available audio devices."""
        from audio_capture import AudioCapture
        capture = AudioCapture(self.config)
        devices = capture.list_devices()
        capture.cleanup()

        print("\n🔊 Audio Devices:\n")
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
    start_parser.add_argument("--format", default="txt", choices=["txt", "srt", "json", "all"], help="Transcript format (all = txt + srt + json)")
    start_parser.add_argument("--device", type=int, default=None, help="Audio device index")
    start_parser.add_argument("--language", default=None, help="Language code (e.g. en, hi)")
    start_parser.add_argument("--chunk", type=float, default=30.0, help="Chunk duration in seconds")
    start_parser.add_argument("--provider", default="local", choices=["local", "openai"], help="Transcription provider: local=faster-whisper, openai=cloud API")
    start_parser.add_argument("--openai-model", default=None, help="OpenAI transcription model (default: OPENAI_TRANSCRIBE_MODEL or whisper-1)")
    start_parser.add_argument("--speakers", type=int, default=None, help="Exact number of speakers, e.g. 2 for two-person meetings")
    start_parser.add_argument("--max-speakers", type=int, default=10, help="Maximum speaker labels when exact count is unknown")
    start_parser.add_argument("--include-mic", action="store_true", help="Mix your microphone with loopback audio")
    start_parser.add_argument("--mic-device", type=int, default=None, help="Microphone device index for --include-mic")
    start_parser.add_argument("--mic-gain", type=float, default=1.0, help="Microphone gain multiplier for --include-mic")

    # list
    subparsers.add_parser("list", help="List past recordings")

    # transcribe
    trans_parser = subparsers.add_parser("transcribe", help="Transcribe an audio file")
    trans_parser.add_argument("file", help="Path to audio file")
    trans_parser.add_argument("--model", default="small", help="Whisper model size")
    trans_parser.add_argument("--format", default="txt", choices=["txt", "srt", "json", "all"])
    trans_parser.add_argument("--provider", default="local", choices=["local", "openai"], help="Transcription provider: local=faster-whisper, openai=cloud API")
    trans_parser.add_argument("--openai-model", default=None, help="OpenAI transcription model")
    trans_parser.add_argument("--language", default=None, help="Language code")

    # devices
    subparsers.add_parser("devices", help="List audio devices")

    # tray (Windows system tray mode)
    tray_parser = subparsers.add_parser("tray", help="Launch in system tray mode (Windows)")
    tray_parser.add_argument("--model", default="small", help="Whisper model size")
    tray_parser.add_argument("--output", default=None, help="Output directory")
    tray_parser.add_argument("--format", default="all", choices=["txt", "srt", "json", "all"])
    tray_parser.add_argument("--language", default=None, help="Language code")
    tray_parser.add_argument("--provider", default="local", choices=["local", "openai"], help="Transcription provider: local=faster-whisper, openai=cloud API")
    tray_parser.add_argument("--openai-model", default=None, help="OpenAI transcription model")
    tray_parser.add_argument("--speakers", type=int, default=None, help="Exact number of speakers")
    tray_parser.add_argument("--max-speakers", type=int, default=10, help="Maximum speaker labels")
    tray_parser.add_argument("--include-mic", action="store_true", help="Mix your microphone with loopback audio")
    tray_parser.add_argument("--mic-device", type=int, default=None, help="Microphone device index for --include-mic")
    tray_parser.add_argument("--mic-gain", type=float, default=1.0, help="Microphone gain multiplier for --include-mic")

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
    if hasattr(args, "provider"):
        config.transcription_provider = args.provider
    if hasattr(args, "openai_model") and args.openai_model:
        config.openai_model = args.openai_model
    if hasattr(args, "speakers") and args.speakers is not None:
        config.speaker_count = args.speakers
        config.max_speakers = args.speakers
    elif hasattr(args, "max_speakers"):
        config.max_speakers = args.max_speakers
    if hasattr(args, "include_mic"):
        config.include_microphone = args.include_mic
    if hasattr(args, "mic_device") and args.mic_device is not None:
        config.microphone_device_index = args.mic_device
    if hasattr(args, "mic_gain"):
        config.microphone_gain = args.mic_gain

    # Tray mode does not need the heavy Recorder (and its ML deps) loaded
    # up-front — the tray app constructs the Recorder lazily when the user
    # actually clicks "Start Recording". Dispatch before instantiating it so
    # the tray icon appears quickly.
    if args.command == "tray":
        from tray_app import run_tray
        run_tray(config)
        return

    recorder = Recorder(config)

    if args.command == "start":
        recorder.start_recording()
    elif args.command == "list":
        recorder.list_recordings()
    elif args.command == "transcribe":
        recorder.transcribe_file(args.file)
    elif args.command == "devices":
        recorder.list_devices()


if __name__ == "__main__":
    main()
