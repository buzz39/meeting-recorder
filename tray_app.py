"""
System tray application for Meeting Recorder (Windows).

Provides a system tray icon with recording controls.
Requires: pystray, Pillow

NOTE: This module is Windows-only. On other platforms, it will show an error.
"""

import os
import sys
import threading
import time
import subprocess
from datetime import datetime

try:
    import pystray
    from PIL import Image, ImageDraw
    _TRAY_AVAILABLE = True
except ImportError:
    _TRAY_AVAILABLE = False

from config import Config


def _create_icon(color: str = "gray", size: int = 64) -> "Image.Image":
    """Create a simple circle icon. Red = recording, yellow = stopping, gray = idle."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 4
    if color == "red":
        fill = (220, 50, 50, 255)
    elif color == "yellow":
        fill = (230, 180, 30, 255)
    else:
        fill = (150, 150, 150, 255)
    draw.ellipse([margin, margin, size - margin, size - margin], fill=fill)
    return img


class TrayApp:
    """System tray interface for Meeting Recorder."""

    def __init__(self, config: Config):
        self.config = config
        self.is_recording = False
        self._is_stopping = False
        self._recorder = None
        self._record_thread = None
        self._start_time = None
        self._icon = None

    def _get_tooltip(self) -> str:
        if self._is_stopping:
            return "Meeting Recorder — Stopping…"
        if self.is_recording and self._start_time:
            elapsed = int(time.time() - self._start_time)
            mins, secs = divmod(elapsed, 60)
            return f"Recording… {mins}:{secs:02d}"
        return "Meeting Recorder — Idle"

    def _update_icon(self):
        if self._icon:
            if self._is_stopping:
                color = "yellow"
            elif self.is_recording:
                color = "red"
            else:
                color = "gray"
            self._icon.icon = _create_icon(color)
            self._icon.title = self._get_tooltip()

    def _start_recording(self, icon=None, item=None):
        if self.is_recording or self._is_stopping:
            return

        self.is_recording = True
        self._start_time = time.time()
        self._update_icon()

        def _record():
            from recorder import Recorder
            self._recorder = Recorder(self.config)
            self._recorder.start_recording()
            # Recording ended (Ctrl+C or stop)
            self.is_recording = False
            self._is_stopping = False
            self._start_time = None
            self._update_icon()
            self._notify("Recording saved", "Your meeting recording has been saved.")

        self._record_thread = threading.Thread(target=_record, daemon=True)
        self._record_thread.start()

        # Start tooltip updater
        def _update_tooltip():
            while self.is_recording or self._is_stopping:
                self._update_icon()
                time.sleep(1)

        threading.Thread(target=_update_tooltip, daemon=True).start()

    def _stop_recording(self, icon=None, item=None):
        if not self.is_recording or not self._recorder or self._is_stopping:
            return
        # Show immediate visual feedback
        self._is_stopping = True
        self._update_icon()
        # Signal the recorder to stop (non-blocking; recorder thread handles the rest)
        self._recorder._stop_event.set()

    def _notify(self, title: str, message: str):
        """Show a system tray notification if supported."""
        if self._icon:
            try:
                self._icon.notify(message, title)
            except Exception:
                # Not all pystray backends support notifications
                pass

    def _open_recordings(self, icon=None, item=None):
        folder = self.config.output_dir
        os.makedirs(folder, exist_ok=True)
        # Windows-specific
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

    def _quit(self, icon=None, item=None):
        if self.is_recording:
            self._stop_recording()
            # Wait for the recorder thread to finish saving.  The responsive
            # stop should complete well within this window; 10 s is a generous
            # upper bound that covers final WAV/transcript writes.
            if self._record_thread:
                self._record_thread.join(timeout=10)
        if self._icon:
            self._icon.stop()

    def run(self):
        """Launch the system tray icon."""
        menu = pystray.Menu(
            pystray.MenuItem(
                "Start Recording",
                self._start_recording,
                enabled=lambda item: not self.is_recording and not self._is_stopping,
            ),
            pystray.MenuItem(
                lambda item: "Stopping…" if self._is_stopping else "Stop Recording",
                self._stop_recording,
                enabled=lambda item: self.is_recording and not self._is_stopping,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open Recordings Folder", self._open_recordings),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )

        self._icon = pystray.Icon(
            name="meeting-recorder",
            icon=_create_icon("gray"),
            title=self._get_tooltip(),
            menu=menu,
        )

        print("🖥️  Meeting Recorder running in system tray.")
        print("   Right-click the tray icon to start/stop recording.")
        self._icon.run()


def run_tray(config: Config):
    """Entry point for tray mode."""
    if not _TRAY_AVAILABLE:
        print("❌ System tray requires pystray and Pillow.")
        print("   Install: pip install pystray Pillow")
        sys.exit(1)

    app = TrayApp(config)
    app.run()
