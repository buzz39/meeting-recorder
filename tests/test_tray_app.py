import threading

from config import Config
from tray_app import TrayApp


class FakeThread:
    def __init__(self):
        self.timeout = None

    def join(self, timeout=None):
        self.timeout = timeout


class FakeIcon:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakeRecorder:
    def __init__(self):
        self._stop_event = threading.Event()


def test_quit_stops_and_waits_for_active_recording():
    app = TrayApp(Config())
    app.is_recording = True
    app._recorder = FakeRecorder()
    app._record_thread = FakeThread()
    app._icon = FakeIcon()
    app._update_icon = lambda: None

    app._quit()

    assert app._recorder._stop_event.is_set()
    assert app._record_thread.timeout == 10
    assert app._icon.stopped
