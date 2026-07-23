"""Windowed entry point used by the packaged Windows desktop application."""

from config import Config
from desktop_settings import apply_desktop_settings
from tray_app import run_tray


def main() -> None:
    config = apply_desktop_settings(Config())
    run_tray(config)


if __name__ == "__main__":
    main()
