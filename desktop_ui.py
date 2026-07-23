"""Native setup and settings window for the desktop tray application."""

import os
import threading
from typing import Any

from config import CLOUD_TRANSCRIPTION_PROVIDERS, Config
from desktop_settings import mark_setup_complete, save_settings

MODEL_SIZES = ("tiny", "base", "small", "medium", "large-v2")
OUTPUT_FORMATS = ("txt", "srt", "json", "all")
PROVIDERS = ("local", *CLOUD_TRANSCRIPTION_PROVIDERS)
_window_lock = threading.Lock()


def _optional_positive_int(value: str, label: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    return parsed


def apply_form_values(config: Config, values: dict[str, Any]) -> None:
    """Validate settings form values and apply them to the live config."""
    output_dir = str(values["output_dir"]).strip()
    if not output_dir:
        raise ValueError("Choose a recordings folder.")

    gain = float(values["microphone_gain"])
    if gain <= 0:
        raise ValueError("Microphone gain must be greater than zero.")

    speaker_count = _optional_positive_int(str(values["speaker_count"]), "Speaker count")
    max_speakers = _optional_positive_int(str(values["max_speakers"]), "Maximum speakers")
    if max_speakers is None:
        raise ValueError("Maximum speakers is required.")
    if speaker_count is not None and speaker_count > max_speakers:
        max_speakers = speaker_count

    config.output_dir = output_dir
    config.output_format = values["output_format"]
    config.model_size = values["model_size"]
    config.language = str(values["language"]).strip() or None
    config.speaker_count = speaker_count
    config.max_speakers = max_speakers
    config.device_index = values["device_index"]
    config.device_name = values.get("device_name")
    config.include_microphone = bool(values["include_microphone"])
    config.microphone_device_index = values["microphone_device_index"]
    config.microphone_device_name = values.get("microphone_device_name")
    config.microphone_gain = gain
    config.transcription_provider = values["transcription_provider"]
    config.transcription_model = str(values["transcription_model"]).strip() or "whisper-1"
    config.transcription_base_url = str(values["transcription_base_url"]).strip() or None


def _list_audio_devices(
    config: Config,
) -> tuple[list[tuple[str, int | None, str | None]], list[tuple[str, int | None, str | None]]]:
    loopback_devices = [("Default Windows output", None, None)]
    microphones = [("Default microphone", None, None)]
    try:
        from audio_capture import AudioCapture

        capture = AudioCapture(config)
        try:
            for device in capture.list_devices():
                label = f"[{device['index']}] {device['name']}"
                if device["loopback"]:
                    loopback_devices.append((label, device["index"], device["name"]))
                elif device["channels"] > 0:
                    microphones.append((label, device["index"], device["name"]))
        finally:
            capture.cleanup()
    except Exception as exc:
        print(f"⚠️  Audio devices could not be listed: {exc}")
    return loopback_devices, microphones


def run_settings_dialog(config: Config, first_run: bool = False) -> bool:
    """Show a modal native settings window and return whether settings were saved."""
    if not _window_lock.acquire(blocking=False):
        return False
    try:
        return _run_settings_dialog(config, first_run)
    finally:
        _window_lock.release()


def _run_settings_dialog(config: Config, first_run: bool) -> bool:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"❌ Unable to open desktop settings: {exc}")
        return False

    root.title("Meeting Recorder Setup" if first_run else "Meeting Recorder Settings")
    root.resizable(False, False)
    result = False
    content = ttk.Frame(root, padding=16)
    content.grid(sticky="nsew")
    content.columnconfigure(1, weight=1)

    row = 0
    if first_run:
        ttk.Label(content, text="Welcome to Meeting Recorder", font=("", 14, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        row += 1
        notice = (
            "Only record people with their knowledge and consent. Recordings and transcripts are stored "
            "unencrypted in the folder below. Microphone mixing records your voice. Local transcription "
            "stays on this PC after model downloads; cloud providers receive uploaded meeting audio."
        )
        ttk.Label(content, text=notice, wraplength=620, justify="left").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )
        row += 1

    loopback_devices, microphones = _list_audio_devices(config)
    loopback_labels = [label for label, _, _ in loopback_devices]
    microphone_labels = [label for label, _, _ in microphones]
    loopback_by_label = {label: (index, name) for label, index, name in loopback_devices}
    microphone_by_label = {label: (index, name) for label, index, name in microphones}

    def selected_label(
        devices: list[tuple[str, int | None, str | None]], index: int | None, name: str | None
    ) -> str:
        return next(
            (label for label, value, device_name in devices if value == index and (not name or device_name == name)),
            devices[0][0],
        )

    output_var = tk.StringVar(value=config.output_dir)
    output_format_var = tk.StringVar(value=config.output_format)
    loopback_var = tk.StringVar(value=selected_label(loopback_devices, config.device_index, config.device_name))
    microphone_var = tk.StringVar(
        value=selected_label(microphones, config.microphone_device_index, config.microphone_device_name)
    )
    include_mic_var = tk.BooleanVar(value=config.include_microphone)
    mic_gain_var = tk.StringVar(value=f"{config.microphone_gain:g}")
    model_var = tk.StringVar(value=config.model_size)
    language_var = tk.StringVar(value=config.language or "")
    speaker_count_var = tk.StringVar(value="" if config.speaker_count is None else str(config.speaker_count))
    max_speakers_var = tk.StringVar(value=str(config.max_speakers))
    provider_var = tk.StringVar(value=config.transcription_provider)
    transcription_model_var = tk.StringVar(value=config.transcription_model)
    base_url_var = tk.StringVar(value=config.transcription_base_url or "")
    consent_var = tk.BooleanVar(value=not first_run)

    def add_label(text: str) -> None:
        nonlocal row
        ttk.Label(content, text=text).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)

    add_label("Recordings folder")
    ttk.Entry(content, textvariable=output_var, width=55).grid(row=row, column=1, sticky="ew", pady=4)

    def browse_output() -> None:
        selected = filedialog.askdirectory(initialdir=output_var.get() or os.getcwd())
        if selected:
            output_var.set(selected)

    ttk.Button(content, text="Browse…", command=browse_output).grid(row=row, column=2, padx=(8, 0), pady=4)
    row += 1

    fields = (
        ("System audio", loopback_var, loopback_labels),
        ("Microphone", microphone_var, microphone_labels),
        ("Local Whisper model", model_var, MODEL_SIZES),
        ("Transcript format", output_format_var, OUTPUT_FORMATS),
        ("Transcription provider", provider_var, PROVIDERS),
    )
    for label, variable, choices in fields:
        add_label(label)
        ttk.Combobox(content, textvariable=variable, values=choices, state="readonly", width=52).grid(
            row=row, column=1, columnspan=2, sticky="ew", pady=4
        )
        row += 1

    entries = (
        ("Language (blank = auto)", language_var),
        ("Exact speaker count", speaker_count_var),
        ("Maximum speakers", max_speakers_var),
        ("Microphone gain", mic_gain_var),
        ("Cloud model", transcription_model_var),
        ("Cloud base URL (optional)", base_url_var),
    )
    for label, variable in entries:
        add_label(label)
        ttk.Entry(content, textvariable=variable, width=55).grid(
            row=row, column=1, columnspan=2, sticky="ew", pady=4
        )
        row += 1

    ttk.Checkbutton(content, text="Mix microphone into recordings", variable=include_mic_var).grid(
        row=row, column=0, columnspan=3, sticky="w", pady=(8, 4)
    )
    row += 1
    ttk.Label(
        content,
        text=(
            "Local models download automatically on first recording. Accurate pyannote speaker detection "
            "requires the optional full install and HF_TOKEN. Cloud API keys are read from environment variables "
            "and are never saved here."
        ),
        wraplength=620,
        justify="left",
    ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(4, 10))
    row += 1

    if first_run:
        ttk.Checkbutton(
            content,
            text="I will record only with the participants' knowledge and consent.",
            variable=consent_var,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))
        row += 1

    def save() -> None:
        nonlocal result
        if first_run and not consent_var.get():
            messagebox.showerror("Consent required", "Acknowledge the recording consent notice to continue.")
            return
        loopback_index, loopback_name = loopback_by_label[loopback_var.get()]
        microphone_index, microphone_name = microphone_by_label[microphone_var.get()]
        values = {
            "output_dir": output_var.get(),
            "output_format": output_format_var.get(),
            "device_index": loopback_index,
            "device_name": loopback_name,
            "microphone_device_index": microphone_index,
            "microphone_device_name": microphone_name,
            "include_microphone": include_mic_var.get(),
            "microphone_gain": mic_gain_var.get(),
            "model_size": model_var.get(),
            "language": language_var.get(),
            "speaker_count": speaker_count_var.get(),
            "max_speakers": max_speakers_var.get(),
            "transcription_provider": provider_var.get(),
            "transcription_model": transcription_model_var.get(),
            "transcription_base_url": base_url_var.get(),
        }
        try:
            apply_form_values(config, values)
            save_settings(config)
            if first_run:
                mark_setup_complete()
        except (KeyError, OSError, TypeError, ValueError) as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        result = True
        root.destroy()

    buttons = ttk.Frame(content)
    buttons.grid(row=row, column=0, columnspan=3, sticky="e", pady=(4, 0))
    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side="right", padx=(8, 0))
    ttk.Button(buttons, text="Save", command=save).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return result
