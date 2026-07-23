"""PyInstaller configuration for the lightweight Windows desktop build."""

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

hiddenimports = []
datas = []
binaries = []

for package in ("faster_whisper", "pystray", "PIL", "pyaudiowpatch"):
    hiddenimports += collect_submodules(package)
    datas += collect_data_files(package)

binaries += collect_dynamic_libs("ctranslate2")

analysis = Analysis(
    ["desktop_main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "pyannote"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="MeetingRecorder",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
bundle = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    name="MeetingRecorder",
)
