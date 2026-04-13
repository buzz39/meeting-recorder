@echo off
:: Meeting Recorder - System Tray Launcher
:: Double-click this file to start Meeting Recorder in system tray mode

:: Suppress warnings for clean startup
set PYTHONWARNINGS=ignore

:: Check if we're in a venv, if not activate it
if exist "%~dp0venv\Scripts\activate.bat" (
    call "%~dp0venv\Scripts\activate.bat"
)

:: Launch tray mode
python "%~dp0recorder.py" tray

pause
