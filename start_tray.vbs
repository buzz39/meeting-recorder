' Meeting Recorder - Silent Tray Launcher
' Double-click this file to start Meeting Recorder silently (no console window)

Set WshShell = CreateObject("WScript.Shell")
scriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

' Run the batch file hidden (no console window)
WshShell.Run """" & scriptDir & "\start_tray.bat""", 0, False
