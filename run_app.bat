@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
  echo First-time setup needed...
  call setup.bat
)
start "" .venv\Scripts\pythonw.exe rockwizard.pyw
