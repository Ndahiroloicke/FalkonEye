@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python test_simple_tracking.py
pause
