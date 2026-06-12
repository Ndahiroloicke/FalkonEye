@echo off
cd /d "%~dp0"
.\.venv\Scripts\python.exe test_servo_sweep.py %*
pause
