@echo off
REM Windows setup script for face recognition project

echo.
echo ======================================================================
echo  Face Recognition Setup (Windows)
echo ======================================================================
echo.

REM Check Python
python --version || (
    echo ERROR: Python not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment (.venv)...
if exist .venv (
    echo Virtual environment already exists. Skipping creation.
) else (
    python -m venv .venv || (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo [2/5] Activating virtual environment...
call .venv\Scripts\activate.bat || (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo [3/5] Ensuring pip is available...
python -m ensurepip --upgrade || (
    echo ERROR: Failed to bootstrap pip
    pause
    exit /b 1
)

echo [4/5] Upgrading pip and build tools...
python -m pip install --upgrade pip setuptools wheel || (
    echo ERROR: Failed to upgrade pip
    pause
    exit /b 1
)

echo [5/5] Installing dependencies (large downloads; may take a while)...
python -m pip install --default-timeout=1000 --retries 15 --no-cache-dir -r requirements.txt || (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo Creating project directories...
python -c "from src import config; config.ensure_dirs()" || (
    echo ERROR: Failed to create directories
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo  Setup Complete!
echo ======================================================================
echo.
echo Next steps:
echo   1. Activate environment: .venv\Scripts\activate.bat
echo   2. Download ArcFace model: python -m src.download_model
echo   3. Test camera: python -m src.camera
echo   4. Start enrollment: python -m src.enroll
echo   5. Run recognition: python -m src.recognize
echo.
pause
