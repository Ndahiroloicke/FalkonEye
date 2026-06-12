#!/bin/bash
# macOS/Linux setup script for face recognition project

set -e  # Exit on error

echo ""
echo "======================================================================"
echo " Face Recognition Setup (macOS/Linux)"
echo "======================================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not installed"
    echo "Install from: https://www.python.org"
    exit 1
fi

echo "[1/5] Python found. Upgrading pip..."
python3 -m pip install --upgrade pip > /dev/null 2>&1 || true

echo "[2/5] Creating virtual environment (.venv)..."
if [ -d ".venv" ] && ! .venv/bin/python3 -m pip --version > /dev/null 2>&1; then
    echo "Existing .venv is broken (pip missing or Python mismatch). Recreating..."
    rm -rf .venv
fi

if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    python3 -m venv .venv
fi

echo "[3/5] Activating virtual environment..."
source .venv/bin/activate

echo "[4/5] Installing dependencies..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo "[5/5] Creating project directories..."
python -c "from src import config; config.ensure_dirs()"

echo ""
echo "======================================================================"
echo " Setup Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Download ArcFace model: python download_model.py"
echo "  3. Test camera: python -m src.camera"
echo "  4. Start enrollment: python -m src.enroll"
echo "  5. Run recognition: python -m src.recognize"
echo "  6. MQTT tracking: python track.py"
echo ""
