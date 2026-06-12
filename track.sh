#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true
python -m src.recognize_with_tracking "$@"
