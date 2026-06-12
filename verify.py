#!/usr/bin/env python3
"""Verify project structure and core modules."""

import sys
from pathlib import Path


def verify_project() -> bool:
    print("\n" + "=" * 70)
    print(" Project Integrity Verification")
    print("=" * 70 + "\n")

    root = Path(__file__).parent
    all_ok = True

    required_dirs = ["src", "data/db", "data/enroll", "data/history", "models", "arduino"]
    print("Directories:")
    for d in required_dirs:
        ok = (root / d).exists()
        print(f"  {'✓' if ok else '✗'} {d}/")
        all_ok &= ok

    required_files = [
        "src/config.py",
        "src/recognition_core.py",
        "src/recognize.py",
        "src/recognize_with_tracking.py",
        "src/mqtt_camera_controller.py",
        "src/tracking.py",
        "src/enroll.py",
        "src/embed.py",
        "src/haar_5pt.py",
        "src/face_mesh.py",
        "track.py",
        "requirements.txt",
        "download_model.py",
        "reset_data.py",
        "arduino/esp8266_camera_tracker/esp8266_camera_tracker.ino",
    ]
    print("\nFiles:")
    for f in required_files:
        ok = (root / f).exists()
        print(f"  {'✓' if ok else '✗'} {f}")
        all_ok &= ok

    print("\nImports:")
    try:
        from src.mqtt_camera_controller import MQTTCameraController
        from src.tracking import PanTracker
        from src.recognition_core import load_database
        print("  ✓ Core modules import OK")
    except Exception as exc:
        print(f"  ✗ Import failed: {exc}")
        all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print(" ✓ All checks passed")
        print("=" * 70)
        print("\nNext: python download_model.py → python -m src.enroll → python track.py")
        return True
    print(" ✗ Some checks failed")
    print("=" * 70)
    return False


if __name__ == "__main__":
    sys.exit(0 if verify_project() else 1)
