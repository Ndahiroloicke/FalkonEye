#!/usr/bin/env python3
"""Delete all enrolled faces, database, activity logs, and debug crops."""

import shutil
from pathlib import Path

from src import config


def reset_all_data() -> None:
    """Remove every user-generated record and recreate empty data dirs."""
    targets = [
        config.DB_DIR,
        config.ENROLL_DIR,
        config.HISTORY_DIR,
        config.DEBUG_ALIGNED_DIR,
        config.DATA_DIR / "activity_logs",
    ]

    removed = 0
    for path in targets:
        if path.exists():
            shutil.rmtree(path)
            removed += 1

    config.ensure_dirs()

    print("✓ All user data removed.")
    print(f"  Cleared {removed} data folder(s)")
    print("  Fresh directories created under data/")
    print("\nNext steps:")
    print("  python -m src.enroll")
    print("  python -m src.recognize")


if __name__ == "__main__":
    reset_all_data()
