#!/usr/bin/env python3
"""Download ArcFace ONNX model and MediaPipe face landmarker model."""

import os
import sys
import urllib.request
from pathlib import Path

from src import config
from src.face_mesh import FACE_LANDMARKER_URL, ensure_face_landmarker_model


def download_arcface() -> bool:
    config.ensure_dirs()
    if config.ARCFACE_MODEL_PATH.exists():
        print(f"✓ ArcFace model already exists: {config.ARCFACE_MODEL_PATH}")
        return True

    url = "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"
    zip_path = config.PROJECT_ROOT / "buffalo_l.zip"
    print(f"Downloading ArcFace model (~170 MB)...")
    print(f"  {url}")

    try:
        os.system(f'curl -L -o "{zip_path}" "{url}"')
        if not zip_path.exists():
            print("ERROR: Download failed.")
            return False
        os.system(f'unzip -o "{zip_path}" -d "{config.PROJECT_ROOT}"')
        src_model = config.PROJECT_ROOT / "w600k_r50.onnx"
        if src_model.exists():
            import shutil
            shutil.copy(str(src_model), str(config.ARCFACE_MODEL_PATH))
            print(f"✓ ArcFace model saved to {config.ARCFACE_MODEL_PATH}")
        zip_path.unlink(missing_ok=True)
        for f in ["w600k_r50.onnx", "1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx"]:
            (config.PROJECT_ROOT / f).unlink(missing_ok=True)
        return config.ARCFACE_MODEL_PATH.exists()
    except Exception as exc:
        print(f"ERROR: {exc}")
        return False


def download_models() -> bool:
    print("=" * 60)
    print(" Model Download")
    print("=" * 60)
    ok_arc = download_arcface()
    try:
        ensure_face_landmarker_model()
        ok_landmark = config.FACE_LANDMARKER_MODEL_PATH.exists()
    except Exception as exc:
        print(f"Face landmarker download failed: {exc}")
        ok_landmark = False
    if ok_arc and ok_landmark:
        print("\n✓ All models ready.")
        return True
    print("\n✗ Some models missing. See errors above.")
    return False


if __name__ == "__main__":
    sys.exit(0 if download_models() else 1)
