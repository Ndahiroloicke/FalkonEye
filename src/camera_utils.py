#!/usr/bin/env python3
"""
Camera detection and resilient capture for USB webcams.

Linux: prefers V4L2. Windows: prefers DirectShow, then Media Foundation.
CameraStream retries reads and reconnects instead of killing the loop.
"""

import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config

_IS_WINDOWS = sys.platform == "win32"


def _capture_backends() -> Tuple[int, ...]:
    """Platform-ordered OpenCV backends to try when opening a camera."""
    if _IS_WINDOWS:
        return (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
    return (cv2.CAP_V4L2, cv2.CAP_ANY)


def get_available_devices() -> List[Tuple[int, str]]:
    """List camera indices that can deliver a frame."""
    devices: List[Tuple[int, str]] = []
    if _IS_WINDOWS:
        for idx in range(10):
            cap = _open_capture(idx, quiet=True)
            if cap is not None:
                cap.release()
                devices.append((idx, f"Camera {idx} (USB/local)"))
        return devices

    for i in range(10):
        dev_path = f"/dev/video{i}"
        if os.path.exists(dev_path):
            devices.append((i, dev_path))
    return devices


def find_working_camera(max_index: int = 10, quiet: bool = False) -> int:
    """
    Return the first camera index that can open and deliver a valid frame.
    Returns -1 if none found.
    """
    if not quiet:
        print("Searching for working camera...")
    for idx in range(max_index):
        cap = _open_capture(idx)
        if cap is not None:
            cap.release()
            if not quiet:
                print(f"OK: Found working camera at index {idx}")
            return idx
    return -1


def _open_capture(index: int, quiet: bool = False) -> Optional[cv2.VideoCapture]:
    """Try platform backends; verify with a real frame read."""
    for backend in _capture_backends():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if config.CAMERA_FPS_TARGET:
            cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS_TARGET)
        # USB webcams on Windows often need a short warm-up before the first frame.
        time.sleep(0.15 if _IS_WINDOWS else 0.05)
        for _ in range(config.CAMERA_OPEN_VERIFY_FRAMES):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                if not quiet:
                    backend_name = {
                        cv2.CAP_DSHOW: "DirectShow",
                        cv2.CAP_MSMF: "Media Foundation",
                        cv2.CAP_V4L2: "V4L2",
                    }.get(backend, str(backend))
                    print(f"  backend: {backend_name}")
                return cap
            time.sleep(0.08 if _IS_WINDOWS else 0.05)
        cap.release()
    return None


class CameraStream:
    """Resilient wrapper around cv2.VideoCapture with reconnect support."""

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.index: int = config.CAMERA_INDEX
        self._fail_streak = 0

    def open(self) -> bool:
        indices = self._candidate_indices()
        for idx in indices:
            cap = _open_capture(idx)
            if cap is None:
                continue
            self.cap = cap
            self.index = idx
            if idx != config.CAMERA_INDEX:
                print(
                    f"OK: Using camera index {idx} "
                    f"(index {config.CAMERA_INDEX} unavailable — update CAMERA_INDEX in config.py)"
                )
            else:
                print(f"OK: Camera opened on index {idx}")
            self._fail_streak = 0
            return True
        return False

    def _candidate_indices(self) -> List[int]:
        ordered: List[int] = []
        if config.CAMERA_INDEX not in ordered:
            ordered.append(config.CAMERA_INDEX)
        if config.CAMERA_AUTO_DETECT:
            for idx in range(10):
                if idx not in ordered:
                    ordered.append(idx)
        return ordered

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame; retry and reconnect on transient V4L2 failures."""
        if self.cap is None or not self.cap.isOpened():
            if not self._reconnect():
                return False, None

        for _ in range(config.CAMERA_READ_RETRIES):
            try:
                ret, frame = self.cap.read()
            except cv2.error:
                ret, frame = False, None
            if ret and frame is not None and frame.size > 0:
                self._fail_streak = 0
                return True, frame
            time.sleep(0.02)

        self._fail_streak += 1
        if self._fail_streak >= config.CAMERA_RECONNECT_AFTER_FAILS:
            print(
                f"WARN: Camera read failed {self._fail_streak} times - reconnecting "
                f"(index {self.index})..."
            )
            if self._reconnect():
                self._fail_streak = 0
                try:
                    ret, frame = self.cap.read()
                except cv2.error:
                    ret, frame = False, None
                if ret and frame is not None and frame.size > 0:
                    return True, frame
        return False, None

    def _reconnect(self) -> bool:
        self.release(silent=True)
        time.sleep(config.CAMERA_RECONNECT_DELAY_SEC)
        return self.open()

    def release(self, silent: bool = False) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except cv2.error:
                pass
            self.cap = None

    @staticmethod
    def is_window_open(window_name: str) -> bool:
        try:
            return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False


if __name__ == "__main__":
    platform = "Windows" if _IS_WINDOWS else "Linux"
    print(f"Camera scan ({platform})")
    devices = get_available_devices()
    if devices:
        print("Working cameras:")
        for idx, label in devices:
            print(f"  index {idx}: {label}")
    else:
        print("No working cameras found yet.")
    print("\nTesting preferred index from config...")
    working_idx = find_working_camera()
    if working_idx >= 0:
        print(f"\nOK: Set CAMERA_INDEX = {working_idx} in src/config.py")
    else:
        print("\nERROR: No working camera found.")
        if _IS_WINDOWS:
            print("  - Close other apps using the webcam (Teams, Zoom, browser).")
            print("  - Replug the USB camera and run this script again.")
