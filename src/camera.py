"""
Camera module: validates video capture and FPS.
Cross-platform USB / built-in webcam test.
"""

import sys
import time
import cv2

from . import config
from .camera_utils import _open_capture


def main():
    """
    Open webcam and display live video with FPS counter.
    Press 'q' to exit.
    """
    cap = _open_capture(config.CAMERA_INDEX)
    if cap is None and config.CAMERA_AUTO_DETECT:
        for idx in range(1, 10):
            cap = _open_capture(idx, quiet=True)
            if cap is not None:
                print(f"✓ Using camera index {idx}")
                break

    if cap is None:
        print(f"ERROR: Cannot open camera (tried index {config.CAMERA_INDEX} and auto-detect)")
        print("Run: python -m src.camera_utils")
        return False

    print("✓ Camera opened successfully")
    print(f"  Resolution target: {config.CAMERA_FRAME_WIDTH}x{config.CAMERA_FRAME_HEIGHT}")
    print("  Press 'q' to exit")

    cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)

    frame_count = 0
    fps = 0.0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from camera")
                break

            frame_count += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t0 = time.time()

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Press 'q' to quit",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("✓ Camera test passed. Pipeline ready.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
