"""
Face detection module using Haar Cascades.
Tests face bounding box generation.
"""

import sys
import cv2
from pathlib import Path


def main():
    """
    Detect faces using Haar cascade and draw bounding boxes.
    Press 'q' to exit.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print(f"ERROR: Failed to load cascade from {cascade_path}")
        return False
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return False
    
    print("✓ Haar face detector initialized.")
    print("  Detecting faces... Press 'q' to exit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )
            
            vis = frame.copy()
            
            # Draw bounding boxes
            for (x, y, w, h) in faces:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Status
            status = f"Faces detected: {len(faces)}"
            cv2.putText(
                vis, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            cv2.putText(
                vis, "Press 'q' to quit", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
            )
            
            cv2.imshow("Face Detection", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("✓ Face detection test complete.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
