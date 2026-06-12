"""
5-point facial landmark detection using MediaPipe FaceMesh.
Extracts: left eye, right eye, nose tip, left mouth corner, right mouth corner.
"""

import sys
import cv2
import numpy as np

from . import config
from .face_mesh import FaceMesh


def main():
    """
    Detect 5 facial landmarks using MediaPipe FaceMesh.
    Requires face to be detected by Haar first.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print(f"ERROR: Failed to load cascade from {cascade_path}")
        return False
    
    # Initialize FaceMesh
    mp_face_mesh = FaceMesh(
        static_image_mode=config.FACEMESH_STATIC_MODE,
        max_num_faces=config.FACEMESH_MAX_NUM_FACES,
        refine_landmarks=config.FACEMESH_REFINE_LANDMARKS,
        min_detection_confidence=config.FACEMESH_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.FACEMESH_MIN_TRACKING_CONFIDENCE,
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return False
    
    # 5-point indices
    idx_left_eye = config.LANDMARK_INDICES["left_eye"]
    idx_right_eye = config.LANDMARK_INDICES["right_eye"]
    idx_nose = config.LANDMARK_INDICES["nose_tip"]
    idx_mouth_left = config.LANDMARK_INDICES["mouth_left"]
    idx_mouth_right = config.LANDMARK_INDICES["mouth_right"]
    
    print("✓ 5-point landmark detector initialized.")
    print("  Detecting landmarks... Press 'q' to exit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            H, W = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            
            vis = frame.copy()
            
            # Draw face bounding boxes
            for (x, y, w, h) in faces:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Run FaceMesh on full frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Extract 5 points
                pts = []
                indices = [idx_left_eye, idx_right_eye, idx_nose, idx_mouth_left, idx_mouth_right]
                for idx in indices:
                    lm = landmarks[idx]
                    x = int(lm.x * W)
                    y = int(lm.y * H)
                    pts.append([x, y])
                    cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)
                
                # Ensure left/right ordering
                kps = np.array(pts, dtype=np.float32)
                if kps[0, 0] > kps[1, 0]:  # Swap eyes if needed
                    kps[[0, 1]] = kps[[1, 0]]
                if kps[3, 0] > kps[4, 0]:  # Swap mouth corners if needed
                    kps[[3, 4]] = kps[[4, 3]]
                
                cv2.putText(
                    vis, "5-point landmarks detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
            else:
                cv2.putText(
                    vis, "No landmarks detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
            
            cv2.putText(
                vis, "Press 'q' to quit", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
            )
            
            cv2.imshow("5-Point Landmarks", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("✓ Landmark detection test complete.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
