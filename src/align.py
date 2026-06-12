"""
Face alignment using 5-point landmarks.
Warps detected faces into canonical 112x112 pose using similarity transform.
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path

from . import config
from .face_mesh import FaceMesh


class FaceAligner:
    """Aligns faces using 5-point landmarks to canonical pose."""
    
    def __init__(self, out_size=config.ALIGNMENT_OUTPUT_SIZE):
        self.out_size = out_size
        self.out_w, self.out_h = int(out_size[0]), int(out_size[1])
        
        # ArcFace standard template (InsightFace)
        self.template = np.array([
            [38.2946, 51.6963],   # left eye
            [73.5318, 51.5014],   # right eye
            [56.0252, 71.7366],   # nose
            [41.5493, 92.3655],   # left mouth
            [70.7299, 92.2041],   # right mouth
        ], dtype=np.float32)
        
        # Scale template if using different output size
        if (self.out_w, self.out_h) != (112, 112):
            sx = self.out_w / 112.0
            sy = self.out_h / 112.0
            self.template = self.template * np.array([sx, sy], dtype=np.float32)
    
    def align(self, frame, landmarks):
        """
        Align face using 5 landmarks.
        
        Args:
            frame: BGR image
            landmarks: (5, 2) array of [x, y] coordinates
        
        Returns:
            aligned: Warped 112x112 BGR image
            M: Affine transformation matrix
        """
        landmarks = landmarks.astype(np.float32)
        
        # Estimate similarity transform
        M, _ = cv2.estimateAffinePartial2D(
            landmarks, self.template,
            method=cv2.LMEDS
        )
        
        if M is None:
            # Fallback to 3-point transform
            M = cv2.getAffineTransform(
                landmarks[:3], self.template[:3]
            )
        
        # Warp
        aligned = cv2.warpAffine(
            frame, M, (self.out_w, self.out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        return aligned, M


def main():
    """
    Test face alignment pipeline.
    Shows original + aligned faces side by side.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print(f"ERROR: Failed to load cascade.")
        return False
    
    mp_face_mesh = FaceMesh(
        static_image_mode=config.FACEMESH_STATIC_MODE,
        max_num_faces=1,
        refine_landmarks=config.FACEMESH_REFINE_LANDMARKS,
        min_detection_confidence=config.FACEMESH_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.FACEMESH_MIN_TRACKING_CONFIDENCE,
    )
    
    aligner = FaceAligner()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return False
    
    idx_left_eye = config.LANDMARK_INDICES["left_eye"]
    idx_right_eye = config.LANDMARK_INDICES["right_eye"]
    idx_nose = config.LANDMARK_INDICES["nose_tip"]
    idx_mouth_left = config.LANDMARK_INDICES["mouth_left"]
    idx_mouth_right = config.LANDMARK_INDICES["mouth_right"]
    
    print("✓ Face alignment initialized.")
    print("  Press 'q' to quit, 's' to save aligned crop.")
    
    config.ensure_dirs()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            H, W = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            
            vis = frame.copy()
            aligned_vis = np.zeros((config.ALIGNMENT_OUTPUT_SIZE[1], 
                                   config.ALIGNMENT_OUTPUT_SIZE[0], 3), dtype=np.uint8)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_face_mesh.process(rgb)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    pts = []
                    indices = [idx_left_eye, idx_right_eye, idx_nose, 
                              idx_mouth_left, idx_mouth_right]
                    for idx in indices:
                        lm = landmarks[idx]
                        pts.append([lm.x * W, lm.y * H])
                    
                    kps = np.array(pts, dtype=np.float32)
                    
                    # Enforce ordering
                    if kps[0, 0] > kps[1, 0]:
                        kps[[0, 1]] = kps[[1, 0]]
                    if kps[3, 0] > kps[4, 0]:
                        kps[[3, 4]] = kps[[4, 3]]
                    
                    # Draw landmarks on original
                    for (px, py) in kps.astype(int):
                        cv2.circle(vis, (int(px), int(py)), 3, (0, 255, 0), -1)
                    
                    # Align
                    aligned_vis, _ = aligner.align(frame, kps)
                    
                    cv2.putText(
                        vis, "Alignment successful", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )
                else:
                    cv2.putText(
                        vis, "Landmarks not found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                    )
            else:
                cv2.putText(
                    vis, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )
            
            cv2.putText(
                vis, "q=quit, s=save", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
            )
            
            cv2.imshow("Original", vis)
            cv2.imshow("Aligned 112x112", aligned_vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                ts = int(time.time() * 1000)
                path = config.DEBUG_ALIGNED_DIR / f"{ts}.jpg"
                cv2.imwrite(str(path), aligned_vis)
                print(f"Saved: {path}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("✓ Alignment test complete.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
