"""
Combined Haar + MediaPipe FaceMesh 5-point detector.
Provides robust face detection with landmark confirmation.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import cv2
import numpy as np

from . import config
from .face_mesh import FaceMesh


@dataclass
class FaceDetection:
    """Face detection result with 5 landmarks."""
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    landmarks: np.ndarray  # (5, 2) float32
    full_landmarks: Optional[List[Any]] = None  # 478 mesh points for blink/smile


class HaarMediaPipeFaceDetector:
    """Robust face detector using Haar + MediaPipe FaceMesh."""

    def __init__(self, min_size=config.HAAR_MIN_SIZE):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(cascade_path)

        if self.haar.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

        self.mesh = FaceMesh(
            static_image_mode=config.FACEMESH_STATIC_MODE,
            max_num_faces=config.FACEMESH_MAX_NUM_FACES,
            refine_landmarks=config.FACEMESH_REFINE_LANDMARKS,
            min_detection_confidence=config.FACEMESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACEMESH_MIN_TRACKING_CONFIDENCE,
        )

        self.min_size = min_size

    def _bbox_from_landmarks(self, kps):
        """Build face-like bbox from 5 landmarks with padding."""
        x_min = float(np.min(kps[:, 0]))
        x_max = float(np.max(kps[:, 0]))
        y_min = float(np.min(kps[:, 1]))
        y_max = float(np.max(kps[:, 1]))

        w = max(1.0, x_max - x_min)
        h = max(1.0, y_max - y_min)

        x1 = x_min - config.ALIGNMENT_PAD_X * w
        x2 = x_max + config.ALIGNMENT_PAD_X * w
        y1 = y_min - config.ALIGNMENT_PAD_Y_TOP * h
        y2 = y_max + config.ALIGNMENT_PAD_Y_BOT * h

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _clip_bbox(self, bbox, H, W):
        """Clip bounding box to image boundaries."""
        x1, y1, x2, y2 = bbox.astype(np.float32)
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2))
        y2 = max(0, min(H - 1, y2))
        return int(x1), int(y1), int(x2), int(y2)

    def _validate_landmarks_geometry(self, kps):
        """Sanity check on landmark positions."""
        eye_dist = np.linalg.norm(kps[1] - kps[0])
        if eye_dist < config.MIN_EYE_DISTANCE:
            return False

        if not (kps[3, 1] > kps[2, 1] and kps[4, 1] > kps[2, 1]):
            return False

        return True

    def _detect_on_frame(
        self, frame: np.ndarray, frame_scale: float = 1.0
    ) -> List[FaceDetection]:
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale_back = 1.0 / frame_scale if frame_scale > 0 else 1.0

        # minSize is in pixels of *this* frame; shrink with downscale, not enlarge.
        min_w = max(20, int(self.min_size[0] * frame_scale))
        min_h = max(20, int(self.min_size[1] * frame_scale))
        haar_faces = self.haar.detectMultiScale(
            gray,
            scaleFactor=config.HAAR_SCALE_FACTOR,
            minNeighbors=config.HAAR_MIN_NEIGHBORS,
            minSize=(min_w, min_h),
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)

        if not results.multi_face_landmarks:
            return []

        detected_faces = []
        indices = [
            config.LANDMARK_INDICES["left_eye"],
            config.LANDMARK_INDICES["right_eye"],
            config.LANDMARK_INDICES["nose_tip"],
            config.LANDMARK_INDICES["mouth_left"],
            config.LANDMARK_INDICES["mouth_right"],
        ]

        for face_landmarks in results.multi_face_landmarks:
            landmarks_full = face_landmarks.landmark

            kps = np.array(
                [[lm.x * W, lm.y * H] for lm in (landmarks_full[i] for i in indices)],
                dtype=np.float32,
            )

            if kps[0, 0] > kps[1, 0]:
                kps[[0, 1]] = kps[[1, 0]]
            if kps[3, 0] > kps[4, 0]:
                kps[[3, 4]] = kps[[4, 3]]

            if scale_back != 1.0:
                kps *= scale_back

            if not self._validate_landmarks_geometry(kps):
                continue

            matched_haar = None
            best_overlap = 0

            for x, y, w, h in haar_faces:
                if config.KPS_MUST_BE_IN_HAAR_BOX:
                    bx1, by1 = (x - config.KPS_IN_BOX_MARGIN * w) * scale_back, (y - config.KPS_IN_BOX_MARGIN * h) * scale_back
                    bx2, by2 = (x + (1.0 + config.KPS_IN_BOX_MARGIN) * w) * scale_back, (y + (1.0 + config.KPS_IN_BOX_MARGIN) * h) * scale_back
                    inside = (
                        (kps[:, 0] >= bx1) & (kps[:, 0] <= bx2) &
                        (kps[:, 1] >= by1) & (kps[:, 1] <= by2)
                    )
                    overlap = inside.mean()
                    if overlap > best_overlap and overlap >= config.KPS_IN_BOX_MIN_RATIO:
                        best_overlap = overlap
                        matched_haar = (x, y, w, h)

            if (
                matched_haar is None
                and config.KPS_MUST_BE_IN_HAAR_BOX
                and len(haar_faces) > 0
            ):
                continue

            full_h, full_w = int(H * scale_back), int(W * scale_back)
            bbox = self._bbox_from_landmarks(kps)
            x1, y1, x2, y2 = self._clip_bbox(bbox, full_h, full_w)

            detected_faces.append(
                FaceDetection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    score=1.0,
                    landmarks=kps.astype(np.float32),
                    full_landmarks=landmarks_full,
                )
            )

        return detected_faces

    def detect(self, frame):
        """
        Detect faces in frame. Optionally runs on a downscaled copy for speed.
        """
        scale = config.DETECT_FRAME_SCALE
        if scale < 1.0:
            small = cv2.resize(
                frame, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_LINEAR,
            )
            return self._detect_on_frame(small, frame_scale=scale)
        return self._detect_on_frame(frame, frame_scale=1.0)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.mesh.close()
