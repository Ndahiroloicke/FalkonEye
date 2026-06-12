"""
MediaPipe face landmark detection with backward-compatible FaceMesh API.

MediaPipe >=0.10.30 removed mp.solutions; this module uses the Tasks API
(FaceLandmarker) and exposes the legacy .process() / .multi_face_landmarks interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import mediapipe as mp
from mediapipe.tasks.python import vision

from . import config


FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class _Landmark:
    x: float
    y: float
    z: float = 0.0


@dataclass
class _FaceLandmarks:
    landmark: List[_Landmark]


@dataclass
class _FaceMeshResults:
    multi_face_landmarks: Optional[List[_FaceLandmarks]]


def ensure_face_landmarker_model() -> None:
    """Download face_landmarker.task if missing."""
    model_path = config.FACE_LANDMARKER_MODEL_PATH
    if model_path.exists():
        return

    config.ensure_dirs()
    print(f"Downloading face landmarker model to {model_path} ...")

    try:
        import urllib.request

        urllib.request.urlretrieve(FACE_LANDMARKER_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download face landmarker model.\n"
            f"Download manually from:\n  {FACE_LANDMARKER_URL}\n"
            f"Save to: {model_path}"
        ) from exc

    print("✓ Face landmarker model downloaded.")


class FaceMesh:
    """Drop-in replacement for mp.solutions.face_mesh.FaceMesh."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        del refine_landmarks  # handled by bundled model

        ensure_face_landmarker_model()
        if not config.FACE_LANDMARKER_MODEL_PATH.exists():
            raise RuntimeError(
                f"Face landmarker model not found: {config.FACE_LANDMARKER_MODEL_PATH}\n"
                "Run: python download_model.py"
            )

        running_mode = (
            vision.RunningMode.IMAGE
            if static_image_mode
            else vision.RunningMode.VIDEO
        )

        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(config.FACE_LANDMARKER_MODEL_PATH)
            ),
            running_mode=running_mode,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._video_mode = running_mode == vision.RunningMode.VIDEO
        self._frame_timestamp_ms = 0

    def process(self, rgb_image) -> _FaceMeshResults:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        if self._video_mode:
            self._frame_timestamp_ms += 33
            result = self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return _FaceMeshResults(multi_face_landmarks=None)

        faces = []
        for face in result.face_landmarks:
            faces.append(
                _FaceLandmarks(
                    landmark=[_Landmark(lm.x, lm.y, getattr(lm, "z", 0.0)) for lm in face]
                )
            )
        return _FaceMeshResults(multi_face_landmarks=faces)

    def close(self) -> None:
        self._landmarker.close()
