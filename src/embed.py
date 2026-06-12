"""
Face embedding extraction using ArcFace ONNX model.
Converts aligned 112x112 faces to 512-dimensional L2-normalized embeddings.
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed. Run: pip install onnxruntime")
    sys.exit(1)

from . import config
from .face_mesh import FaceMesh
from .align import FaceAligner


class ArcFaceEmbedder:
    """ArcFace ONNX embedder for face embedding extraction."""
    
    def __init__(self, model_path=config.ARCFACE_MODEL_PATH):
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Please download ArcFace ONNX model (see README)."
            )
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = config.ONNX_INTRA_OP_THREADS
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=[config.ONNX_EXECUTION_PROVIDER],
        )
        
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.output_name = self.session.get_outputs()[0].name
        self._input_layout = self._infer_input_layout(input_meta.shape)

    @staticmethod
    def _infer_input_layout(shape) -> str:
        """Detect NCHW vs NHWC from ONNX input dimensions."""
        if len(shape) != 4:
            return "NCHW"
        _, d1, d2, d3 = shape
        if d3 in (1, 3):
            return "NHWC"
        if d1 in (1, 3):
            return "NCHW"
        return "NCHW"

    def _preprocess(self, aligned_bgr):
        """Convert 112x112 BGR to normalized float32 batch tensor."""
        if aligned_bgr.shape[:2] != config.EMBEDDING_INPUT_SIZE:
            aligned_bgr = cv2.resize(
                aligned_bgr, config.EMBEDDING_INPUT_SIZE,
                interpolation=cv2.INTER_LINEAR
            )

        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - config.EMBEDDING_PREPROCESS_MEAN) / config.EMBEDDING_PREPROCESS_SCALE

        if self._input_layout == "NHWC":
            x = rgb[np.newaxis, ...]
        else:
            x = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

        return x.astype(np.float32)
    
    def _l2_normalize(self, v):
        """L2 normalize vector to unit length."""
        v = v.astype(np.float32).reshape(-1)
        norm = np.linalg.norm(v) + config.EMBEDDING_NORM_EPSILON
        return v / norm
    
    def embed(self, aligned_bgr):
        """
        Extract embedding from aligned face image.
        
        Args:
            aligned_bgr: 112x112 BGR image
        
        Returns:
            embedding: (512,) L2-normalized float32 vector
            norm_before: Raw norm before L2 normalization
        """
        x = self._preprocess(aligned_bgr)
        y = self.session.run([self.output_name], {self.input_name: x})[0]
        
        v = y.reshape(-1).astype(np.float32)
        norm_before = np.linalg.norm(v)
        
        v_normalized = self._l2_normalize(v)

        return v_normalized, norm_before

    def embed_batch(self, aligned_images: list) -> np.ndarray:
        """Embed multiple faces; vectorized matching happens in process_faces."""
        if not aligned_images:
            return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)
        return np.stack(
            [self.embed(img)[0] for img in aligned_images],
            axis=0,
        ).astype(np.float32)


def main():
    """Test embedding extraction from live camera."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("ERROR: Failed to load Haar cascade.")
        return False
    
    mp_face_mesh = FaceMesh(
        static_image_mode=config.FACEMESH_STATIC_MODE,
        max_num_faces=1,
        refine_landmarks=config.FACEMESH_REFINE_LANDMARKS,
        min_detection_confidence=config.FACEMESH_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.FACEMESH_MIN_TRACKING_CONFIDENCE,
    )
    
    try:
        embedder = ArcFaceEmbedder(config.ARCFACE_MODEL_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nTo download ArcFace model:")
        print("  1. curl -L -o buffalo_l.zip https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download")
        print("  2. unzip -o buffalo_l.zip")
        print("  3. cp w600k_r50.onnx models/embedder_arcface.onnx")
        return False
    
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
    
    print("✓ ArcFace embedder initialized.")
    print("  Press 'q' to quit, 'p' to print embedding stats.")
    
    prev_emb = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            H, W = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            
            vis = frame.copy()
            
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
                    
                    if kps[0, 0] > kps[1, 0]:
                        kps[[0, 1]] = kps[[1, 0]]
                    if kps[3, 0] > kps[4, 0]:
                        kps[[3, 4]] = kps[[4, 3]]
                    
                    for (px, py) in kps.astype(int):
                        cv2.circle(vis, (int(px), int(py)), 3, (0, 255, 0), -1)
                    
                    aligned, _ = aligner.align(frame, kps)
                    emb, norm_before = embedder.embed(aligned)
                    
                    info = [
                        f"embedding dim: {emb.size}",
                        f"norm(before L2): {norm_before:.2f}",
                    ]
                    
                    if prev_emb is not None:
                        sim = float(np.dot(prev_emb, emb))
                        info.append(f"cos(prev,this): {sim:.3f}")
                    
                    prev_emb = emb
                    
                    y_pos = 30
                    for line in info:
                        cv2.putText(
                            vis, line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                        )
                        y_pos += 25
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
                vis, "q=quit, p=print", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
            )
            
            cv2.imshow("Embedding Extraction", vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p") and prev_emb is not None:
                print("\nEmbedding Statistics:")
                print(f"  Shape: {prev_emb.shape}")
                print(f"  Norm: {np.linalg.norm(prev_emb):.6f}")
                print(f"  Mean: {prev_emb.mean():.6f}")
                print(f"  Std: {prev_emb.std():.6f}")
                print(f"  First 10: {prev_emb[:10]}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print("✓ Embedding extraction test complete.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
