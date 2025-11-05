import os
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - optional dep
    ort = None  # type: ignore

from .base import FaceEmbedder


class ONNXArcFaceEmbedder(FaceEmbedder):
    """
    ArcFace-like ONNX embedder.
    - Expects aligned RGB 112x112 input in [0,1] with mean/std normalization.
    - Produces 512-dim L2-normalized embedding.

    Place the ONNX model in smart_kiosk/models/assets/arcface_r100.onnx or
    set SMART_KIOSK_ARCFACE_PATH env var to an absolute path.
    """

    dim = 512

    def __init__(self, model_dir: Optional[str] = None, model_path: Optional[str] = None):
        if ort is None:
            raise RuntimeError("onnxruntime is required for ONNXArcFaceEmbedder")
        # Resolve model path
        env_override = os.getenv("SMART_KIOSK_ARCFACE_PATH")
        if model_path is None:
            if env_override:
                model_path = env_override
            else:
                base = model_dir or os.path.join(os.path.dirname(__file__), "assets")
                model_path = os.path.join(base, "arcface_r100.onnx")
        if not os.path.exists(model_path):
            raise RuntimeError(f"ArcFace ONNX model not found at {model_path}")

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
        providers = [
            ("CUDAExecutionProvider", {}),
            ("CPUExecutionProvider", {}),
        ]
        try:
            self.sess = ort.InferenceSession(model_path, sess_options=sess_opts, providers=[p[0] for p in providers])
        except Exception:
            # Fallback to CPU only
            self.sess = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

        # Cache IO names
        self.inp_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def embed(self, aligned_rgb: np.ndarray) -> np.ndarray:
        # Ensure HWC RGB, 112x112
        img = aligned_rgb
        if img is None or img.size == 0:
            # Degenerate safe vector
            v = np.zeros((self.dim,), dtype=np.float32)
            v[0] = 1.0
            return v
        h, w = img.shape[:2]
        if (h, w) != (112, 112):
            # Resize with naive NN to avoid cv2 dependency here
            img = _resize_nn(img, (112, 112))
        img = img.astype(np.float32) / 255.0
        # Common ArcFace normalization
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        img = (img - mean) / std
        # HWC -> NCHW
        chw = np.transpose(img, (2, 0, 1))[None, ...]

        out = self.sess.run([self.out_name], {self.inp_name: chw})[0]
        vec = out.reshape(-1).astype(np.float32)
        # L2 normalize and guard zeros
        n = float(np.linalg.norm(vec))
        if not np.isfinite(n) or n < 1e-6:
            v = np.zeros((self.dim,), dtype=np.float32)
            v[0] = 1.0
            return v
        return vec / n


def _resize_nn(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    new_w, new_h = size
    y_idx = (np.linspace(0, h - 1, new_h)).astype(int)
    x_idx = (np.linspace(0, w - 1, new_w)).astype(int)
    if img.ndim == 2:
        return img[y_idx][:, x_idx]
    return img[y_idx][:, x_idx, :]

