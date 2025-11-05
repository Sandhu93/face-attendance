import os
from typing import Optional
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None  # type: ignore

from .base import LivenessChecker, LivenessResult, Detection


class OnnxAntiSpoof(LivenessChecker):
    def __init__(self, model_path: str, input_size: int = 80):
        if ort is None:
            raise RuntimeError("onnxruntime is required for OnnxAntiSpoof")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, os.cpu_count() - 1 if os.cpu_count() else 1)
        self.session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])  # CPU-only on Pi
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = int(input_size)

    def passive(self, frame_bgr: np.ndarray, det: Detection) -> LivenessResult:
        x, y, w, h = det.bbox
        y0 = max(0, y)
        x0 = max(0, x)
        y1 = min(frame_bgr.shape[0], y + h)
        x1 = min(frame_bgr.shape[1], x + w)
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return LivenessResult(0.0, False, "invalid crop")
        # Preprocess: BGR -> RGB, resize, normalize to [0,1], CHW
        import cv2

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img = (rgb.astype(np.float32) / 255.0)
        # Optionally, mean-std normalization could be applied per model; keep identity for common MiniFASNet)
        inp = np.transpose(img, (2, 0, 1))[None, :, :, :]
        out = self.session.run(None, {self.input_name: inp})
        logits = out[0]
        # Expect shape [1, 2] for [spoof, live]
        if logits.ndim == 2 and logits.shape[1] >= 2:
            v = logits[0]
            # softmax
            e = np.exp(v - np.max(v))
            probs = e / (np.sum(e) + 1e-8)
            live_score = float(probs[1])
        else:
            # If single logit, apply sigmoid
            v = float(logits.ravel()[0])
            live_score = float(1.0 / (1.0 + np.exp(-v)))
        is_live = live_score >= 0.6
        return LivenessResult(live_score, is_live, "onnx-antispoof")

