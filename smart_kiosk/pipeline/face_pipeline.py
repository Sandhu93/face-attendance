import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from smart_kiosk.app.config import AppConfig
from smart_kiosk.app.utils import compress_embedding, decompress_embedding, cosine_distance
from smart_kiosk.db import Database
from smart_kiosk.models import (
    Detection,
    HaarFaceDetector,
    DNNFaceDetector,
    SimpleAligner,
    DCTEmbedder,
    LightLiveness,
    ONNXArcFaceEmbedder,
)


@dataclass
class IdentifyResult:
    employee_id: Optional[str]
    score: float
    liveness: float
    is_live: bool
    det: Optional[Detection]


class FacePipeline:
    def __init__(self, cfg: AppConfig, db: Database):
        self.cfg = cfg
        self.db = db
        # Backend selection; prefer OpenCV DNN if configured and model files exist
        self.detector = None
        if getattr(self.cfg.backend, "face_backend", "opencv").lower() in ("opencv_dnn", "dnn"):
            assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "assets"))
            try:
                self.detector = DNNFaceDetector(model_dir=assets)
            except Exception:
                self.detector = None
        if self.detector is None:
            try:
                self.detector = HaarFaceDetector()
            except Exception:
                self.detector = _NoopDetector()
        self.aligner = SimpleAligner()
        # Embedder selection
        self.embedder = None
        backend = getattr(self.cfg.backend, "embedder_backend", "dct").lower()
        if backend in ("onnx", "arcface", "onnx_arcface") and ONNXArcFaceEmbedder is not None:
            try:
                assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "assets"))
                self.embedder = ONNXArcFaceEmbedder(model_dir=assets)
            except Exception:
                self.embedder = None
        if self.embedder is None:
            self.embedder = DCTEmbedder()
        self.liveness = LightLiveness()

    def _embedder_version(self) -> int:
        # Version tag to distinguish template types
        # 1 = DCTEmbedder (fallback), 2 = ONNX ArcFace
        if self.embedder.__class__.__name__.lower().startswith("onnxarcface"):
            return 2
        return 1

    def embedder_version(self) -> int:
        return self._embedder_version()

    def detect_best(self, frame_bgr: np.ndarray) -> Optional[Detection]:
        dets = self.detector.detect(frame_bgr)
        if not dets:
            return None
        return dets[0]

    def detect_all(self, frame_bgr: np.ndarray) -> List[Detection]:
        return self.detector.detect(frame_bgr)

    def quality_ok(self, frame_bgr: np.ndarray, det: Detection) -> bool:
        x, y, w, h = det.bbox
        if min(w, h) < self.cfg.thresholds.min_face_px:
            return False
        face = frame_bgr[max(0, y): y + h, max(0, x): x + w]
        gray = (0.114 * face[..., 0] + 0.587 * face[..., 1] + 0.299 * face[..., 2]).astype(np.float32)
        pad = np.pad(gray, 1, mode="edge")
        lap = pad[1:-1, 2:] + pad[1:-1, :-2] + pad[2:, 1:-1] + pad[:-2, 1:-1] - 4 * pad[1:-1, 1:-1]
        sharp = float(lap.var())
        return sharp >= self.cfg.thresholds.quality_sharpness_min

    def embed(self, frame_bgr: np.ndarray, det: Detection) -> np.ndarray:
        aligned = self.aligner.align(frame_bgr, det)
        return self.embedder.embed(aligned)

    def enroll_templates(self, employee_id: str, frames_bgr: List[np.ndarray]) -> List[str]:
        ids: List[str] = []
        ver = self._embedder_version()
        for frm in frames_bgr:
            det = self.detect_best(frm)
            if not det:
                continue
            if not self.quality_ok(frm, det):
                continue
            liv = self.liveness.passive(frm, det)
            if not liv.is_live:
                continue
            emb = self.embed(frm, det)
            blob = compress_embedding(emb)
            tid = self.db.add_template(employee_id, blob, ver, quality=liv.liveness_score)
            ids.append(tid)
        return ids

    def identify(self, frame_bgr: np.ndarray) -> IdentifyResult:
        det = self.detect_best(frame_bgr)
        if det is None:
            return IdentifyResult(None, 1.0, 0.0, False, None)
        return self.identify_det(frame_bgr, det)

    def identify_det(self, frame_bgr: np.ndarray, det: Detection) -> IdentifyResult:
        if not self.quality_ok(frame_bgr, det):
            return IdentifyResult(None, 1.0, 0.0, False, det)
        liv = self.liveness.passive(frame_bgr, det)
        if not liv.is_live:
            return IdentifyResult(None, 1.0, liv.liveness_score, False, det)

        q_emb = self.embed(frame_bgr, det)
        # Group templates per employee and compute min/mean distances
        per_emp: Dict[str, Dict[str, float]] = {}
        cur_ver = self._embedder_version()
        for _, emp_id, blob, ver, q in self.db.get_templates():
            # Skip templates from other embedder versions to avoid mixing spaces
            if int(ver) != int(cur_ver):
                continue
            try:
                emb = decompress_embedding(blob)
                # Guard shape mismatch
                if emb.ndim != 1:
                    continue
                if emb.size != q_emb.size:
                    # Skip incompatible templates (e.g., older embedder version)
                    continue
                dist = float(cosine_distance(q_emb, emb))
                if not np.isfinite(dist):
                    continue
            except Exception:
                continue
            d = per_emp.setdefault(emp_id, {"min": 1.0, "sum": 0.0, "cnt": 0.0})
            if dist < d["min"]:
                d["min"] = dist
            d["sum"] += dist
            d["cnt"] += 1.0

        best_emp: Optional[str] = None
        best_min = 1.0
        best_mean = 1.0
        for emp_id, stats in per_emp.items():
            cnt = max(1.0, stats["cnt"])
            mean = stats["sum"] / cnt
            mn = stats["min"]
            # Primary by min distance, tie-break by mean, then by employee_id
            if (mn + 1e-6) < best_min or (abs(mn - best_min) <= 1e-6 and mean < best_mean) or (
                abs(mn - best_min) <= 1e-6 and abs(mean - best_mean) <= 1e-6 and (best_emp is None or emp_id < best_emp)
            ):
                best_emp = emp_id
                best_min = mn
                best_mean = mean

        # Apply threshold to mark unknowns
        thr = float(self.cfg.thresholds.match_threshold)
        if best_emp is None or not np.isfinite(best_min) or best_min > thr:
            return IdentifyResult(None, float(best_min), liv.liveness_score, True, det)
        return IdentifyResult(best_emp, float(best_min), liv.liveness_score, True, det)


class _NoopDetector:
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        return []
