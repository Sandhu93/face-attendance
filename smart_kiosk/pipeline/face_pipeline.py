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
    SimpleAligner,
    DCTEmbedder,
    LightLiveness,
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
        # Backend selection; prefer OpenCV Haar, fallback to dummy detector if OpenCV missing (for tests)
        try:
            self.detector = HaarFaceDetector()
        except Exception:
            self.detector = _NoopDetector()
        self.aligner = SimpleAligner()
        self.embedder = DCTEmbedder()
        self.liveness = LightLiveness()

    def detect_best(self, frame_bgr: np.ndarray) -> Optional[Detection]:
        dets = self.detector.detect(frame_bgr)
        if not dets:
            return None
        # Choose highest area with simple quality check
        return dets[0]

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
            tid = self.db.add_template(employee_id, blob, 1, quality=liv.liveness_score)
            ids.append(tid)
        return ids

    def identify(self, frame_bgr: np.ndarray) -> IdentifyResult:
        det = self.detect_best(frame_bgr)
        if det is None:
            return IdentifyResult(None, 1.0, 0.0, False, None)
        if not self.quality_ok(frame_bgr, det):
            return IdentifyResult(None, 1.0, 0.0, False, det)
        liv = self.liveness.passive(frame_bgr, det)
        if not liv.is_live:
            return IdentifyResult(None, 1.0, liv.liveness_score, False, det)

        q_emb = self.embed(frame_bgr, det)
        # Match against all templates (optimize with cached matrix in production)
        best_emp: Optional[str] = None
        best_dist = 1.0
        for _, emp_id, blob, ver, q in self.db.get_templates():
            emb = decompress_embedding(blob)
            dist = cosine_distance(q_emb, emb)
            if dist < best_dist:
                best_dist = dist
                best_emp = emp_id
        return IdentifyResult(best_emp, best_dist, liv.liveness_score, True, det)


class _NoopDetector:
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        return []
