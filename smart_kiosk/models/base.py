from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    score: float


@dataclass
class LivenessResult:
    liveness_score: float
    is_live: bool
    explanation: str = ""


class FaceDetector:
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class FaceAligner:
    def align(self, frame_bgr: np.ndarray, det: Detection) -> np.ndarray:
        raise NotImplementedError


class FaceEmbedder:
    dim: int = 512

    def embed(self, aligned_rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LivenessChecker:
    def passive(self, frame_bgr: np.ndarray, det: Detection) -> LivenessResult:
        raise NotImplementedError

    def active_prompt(self) -> str:
        return "blink twice"

    def active_validate(self, frames_bgr: List[np.ndarray], dets: List[Detection], prompt: str) -> LivenessResult:
        return LivenessResult(0.5, False, "active liveness not implemented")

