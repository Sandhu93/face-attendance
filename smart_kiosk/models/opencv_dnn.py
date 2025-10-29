import os
from typing import List
import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None  # type: ignore

from .base import Detection, FaceDetector


class DnnFaceDetector(FaceDetector):
    def __init__(self, prototxt_path: str, weights_path: str, conf_threshold: float = 0.5):
        if cv2 is None:
            raise RuntimeError("OpenCV not available")
        if not (os.path.exists(prototxt_path) and os.path.exists(weights_path)):
            raise FileNotFoundError("DNN face model files not found")
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        (h, w) = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        dets: List[Detection] = []
        for i in range(0, detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.conf_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x0, y0, x1, y1) = box.astype("int")
            x = max(0, int(x0))
            y = max(0, int(y0))
            x1 = min(w - 1, int(x1))
            y1 = min(h - 1, int(y1))
            dets.append(Detection((x, y, max(0, x1 - x), max(0, y1 - y)), conf))
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets

