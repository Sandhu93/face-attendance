import os
import numpy as np
from typing import List, Tuple, Optional
from .base import Detection, FaceDetector, FaceAligner, FaceEmbedder, LivenessChecker, LivenessResult


class HaarFaceDetector(FaceDetector):
    def __init__(self):
        try:
            import cv2  # type: ignore

            self._cv2 = cv2
            base = cv2.data.haarcascades
            names = [
                "haarcascade_frontalface_default.xml",
                "haarcascade_frontalface_alt2.xml",
                "haarcascade_profileface.xml",
            ]
            self.cascades = []
            for n in names:
                path = base + n
                if os.path.exists(path):
                    self.cascades.append(cv2.CascadeClassifier(path))
            if not self.cascades:
                raise RuntimeError("No Haar cascades found")
        except Exception as e:
            raise RuntimeError("OpenCV is required for HaarFaceDetector") from e

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        cv2 = self._cv2
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Normalize contrast for tougher lighting
        try:
            gray = cv2.equalizeHist(gray)
        except Exception:
            pass
        params = dict(scaleFactor=1.05, minNeighbors=3, minSize=(40, 40))
        dets: List[Detection] = []
        H, W = gray.shape[:2]
        for cas in self.cascades:
            faces = cas.detectMultiScale(gray, **params)
            for (x, y, w, h) in faces:
                score = float(min(1.0, (w * h) / (H * W) * 10.0))
                dets.append(Detection((int(x), int(y), int(w), int(h)), score))
        # Fallback: also try mirrored image to catch slight yaw
        if not dets:
            try:
                gray_flip = cv2.flip(gray, 1)
                for cas in self.cascades:
                    faces = cas.detectMultiScale(gray_flip, **params)
                    for (x, y, w, h) in faces:
                        x_m = W - (x + w)
                        score = float(min(1.0, (w * h) / (H * W) * 10.0))
                        dets.append(Detection((int(x_m), int(y), int(w), int(h)), score))
            except Exception:
                pass
        # Final fallback: simple skin-color blob finder to provide a usable ROI
        if not dets:
            fb = self._skin_fallback(frame_bgr)
            if fb is not None:
                dets.append(fb)
        dets.sort(key=lambda d: d.bbox[2] * d.bbox[3], reverse=True)
        return dets

    def _skin_fallback(self, frame_bgr: np.ndarray) -> Optional[Detection]:
        cv2 = self._cv2
        try:
            ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            Y, Cr, Cb = cv2.split(ycrcb)
            # Broad skin-color range in YCrCb
            mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
            # Smooth and morph to reduce noise
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None
            cnt = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 60 or h < 60:
                return None
            ar = w / (h + 1e-6)
            if ar < 0.6 or ar > 1.8:
                return None
            H, W = frame_bgr.shape[:2]
            score = float(min(1.0, (w * h) / (H * W) * 10.0))
            return Detection((int(x), int(y), int(w), int(h)), score)
        except Exception:
            return None


class SimpleAligner(FaceAligner):
    def align(self, frame_bgr: np.ndarray, det: Detection) -> np.ndarray:
        x, y, w, h = det.bbox
        pad = int(0.2 * w)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(frame_bgr.shape[1], x + w + pad)
        y1 = min(frame_bgr.shape[0], y + h + pad)
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            crop = frame_bgr[max(0, y):y + h, max(0, x):x + w]
        # Convert BGR to RGB without cv2
        rgb = crop[..., ::-1].copy()
        try:
            import cv2  # type: ignore

            aligned = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        except Exception:
            aligned = _resize_nn(rgb, (112, 112))
        return aligned


class DCTEmbedder(FaceEmbedder):
    dim = 512

    def embed(self, aligned_rgb: np.ndarray) -> np.ndarray:
        # Use luminance channel, approximate DCT via FFT magnitude top-left block (cv2-free)
        gray = _rgb_to_gray(aligned_rgb)
        gray = _resize_nn(gray, (32, 32)).astype(np.float32) / 255.0
        f = np.fft.fft2(gray)
        mag = np.abs(f)
        block = mag[:24, :24].flatten()
        if block.size < self.dim:
            # Pad with zeros
            block = np.pad(block, (0, self.dim - block.size), mode="constant")
        else:
            block = block[: self.dim]
        vec = block.astype(np.float32)
        # Ensure non-zero vector for degenerate inputs
        n = float(np.linalg.norm(vec))
        if n < 1e-6:
            vec[0] = 1.0
            n = 1.0
        return vec / n


class LightLiveness(LivenessChecker):
    def passive(self, frame_bgr: np.ndarray, det: Detection) -> LivenessResult:
        x, y, w, h = det.bbox
        face = frame_bgr[max(0, y): y + h, max(0, x): x + w]
        if face.size == 0:
            return LivenessResult(0.1, False, "invalid crop")
        gray = _bgr_to_gray(face)
        # Sharpness via Laplacian variance
        sharp = float(_laplacian_var(gray))
        # Frequency content ratio to detect printed/photo flatness
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        center_energy = mag[cy - 5:cy + 5, cx - 5:cx + 5].sum() + 1e-6
        total_energy = mag.sum() + 1e-6
        hf_ratio = 1.0 - float(center_energy / total_energy)

        # Face size proxy
        size_ok = (det.bbox[2] >= 120) and (det.bbox[3] >= 120)

        # Heuristic score
        score = 0.0
        expl = []
        if sharp > 80:
            score += 0.35
            expl.append("sharp")
        if hf_ratio > 0.88:
            score += 0.35
            expl.append("texture")
        if size_ok:
            score += 0.3
            expl.append("size")
        is_live = score >= 0.6
        return LivenessResult(score, is_live, ",".join(expl) or "low confidence")


def _rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
    return (0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]).astype(np.float32)


def _bgr_to_gray(img_bgr: np.ndarray) -> np.ndarray:
    # BGR to grayscale
    return (0.114 * img_bgr[..., 0] + 0.587 * img_bgr[..., 1] + 0.299 * img_bgr[..., 2]).astype(np.float32)


def _resize_nn(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    # Very simple nearest-neighbor resize without cv2
    h, w = img.shape[:2]
    new_w, new_h = size
    y_idx = (np.linspace(0, h - 1, new_h)).astype(int)
    x_idx = (np.linspace(0, w - 1, new_w)).astype(int)
    if img.ndim == 2:
        return img[y_idx][:, x_idx]
    return img[y_idx][:, x_idx, :]

def _laplacian_var(gray: np.ndarray) -> float:
    # Simple 3x3 Laplacian via naive convolution (cv2-free)
    pad = np.pad(gray, 1, mode="edge")
    out = pad[1:-1, 2:] + pad[1:-1, :-2] + pad[2:, 1:-1] + pad[:-2, 1:-1] - 4 * pad[1:-1, 1:-1]
    return float(out.var())
