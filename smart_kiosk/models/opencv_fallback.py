import numpy as np
from typing import List, Tuple
from .base import Detection, FaceDetector, FaceAligner, FaceEmbedder, LivenessChecker, LivenessResult


class HaarFaceDetector(FaceDetector):
    def __init__(self):
        try:
            import cv2  # type: ignore

            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cv2 = cv2
            self.det = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            raise RuntimeError("OpenCV is required for HaarFaceDetector") from e

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        gray = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2GRAY)
        faces = self.det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        dets: List[Detection] = []
        for (x, y, w, h) in faces:
            score = float(min(1.0, (w * h) / (frame_bgr.shape[0] * frame_bgr.shape[1]) * 10.0))
            dets.append(Detection((int(x), int(y), int(w), int(h)), score))
        dets.sort(key=lambda d: d.bbox[2] * d.bbox[3], reverse=True)
        return dets


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
