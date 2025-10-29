import time
import zlib
import numpy as np


def now_ts() -> float:
    return time.time()


def compress_embedding(vec: np.ndarray) -> bytes:
    assert vec.dtype == np.float32
    return zlib.compress(vec.tobytes(), level=6)


def decompress_embedding(blob: bytes) -> np.ndarray:
    raw = zlib.decompress(blob)
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)

