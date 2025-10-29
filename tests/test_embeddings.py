import numpy as np
from smart_kiosk.models.opencv_fallback import DCTEmbedder


def test_embedding_size_and_norm():
    emb = DCTEmbedder()
    dummy = np.zeros((112, 112, 3), dtype=np.uint8)
    v = emb.embed(dummy)
    assert v.shape[0] == emb.dim
    n = np.linalg.norm(v)
    assert 0.99 <= n <= 1.01

