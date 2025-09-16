from typing import Tuple
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def vector_search(query_vec: np.ndarray, matrix: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return np.array([]), np.array([])
    sims = cosine_similarity(query_vec.reshape(1, -1), matrix).reshape(-1)
    idx = np.argsort(-sims)[:top_k]
    return sims[idx], idx
