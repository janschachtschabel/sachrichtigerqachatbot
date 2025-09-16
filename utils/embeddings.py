import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class EmbeddingBackend:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=None)
        # Infer embedding dimension by quick encode
        _tmp = self.model.encode(["test"], normalize_embeddings=True)
        self.dim = int(_tmp.shape[-1])

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32)
