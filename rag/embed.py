r"""Embedding wrapper built on fastembed (onnxruntime).

We use fastembed rather than sentence-transformers because Windows Smart App
Control blocks SciPy's compiled DLLs (which transformers / sentence-transformers
import at load time). fastembed runs the *same* ``BAAI/bge-base-en-v1.5`` model
exported to ONNX on onnxruntime (Microsoft-signed → allowed), and applies BGE's
query instruction internally. Vectors are L2-normalized (cosine == dot product).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import EMBED_BATCH_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)
        self.model_name = model_name
        self.dim = EMBEDDING_DIM

    def embed_passages(self, texts: list[str], *, batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
        """Embed document chunks (fastembed applies the passage formatting)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = list(self.model.passage_embed(texts, batch_size=batch_size))
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query (fastembed prepends the BGE query instruction)."""
        return np.asarray(list(self.model.query_embed([query]))[0], dtype=np.float32)

    def warmup(self) -> None:
        self.embed_query("warmup")


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
