r"""Cross-encoder reranker built on fastembed (onnxruntime), Smart-App-Control safe.

Scores each (query, chunk) pair with ``BAAI/bge-reranker-base`` and returns the
top-``k`` candidates, attaching a ``rerank_score`` to each.
"""
from __future__ import annotations

from functools import lru_cache

from .config import RERANKER_MODEL, TOP_K_RERANK


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name)
        self.model_name = model_name

    def rerank(self, query: str, candidates: list[dict], *, top_k: int = TOP_K_RERANK) -> list[dict]:
        if not candidates:
            return []
        docs = [c.get("text", "") for c in candidates]
        scores = list(self.model.rerank(query, docs))
        ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
        out: list[dict] = []
        for cand, score in ranked[:top_k]:
            d = dict(cand)
            d["rerank_score"] = float(score)
            out.append(d)
        return out

    def warmup(self) -> None:
        self.rerank("warmup", [{"text": "warmup document"}], top_k=1)


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
