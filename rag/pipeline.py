r"""End-to-end RAG query pipeline: embed -> ANN search -> rerank -> generate.

``retrieve`` runs the retrieval half (embed + search + rerank) so the UI can
display the retrieved chunks before streaming the answer. ``answer_query`` runs
the whole thing and is what the API / evaluation use. Both return per-stage
timings so we can check the 2-5 s latency budget.
"""
from __future__ import annotations

import time
from typing import Callable

from . import llm
from .config import (LATENCY_BUDGET_MAX_S, TOP_K_RERANK, TOP_K_RETRIEVE,
                     USE_RERANKER)
from .embed import get_embedder
from .rerank import get_reranker
from .store import get_store


def _snippet(text: str, n: int = 240) -> str:
    text = " ".join((text or "").split())
    return text[:n] + ("…" if len(text) > n else "")


def _ctx(c: dict) -> dict:
    return {
        "filename": c.get("filename", "?"),
        "page_label": c.get("page_label", ""),
        "page_start": c.get("page_start"),
        "text": c.get("text", ""),
    }


def _view(c: dict) -> dict:
    """Compact view of a candidate for the retrieval visualization / sources."""
    return {
        "filename": c.get("filename", "?"),
        "page_label": c.get("page_label", ""),
        "page_start": c.get("page_start"),
        "score": round(float(c.get("score", 0.0)), 4),
        "rerank_score": round(float(c["rerank_score"]), 4) if "rerank_score" in c else None,
        "snippet": _snippet(c.get("text", "")),
    }


def retrieve(question: str, *, top_k: int = TOP_K_RETRIEVE, final_k: int = TOP_K_RERANK,
             use_reranker: bool = USE_RERANKER) -> dict:
    """Embed -> ANN search -> (optional) rerank. Returns candidates + final + timings."""
    timings: dict[str, float] = {}

    t = time.perf_counter()
    qv = get_embedder().embed_query(question)
    timings["embed_ms"] = round((time.perf_counter() - t) * 1000, 1)

    t = time.perf_counter()
    hits = get_store().search(qv, limit=top_k)
    timings["search_ms"] = round((time.perf_counter() - t) * 1000, 1)
    candidates = [{**payload, "score": score} for (_id, score, payload) in hits]

    if use_reranker and candidates:
        t = time.perf_counter()
        final = get_reranker().rerank(question, candidates, top_k=final_k)
        timings["rerank_ms"] = round((time.perf_counter() - t) * 1000, 1)
    else:
        final = candidates[:final_k]
        timings["rerank_ms"] = 0.0

    return {
        "candidates": candidates,
        "final": final,
        "contexts": [_ctx(c) for c in final],
        "timings": timings,
    }


def answer_query(question: str, *, top_k: int = TOP_K_RETRIEVE, final_k: int = TOP_K_RERANK,
                 use_reranker: bool = USE_RERANKER,
                 stream_callback: Callable[[str], None] | None = None) -> dict:
    t_total = time.perf_counter()
    r = retrieve(question, top_k=top_k, final_k=final_k, use_reranker=use_reranker)
    timings = dict(r["timings"])

    t = time.perf_counter()
    answer_text = ""
    if stream_callback:
        for tok in llm.stream_answer(question, r["contexts"]):
            answer_text += tok
            stream_callback(tok)
    else:
        answer_text = llm.answer(question, r["contexts"])
    timings["llm_ms"] = round((time.perf_counter() - t) * 1000, 1)
    timings["total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)

    total_s = timings["total_ms"] / 1000.0
    return {
        "question": question,
        "answer": answer_text,
        "sources": [_view(c) for c in r["final"]],
        "retrieved": [_view(c) for c in r["candidates"]],
        "timings": timings,
        "within_budget": total_s <= LATENCY_BUDGET_MAX_S,
        "num_candidates": len(r["candidates"]),
    }


def warmup() -> None:
    """Load all models (and touch the store) so the first real query is fast."""
    get_embedder().warmup()
    get_reranker().warmup()
    try:
        get_store().count()
    except Exception:
        pass
