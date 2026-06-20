r"""FastAPI app: serves the web UI **and** the RAG API.

Run:  uvicorn api:app --port 8000
  UI:        http://localhost:8000/
  API docs:  http://localhost:8000/docs
  Endpoints: POST /query (full JSON), POST /api/stream (NDJSON stream), GET /health

Note: the embedded Qdrant store holds an exclusive lock on data/qdrant, so run
either this server or the Streamlit app at a time (not both) unless QDRANT_URL
points at a Qdrant server.
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag import config, llm, pipeline
from rag.store import get_store

WEB_DIR = Path(__file__).parent / "web"


class Query(BaseModel):
    question: str = Field(..., description="User question")
    top_k: int = config.TOP_K_RETRIEVE
    final_k: int = config.TOP_K_RERANK
    use_reranker: bool = True
    model: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        pipeline.warmup()  # preload models so the first request meets the budget
    except Exception:
        pass
    yield


app = FastAPI(title="PaperTrail RAG API", version="1.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    store = get_store()
    return {
        "status": "ok",
        "chunks_indexed": store.count(),
        "files_indexed": len(store.list_files()),
        "embedding_model": config.EMBEDDING_MODEL,
        "reranker_model": config.RERANKER_MODEL,
        "llm_model": config.GROQ_MODEL,
        "llm_model_quality": config.GROQ_MODEL_QUALITY,
        "latency_budget_s": config.LATENCY_BUDGET_MAX_S,
    }


@app.post("/query")
def query(q: Query):
    """Full, non-streaming answer (answer + sources + retrieved + timings)."""
    return pipeline.answer_query(
        q.question, top_k=q.top_k, final_k=q.final_k, use_reranker=q.use_reranker,
    )


@app.post("/api/stream")
def stream(q: Query):
    """NDJSON stream consumed by the web UI:

    line 1  -> {"type":"meta", sources, retrieved, timings}   (after retrieval)
    line N  -> {"type":"token", "text": "..."}                (per LLM token)
    last    -> {"type":"done", "llm_ms", "total_ms"}
    """
    def gen():
        r = pipeline.retrieve(q.question, top_k=q.top_k, final_k=q.final_k,
                              use_reranker=q.use_reranker)
        view = pipeline._view
        meta = {
            "type": "meta",
            "sources": [view(c) for c in r["final"]],
            "retrieved": [view(c) for c in r["candidates"]],
            "timings": r["timings"],
        }
        yield json.dumps(meta) + "\n"
        t = time.perf_counter()
        for tok in llm.stream_answer(q.question, r["contexts"], model=q.model):
            yield json.dumps({"type": "token", "text": tok}) + "\n"
        llm_ms = round((time.perf_counter() - t) * 1000, 1)
        rt = r["timings"]
        total = round(rt["embed_ms"] + rt["search_ms"] + rt["rerank_ms"] + llm_ms, 1)
        yield json.dumps({"type": "done", "llm_ms": llm_ms, "total_ms": total}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
