r"""Central configuration for the RAG chatbot.

All tunables live here so ingestion, the query pipeline, the UI, the API and
the evaluation harness share one source of truth. Secrets are read from
``E:\Hackathon\.env`` (GROQ_API_KEY, MISTRAL_API_KEY, GLM_API_KEY).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # E:\Hackathon
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
QDRANT_PATH = DATA_DIR / "qdrant"
EVAL_DIR = DATA_DIR / "eval"

for _d in (DATA_DIR, PDF_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Load secrets from .env at import time.
load_dotenv(BASE_DIR / ".env")


# --- Device ----------------------------------------------------------------
def get_device() -> str:
    """Return 'cuda' if a working CUDA build of torch is present, else 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


DEVICE = get_device()

# --- Embedding model (free / open-source) ----------------------------------
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"   # 768-dim, strong quality/speed
EMBEDDING_DIM = 768
# (fastembed applies BGE's query instruction internally for query_embed.)
EMBED_BATCH_SIZE = 128

# --- Reranker (cross-encoder) ----------------------------------------------
# MiniLM-L6 is ~10x lighter than bge-reranker-base -> keeps rerank well under a
# second on CPU (onnxruntime) so the full query stays in the 2-5 s budget.
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
USE_RERANKER = True

# --- Chunking (deterministic) ----------------------------------------------
# bge-base-en-v1.5 has a 512-token window, so we cap chunks at 480 content
# tokens (+2 special tokens at embed time) to avoid silent truncation.
CHUNK_TOKENS = 480
CHUNK_OVERLAP = 80           # ~17% overlap
EMBED_MAX_SEQ = 512

# --- Retrieval -------------------------------------------------------------
TOP_K_RETRIEVE = 15          # ANN candidates pulled from Qdrant
TOP_K_RERANK = 5             # final chunks passed to the LLM
QDRANT_COLLECTION = "pdf_chunks"
# If QDRANT_URL is set we connect to a running Qdrant server (true HNSW ANN
# index). Otherwise we use the embedded/local store (exact KNN — fine for this
# corpus size and latency; same client code, just flip the env var).
QDRANT_URL = os.getenv("QDRANT_URL", "")
HNSW_M = 16
HNSW_EF_CONSTRUCT = 200
HNSW_EF_SEARCH = 128

# --- OCR (RapidOCR fallback for scanned / image pages) ---------------------
OCR_MIN_CHARS = 50           # pages with less native text than this get OCR'd
OCR_DPI = 200

# --- LLM (Groq — hosted, open-source Llama models) -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
LLM_PROVIDER = "groq"
GROQ_MODEL = "llama-3.1-8b-instant"           # default: fast + rate-limit resilient, in-budget
GROQ_MODEL_QUALITY = "llama-3.3-70b-versatile"  # optional higher-quality synthesis / fallback
# Secondary-provider fallback (used only if Groq is unavailable / rate-limited):
GLM_MODEL = "glm-4.5-flash"
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# --- Latency budget (seconds) ----------------------------------------------
LATENCY_BUDGET_MIN_S = 2.0
LATENCY_BUDGET_MAX_S = 5.0
