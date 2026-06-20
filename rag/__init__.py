"""RAG chatbot core package (Challenge 1).

Modules:
    config    - central configuration (paths, models, params; loads .env)
    extract   - PDF native text + bbox extraction with OCR fallback (RapidOCR)
    chunk     - deterministic token-based chunking with per-chunk metadata
    embed     - sentence-transformers embedding wrapper (GPU autodetect)
    store     - Qdrant (embedded) vector store: HNSW index, upsert, search
    rerank    - cross-encoder reranker
    llm       - Groq client: grounded, cited answer synthesis (streaming)
    ingest    - end-to-end ingestion orchestration
    pipeline  - end-to-end query -> retrieve -> rerank -> generate
"""

__all__ = [
    "config",
    "extract",
    "chunk",
    "embed",
    "store",
    "rerank",
    "llm",
    "ingest",
    "pipeline",
]
