r"""End-to-end ingestion: PDF -> extract(+OCR) -> chunk -> embed -> Qdrant.

``ingest_all`` accepts an optional ``on_event`` callback so both the CLI and the
Streamlit UI can render live progress. Re-ingesting is idempotent: with
``recreate=True`` the collection is rebuilt; with ``recreate=False`` already-
ingested files (matched by filename) are skipped.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .chunk import chunk_pages, chunk_to_payload, chunk_uuid
from .config import PDF_DIR
from .embed import Embedder, get_embedder
from .extract import extract_pdf
from .store import VectorStore, get_store


def _emit(cb: Callable[[dict], None] | None, event: dict) -> None:
    if cb:
        cb(event)


def ingest_pdf(path: str | Path, embedder: Embedder, store: VectorStore, *,
               enable_ocr: bool = True, on_event: Callable[[dict], None] | None = None) -> dict:
    path = Path(path)
    t0 = time.perf_counter()
    _emit(on_event, {"type": "extract", "file": path.name})
    pages = extract_pdf(path, enable_ocr=enable_ocr)
    n_ocr = sum(1 for p in pages if p.ocr)
    chunks = chunk_pages(pages)
    _emit(on_event, {"type": "embed", "file": path.name,
                     "pages": len(pages), "chunks": len(chunks), "ocr_pages": n_ocr})
    if chunks:
        vectors = embedder.embed_passages([c.text for c in chunks])
        ids = [chunk_uuid(c.chunk_id) for c in chunks]
        payloads = [chunk_to_payload(c) for c in chunks]
        store.upsert(ids, vectors, payloads)
    stats = {
        "type": "file_done", "file": path.name, "pages": len(pages),
        "ocr_pages": n_ocr, "chunks": len(chunks),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    _emit(on_event, stats)
    return stats


def ingest_all(pdf_dir: str | Path = PDF_DIR, *, recreate: bool = True, enable_ocr: bool = True,
               on_event: Callable[[dict], None] | None = None) -> list[dict]:
    embedder = get_embedder()
    store = get_store()
    store.ensure_collection(dim=embedder.dim, recreate=recreate)
    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    _emit(on_event, {"type": "start", "num_pdfs": len(pdfs)})
    summary: list[dict] = []
    for path in pdfs:
        if not recreate and store.filename_exists(path.name):
            _emit(on_event, {"type": "skip", "file": path.name})
            continue
        summary.append(ingest_pdf(path, embedder, store, enable_ocr=enable_ocr, on_event=on_event))
    _emit(on_event, {"type": "done", "files": len(summary), "total_chunks": store.count()})
    return summary
