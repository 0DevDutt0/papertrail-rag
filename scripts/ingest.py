#!/usr/bin/env python
r"""CLI: ingest every PDF in ``data/pdfs`` into the Qdrant vector store.

Usage:
    python -m scripts.ingest                 # rebuild the collection from all PDFs
    python -m scripts.ingest --no-recreate   # only add PDFs not already ingested
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import PDF_DIR  # noqa: E402
from rag.ingest import ingest_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest PDFs into Qdrant.")
    ap.add_argument("--no-recreate", action="store_true",
                    help="keep the existing collection and skip already-ingested files")
    ap.add_argument("--no-ocr", action="store_true",
                    help="skip OCR for speed (the bundled corpus PDFs already have text layers)")
    ap.add_argument("--pdf-dir", default=str(PDF_DIR))
    args = ap.parse_args()

    def on_event(e: dict) -> None:
        t = e.get("type")
        if t == "start":
            print(f"Ingesting {e['num_pdfs']} PDF(s) from {args.pdf_dir}\n")
        elif t == "extract":
            print(f"- {e['file']}: extracting (+OCR where needed)…")
        elif t == "embed":
            print(f"  {e['file']}: {e['pages']} pages ({e['ocr_pages']} OCR'd), "
                  f"{e['chunks']} chunks -> embedding…")
        elif t == "file_done":
            print(f"  done {e['file']}: {e['chunks']} chunks in {e['seconds']}s")
        elif t == "skip":
            print(f"= {e['file']}: already ingested, skipped")
        elif t == "done":
            print(f"\nFinished. files={e['files']}  total_chunks_in_store={e['total_chunks']}")

    t0 = time.perf_counter()
    ingest_all(args.pdf_dir, recreate=not args.no_recreate,
               enable_ocr=not args.no_ocr, on_event=on_event)
    print(f"Elapsed: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
