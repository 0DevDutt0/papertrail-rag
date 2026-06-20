r"""Deterministic, token-based chunking with page-level provenance.

We tokenize each page with the embedding model's tokenizer (the standalone HF
``tokenizers`` lib — no SciPy), keeping each token's character offsets, then
slide a fixed window (``CHUNK_TOKENS`` with ``CHUNK_OVERLAP`` stride) across the
document's token stream. Crucially we reconstruct each chunk's text by slicing
the *original* page text via those offsets, so casing, punctuation and special
characters are preserved (what we show to the user and feed to the LLM). Each
chunk records the pages it spans, a per-page text bbox, and a stable ``chunk_id``
so re-ingesting the same PDF yields identical chunks/ids (reproducibility).
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import groupby

from .config import CHUNK_OVERLAP, CHUNK_TOKENS, EMBEDDING_MODEL
from .extract import PageContent


@dataclass
class Chunk:
    chunk_id: str                 # sha1 hex of f"{pdf_id}:{chunk_index}"
    pdf_id: str
    filename: str
    chunk_index: int
    text: str
    token_count: int
    page_start: int
    page_end: int
    pages: list[int]
    language: str = "unknown"
    ocr: bool = False
    page_bboxes: dict[str, list[float]] = field(default_factory=dict)

    @property
    def page_label(self) -> str:
        return f"p.{self.page_start}" if self.page_start == self.page_end \
            else f"p.{self.page_start}-{self.page_end}"


@lru_cache(maxsize=2)
def _get_tokenizer(model_name: str = EMBEDDING_MODEL):
    # Standalone HF `tokenizers` (Rust) — does NOT pull SciPy, so it works under
    # Windows Smart App Control. Same vocab/offsets as the fastembed encoder.
    from tokenizers import Tokenizer

    return Tokenizer.from_pretrained(model_name)


def chunk_uuid(chunk_id: str) -> str:
    """Map a sha1 chunk_id to a deterministic UUID (Qdrant point id)."""
    return str(uuid.UUID(hex=chunk_id[:32]))


def chunk_pages(pages: list[PageContent], *, chunk_tokens: int = CHUNK_TOKENS,
                overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Chunk a document's pages into overlapping token windows (original text)."""
    if not pages:
        return []

    tok = _get_tokenizer()
    pdf_id = pages[0].pdf_id
    filename = pages[0].filename
    by_page = {p.page_number: p for p in pages}
    page_bbox = {p.page_number: list(bb) for p in pages if (bb := p.text_bbox())}

    # Flatten tokens, tagging each with (page_number, char_start, char_end).
    page_texts: dict[int, str] = {}
    tokens: list[tuple[int, int, int]] = []
    for p in pages:
        if not p.text.strip():
            continue
        page_texts[p.page_number] = p.text
        enc = tok.encode(p.text, add_special_tokens=False)
        for start, end in enc.offsets:
            tokens.append((p.page_number, start, end))

    if not tokens:
        return []

    step = max(1, chunk_tokens - overlap)
    chunks: list[Chunk] = []
    i = 0
    idx = 0
    n = len(tokens)
    while i < n:
        window = tokens[i:i + chunk_tokens]
        parts: list[str] = []
        pages_in: list[int] = []
        # Reconstruct text per contiguous page run by slicing the original text.
        for page_no, grp in groupby(window, key=lambda t: t[0]):
            grp = list(grp)
            seg = page_texts[page_no][grp[0][1]:grp[-1][2]].strip()
            if seg:
                parts.append(seg)
                pages_in.append(page_no)
        text = "\n".join(parts).strip()
        if text and pages_in:
            uniq = sorted(set(pages_in))
            cid = hashlib.sha1(f"{pdf_id}:{idx}".encode()).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=cid, pdf_id=pdf_id, filename=filename, chunk_index=idx,
                    text=text, token_count=len(window),
                    page_start=uniq[0], page_end=uniq[-1], pages=uniq,
                    language=by_page[uniq[0]].language,
                    ocr=any(by_page[pg].ocr for pg in uniq),
                    page_bboxes={str(pg): page_bbox[pg] for pg in uniq if pg in page_bbox},
                )
            )
            idx += 1
        if i + chunk_tokens >= n:
            break
        i += step
    return chunks


def chunk_to_payload(c: Chunk) -> dict:
    """Flatten a Chunk into a Qdrant payload (metadata + text)."""
    return {
        "chunk_id": c.chunk_id,
        "pdf_id": c.pdf_id,
        "filename": c.filename,
        "chunk_index": c.chunk_index,
        "text": c.text,
        "token_count": c.token_count,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "pages": c.pages,
        "page_label": c.page_label,
        "language": c.language,
        "ocr": c.ocr,
        "page_bboxes": c.page_bboxes,
    }
