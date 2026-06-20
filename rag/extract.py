r"""PDF text extraction with an OCR fallback.

For each page we pull the native text layer and per-block bounding boxes with
PyMuPDF. Pages whose native text is sparse (scanned / image-only) are rendered
to an image and passed through RapidOCR. We then strip repeated headers/footers,
de-hyphenate line breaks, normalize whitespace, and detect the language.

Returns a list of :class:`PageContent`, one per page (1-based ``page_number``).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from .config import OCR_DPI, OCR_MIN_CHARS

# langdetect is made deterministic via a fixed seed.
try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
except Exception:  # pragma: no cover
    detect = None


@dataclass
class Block:
    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF points


@dataclass
class PageContent:
    pdf_id: str
    filename: str
    page_number: int          # 1-based
    text: str
    blocks: list[Block] = field(default_factory=list)
    language: str = "unknown"
    ocr: bool = False
    width: float = 0.0
    height: float = 0.0

    def text_bbox(self) -> tuple[float, float, float, float] | None:
        """Union bounding box of all text blocks on the page (text region)."""
        if not self.blocks:
            return None
        xs0 = min(b.bbox[0] for b in self.blocks)
        ys0 = min(b.bbox[1] for b in self.blocks)
        xs1 = max(b.bbox[2] for b in self.blocks)
        ys1 = max(b.bbox[3] for b in self.blocks)
        return (round(xs0, 1), round(ys0, 1), round(xs1, 1), round(ys1, 1))


# --- OCR engine (lazy singleton) -------------------------------------------
_ocr_engine = None
_ocr_new_api = False


def _get_ocr():
    global _ocr_engine, _ocr_new_api
    if _ocr_engine is None:
        try:  # unified package (rapidocr >= 2.0)
            from rapidocr import RapidOCR

            _ocr_new_api = True
        except Exception:  # fall back to legacy package name
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            _ocr_new_api = False
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _ocr_page(page: "fitz.Page", dpi: int) -> tuple[str, list[Block]]:
    """Render a page to an image and OCR it. Returns (text, blocks)."""
    import numpy as np

    engine = _get_ocr()
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    img = np.ascontiguousarray(img[:, :, ::-1])  # RGB -> BGR for OpenCV-based OCR

    out = engine(img)
    scale = 72.0 / dpi  # image pixels -> PDF points

    texts: list[str] = []
    boxes = []
    if _ocr_new_api and hasattr(out, "txts"):
        texts = list(out.txts or [])
        boxes = list(out.boxes) if out.boxes is not None else []
    else:
        result = out[0] if isinstance(out, tuple) else out
        result = result or []
        for line in result:
            boxes.append(line[0])
            texts.append(line[1])

    blocks: list[Block] = []
    for quad, txt in zip(boxes, texts):
        try:
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            bbox = (min(xs) * scale, min(ys) * scale, max(xs) * scale, max(ys) * scale)
        except Exception:
            bbox = (0.0, 0.0, 0.0, 0.0)
        blocks.append(Block(text=txt, bbox=bbox))

    return "\n".join(texts).strip(), blocks


# --- Text cleaning ---------------------------------------------------------
def _normalize_line(line: str) -> str:
    s = re.sub(r"\d+", "", line)          # mask page numbers
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _detect_headers_footers(pages_lines: list[list[str]], n_edge: int = 2,
                            threshold: float = 0.6) -> set[str]:
    """Find normalized first/last lines that recur across most pages."""
    counter: Counter[str] = Counter()
    n = len(pages_lines)
    if n < 5:
        return set()
    for lines in pages_lines:
        edge = lines[:n_edge] + lines[-n_edge:]
        for norm in {_normalize_line(l) for l in edge if l.strip()}:
            if norm:
                counter[norm] += 1
    cutoff = max(3, int(threshold * n))
    return {s for s, c in counter.items() if c >= cutoff}


def _clean_text(text: str, header_footer: set[str], n_edge: int = 2) -> str:
    lines = text.split("\n")
    keep: list[str] = []
    for i, line in enumerate(lines):
        is_edge = i < n_edge or i >= len(lines) - n_edge
        if is_edge and _normalize_line(line) in header_footer:
            continue
        keep.append(line)
    out = "\n".join(keep)
    out = unicodedata.normalize("NFKC", out)
    out = re.sub(r"-\n([a-z])", r"\1", out)      # de-hyphenate line breaks
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _detect_lang(text: str) -> str:
    if detect is None or len(text.strip()) < 20:
        return "unknown"
    try:
        return detect(text[:2000])
    except Exception:
        return "unknown"


# --- Public API ------------------------------------------------------------
def extract_pdf(path: str | Path, *, ocr_min_chars: int = OCR_MIN_CHARS,
                ocr_dpi: int = OCR_DPI, enable_ocr: bool = True) -> list[PageContent]:
    """Extract cleaned, per-page content (with OCR fallback) from a PDF."""
    path = Path(path)
    doc = fitz.open(path)
    filename = path.name
    pdf_id = path.stem

    raw_pages = []
    for i, page in enumerate(doc):
        blocks_raw = page.get_text("blocks")
        text_blocks = [
            Block(text=b[4].strip(), bbox=(b[0], b[1], b[2], b[3]))
            for b in blocks_raw
            if b[4].strip() and (len(b) < 7 or b[6] == 0)  # block_type 0 == text
        ]
        page_text = "\n".join(b.text for b in text_blocks).strip()
        ocr_used = False
        # OCR only sparse-text pages that actually contain an image (skip blank
        # pages — OCR-ing them just wastes time).
        if enable_ocr and len(page_text) < ocr_min_chars and page.get_images(full=False):
            ocr_text, ocr_blocks = _ocr_page(page, ocr_dpi)
            if len(ocr_text) > len(page_text):
                page_text, text_blocks, ocr_used = ocr_text, ocr_blocks, True
        raw_pages.append(
            dict(page_no=i + 1, text=page_text, blocks=text_blocks,
                 width=page.rect.width, height=page.rect.height, ocr=ocr_used)
        )
    doc.close()

    header_footer = _detect_headers_footers([p["text"].split("\n") for p in raw_pages])

    pages: list[PageContent] = []
    for p in raw_pages:
        cleaned = _clean_text(p["text"], header_footer)
        pages.append(
            PageContent(
                pdf_id=pdf_id, filename=filename, page_number=p["page_no"],
                text=cleaned, blocks=p["blocks"], language=_detect_lang(cleaned),
                ocr=p["ocr"], width=p["width"], height=p["height"],
            )
        )
    return pages


if __name__ == "__main__":  # quick manual check: python -m rag.extract <pdf>
    import sys

    pdf = sys.argv[1]
    pages = extract_pdf(pdf)
    ocr_pages = sum(1 for p in pages if p.ocr)
    chars = sum(len(p.text) for p in pages)
    print(f"{pdf}: {len(pages)} pages, {ocr_pages} OCR'd, {chars:,} chars")
    for p in pages[:3]:
        print(f"\n--- p.{p.page_number} (lang={p.language}, ocr={p.ocr}) ---")
        print(p.text[:400])
