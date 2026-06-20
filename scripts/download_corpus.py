#!/usr/bin/env python
r"""Download the curated PDF corpus into ``data/pdfs``.

Usage:
    python -m scripts.download_corpus               # all 11 PDFs
    python -m scripts.download_corpus --subset 3    # first 3 (quick smoke test)
    python -m scripts.download_corpus --only biology-2e moby-dick
    python -m scripts.download_corpus --force       # re-download existing files

OpenStax books are resolved through the OpenStax CMS API so we always grab the
current asset URL; everything else uses a verified direct URL. See
``data/pdf_sources.md`` for the human-readable list.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

# Make the project importable whether run as a module or a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import PDF_DIR  # noqa: E402

OPENSTAX_API = (
    "https://openstax.org/apps/cms/api/v2/pages/"
    "?type=books.Book&fields=title,slug,high_resolution_pdf_url,low_resolution_pdf_url&slug={slug}"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (RAG-Hackathon corpus downloader)"}

# Each entry: name, title, filename, source ('openstax'|'direct'), slug/url, fallback_url, pages
CORPUS: list[dict] = [
    dict(name="biology-2e", title="OpenStax Biology 2e", filename="OpenStax_Biology_2e.pdf",
         source="openstax", slug="biology-2e",
         fallback_url="https://assets.openstax.org/oscms-prodcms/media/documents/Biology-2e_-_WEB.pdf",
         pages=1450),
    dict(name="chemistry-2e", title="OpenStax Chemistry 2e", filename="OpenStax_Chemistry_2e.pdf",
         source="openstax", slug="chemistry-2e",
         fallback_url="https://assets.openstax.org/oscms-prodcms/media/documents/chemistry-2e_-_WEB.pdf",
         pages=1300),
    dict(name="calculus-volume-1", title="OpenStax Calculus Volume 1", filename="OpenStax_Calculus_Volume_1.pdf",
         source="openstax", slug="calculus-volume-1",
         fallback_url="https://assets.openstax.org/oscms-prodcms/media/documents/calculus-volume-1_-_WEB.pdf",
         pages=870),
    dict(name="anatomy-and-physiology-2e", title="OpenStax Anatomy & Physiology 2e",
         filename="OpenStax_Anatomy_and_Physiology_2e.pdf",
         source="openstax", slug="anatomy-and-physiology-2e",
         fallback_url="https://assets.openstax.org/oscms-prodcms/media/documents/anatomy-and-physiology-2e_-_WEB.pdf",
         pages=1400),
    dict(name="astronomy-2e", title="OpenStax Astronomy 2e", filename="OpenStax_Astronomy_2e.pdf",
         source="openstax", slug="astronomy-2e",
         fallback_url="https://assets.openstax.org/oscms-prodcms/media/documents/astronomy-2e_-_WEB.pdf",
         pages=1250),
    dict(name="university-physics-volume-1", title="OpenStax University Physics Volume 1",
         filename="OpenStax_University_Physics_Volume_1.pdf",
         source="openstax", slug="university-physics-volume-1",
         fallback_url="https://d3bxy9euw4e147.cloudfront.net/oscms-prodcms/media/documents/UniversityPhysicsVolume1-LR.pdf",
         pages=900),
    dict(name="university-physics-volume-2", title="OpenStax University Physics Volume 2",
         filename="OpenStax_University_Physics_Volume_2.pdf",
         source="openstax", slug="university-physics-volume-2",
         fallback_url="https://d3bxy9euw4e147.cloudfront.net/oscms-prodcms/media/documents/UniversityPhysicsVolume2-LR.pdf",
         pages=840),
    dict(name="university-physics-volume-3", title="OpenStax University Physics Volume 3",
         filename="OpenStax_University_Physics_Volume_3.pdf",
         source="openstax", slug="university-physics-volume-3",
         fallback_url="https://d3bxy9euw4e147.cloudfront.net/oscms-prodcms/media/documents/UniversityPhysicsVolume3-LR.pdf",
         pages=700),
    dict(name="9-11-commission-report", title="The 9/11 Commission Report",
         filename="9-11_Commission_Report.pdf", source="direct",
         url="https://www.govinfo.gov/content/pkg/GPO-911REPORT/pdf/GPO-911REPORT.pdf", pages=585),
    dict(name="moby-dick", title="Moby-Dick (scanned)", filename="Moby_Dick_scanned.pdf", source="direct",
         url="https://archive.org/download/mobydickorwhale01melvuoft/mobydickorwhale01melvuoft.pdf", pages=398),
    dict(name="war-and-peace", title="War and Peace (scanned)", filename="War_and_Peace_scanned.pdf",
         source="direct",
         url="https://archive.org/download/warandpeace030164mbp/warandpeace030164mbp.pdf", pages=725),
]


def resolve_openstax_url(slug: str, fallback: str | None) -> str | None:
    """Ask the OpenStax CMS API for the current PDF URL; fall back if it fails."""
    try:
        r = requests.get(OPENSTAX_API.format(slug=slug), headers=HEADERS, timeout=30)
        r.raise_for_status()
        items = r.json().get("items", [])
        if items:
            # Prefer the smaller low-res PDF (same text, far smaller/faster to
            # ingest); fall back to high-res, then the hardcoded asset URL.
            url = (items[0].get("low_resolution_pdf_url")
                   or items[0].get("high_resolution_pdf_url"))
            if url:
                return url
    except Exception as e:  # noqa: BLE001
        print(f"    ! OpenStax API failed for '{slug}' ({e}); using fallback URL")
    return fallback


def download(url: str, dest: Path) -> bool:
    """Stream-download ``url`` to ``dest``. Returns True on success."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        print(f"\r    {pct:3d}%  ({done/1e6:,.1f} / {total/1e6:,.1f} MB)", end="")
            print()
        tmp.replace(dest)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"\n    ! download failed: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Download the curated RAG PDF corpus.")
    ap.add_argument("--subset", type=int, default=None, help="download only the first N PDFs")
    ap.add_argument("--only", nargs="*", default=None, help="download only these entry names")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()

    entries = CORPUS
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e["name"] in wanted]
    if args.subset:
        entries = entries[: args.subset]

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(entries)} PDF(s) into {PDF_DIR}\n")

    ok, skipped, failed = 0, 0, 0
    for i, e in enumerate(entries, 1):
        dest = PDF_DIR / e["filename"]
        print(f"[{i}/{len(entries)}] {e['title']}  (~{e['pages']} pages)")
        if dest.exists() and not args.force:
            print(f"    = already present, skipping ({dest.name})")
            skipped += 1
            continue
        if e["source"] == "openstax":
            url = resolve_openstax_url(e["slug"], e.get("fallback_url"))
        else:
            url = e["url"]
        if not url:
            print("    ! no URL resolved, skipping")
            failed += 1
            continue
        print(f"    -> {url}")
        if download(url, dest):
            ok += 1
        else:
            failed += 1

    print(f"\nDone. downloaded={ok}  skipped={skipped}  failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
