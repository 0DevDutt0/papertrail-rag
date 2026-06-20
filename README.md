# 📚 PaperTrail — PDF RAG Chatbot (Challenge 1)

A Retrieval-Augmented Generation chatbot that answers questions from a **large private corpus of PDFs**
(≥10 PDFs, each ≥200 pages) using a **free / open-source** embedding model and vector DB, streams answers in
**2–5 seconds**, and cites every claim as **`[filename p.PAGE]`**.

It supports native PDF text extraction **and OCR** for scanned pages, deterministic chunking with per-chunk
metadata (file, page, bbox), ANN retrieval with a cross-encoder reranker, grounded answer synthesis with
citations, a polished **web UI** (served by FastAPI), a **Streamlit** app, a **REST API**, and an **evaluation
harness** (latency p95, Recall@k, MRR, citation accuracy, hallucination).

Live corpus: **11 PDFs → 14,049 chunks**. Retrieval pipeline p95 **~1.1 s**; end-to-end **~2 s** per query.

---

## The web interface

`uvicorn api:app --port 8000`, then open **http://localhost:8000** — a single dark, Linear-style chat app
(vanilla HTML/CSS/JS, no build step) that shows everything the brief asks a demo to show:

- **Streamed, cited answers** — the answer types out token-by-token; `[filename p.PAGE]` citations render as
  inline source chips.
- **Sources** — a card per source document (filename, page, similarity + rerank score), built with a nested
  "double-bezel" surface.
- **Retrieval inspector** — an expandable panel visualizing the top retrieved passages with score bars
  (the retrieval visualization).
- **Latency breakdown** — per-stage chips (embed / search / rerank / LLM) and a total badge that turns red if it
  ever exceeds the 5 s budget.
- **Settings** — live controls for top-K, context size, reranker on/off, and model (8B fast / 70B quality).
- **States** — composed empty state with example questions, skeleton loaders while retrieving, inline errors.

> The UI was built against the design system in `.agents/skills` (high-end-visual-design, minimalist-ui,
> design-taste-frontend): one locked emerald accent, Clash Display + Satoshi type, Phosphor icons, a single
> corner-radius scale, motivated motion only, and `prefers-reduced-motion` support.

A **Streamlit** version (`streamlit run app.py`) is also included with the same chat + ingestion + retrieval views.

---

## Architecture

```
INGESTION (offline, precomputed)
  PDF ─► PyMuPDF text + bbox ─► OCR fallback (RapidOCR) for scanned pages
      ─► clean (strip headers/footers, de-hyphenate, normalize, detect language)
      ─► chunk (480 tokens, 80 overlap, original text + page/bbox metadata)
      ─► embed (bge-base-en-v1.5, onnxruntime)  ─► upsert into Qdrant (HNSW, cosine)

QUERY (online, target 2–5 s)
  question ─► embed query ─► Qdrant ANN search (top-15 + score + metadata)
           ─► cross-encoder rerank (top-5) ─► Groq LLM (grounded + cited, streamed)
           ─► answer + sources[{filename, page, score}] + per-stage latency
```

## Open-source / free stack

| Component | Choice | Notes |
|---|---|---|
| PDF text + bbox | **PyMuPDF** | per-page text and block bounding boxes |
| OCR | **RapidOCR** (onnxruntime) | runs only on pages with sparse native text |
| Embeddings | **BAAI/bge-base-en-v1.5** (768-d) | via **fastembed / onnxruntime** |
| Vector DB | **Qdrant** (embedded) | HNSW index, cosine, metadata payload + scores |
| Reranker | **ms-marco-MiniLM-L-6-v2** | cross-encoder, keeps rerank < 1 s on CPU |
| LLM (synthesis) | **Groq** → open-source Llama | `llama-3.1-8b-instant` default, `llama-3.3-70b-versatile` optional; **GLM** fallback |
| Web UI | vanilla **HTML/CSS/JS** served by FastAPI | streamed (NDJSON), no build step |
| App / API | **Streamlit** / **FastAPI + uvicorn** | chat, ingestion view, retrieval viz, metrics |

> **Why onnxruntime, not PyTorch?** This machine has Windows **Smart App Control** enabled, which blocks SciPy's
> compiled DLLs — and `transformers` / `sentence-transformers` import SciPy at load time. So embeddings and
> reranking run on **onnxruntime** via `fastembed` (Microsoft-signed, allowed) using the *same* BGE models.
> **No PyTorch is required.**

---

## Setup

```bash
# from E:\Hackathon
venv\Scripts\activate                 # (venv already created with Python 3.13)
pip install -r requirements.txt
```

`.env` (already present) holds the keys:

```
GROQ_API_KEY=...        # primary answer synthesis (free tier, open-source Llama)
GLM_API_KEY=...         # secondary fallback (GLM / Zhipu) if Groq is rate-limited
MISTRAL_API_KEY=...     # present but unused (mistralai wheel ships broken on Python 3.13 here)
```

## Quickstart

```bash
# 1) Download the corpus (11 PDFs, each ≥200 pages) into data/pdfs
python -m scripts.download_corpus              # or  --subset 3  for a quick start

# 2) Ingest into Qdrant (PDF → extract/OCR → chunk → embed → vector DB)
python -m scripts.ingest                       # rebuilds the index; add --no-ocr to skip OCR for speed

# 3) Run the web UI + API (recommended)
uvicorn api:app --port 8000                    # then open http://localhost:8000
#   REST:  curl -X POST localhost:8000/query -H "Content-Type: application/json" \
#                -d '{"question":"What happened aboard United Airlines Flight 93?"}'

# 3b) …or the Streamlit app
streamlit run app.py

# 4) Evaluate (latency p95, Recall@k, MRR, citation accuracy, hallucination)
python evaluate.py --save --judge
```

The corpus and direct download links are documented in **[`data/pdf_sources.md`](data/pdf_sources.md)**.

## Latency (2–5 s)

Models are **warmed at startup** and embeddings are **precomputed**, so a typical query is:
`embed ≈ 10 ms · search ≈ 5 ms · rerank ≈ 0.9 s · LLM ≈ 1–1.5 s` → **~2 s end-to-end**.
The 70B model adds ~1 s. During *burst* testing, Groq's free tier rate-limits and adds latency, so the eval
harness spaces requests (`--delay`) and the app fails over fast (Groq 8B → 70B → GLM).

## Evaluation

`python evaluate.py --save` writes `data/eval/results/eval_report.md`. On the full corpus the pipeline scores:

- **Retrieval pipeline latency:** p50 **0.98 s**, p95 **1.11 s**
- **End-to-end:** p50 **1.84 s**, p95 **2.06 s** — **100% within the 5 s budget**
- **Recall@15 = 1.00 · MRR = 0.84 · citation accuracy = 0.88–1.00 · answer-keyword accuracy = 1.00 · 0% abstention**

Add `--judge` for an LLM-as-judge hallucination check. Extend `data/eval/qa_testset.json` to cover more documents.

---

## Notes & limitations

- **Run one of {web/API server, Streamlit, CLI ingest} at a time.** The embedded Qdrant store holds an exclusive
  lock on `data/qdrant`. (A hard-killed process leaves a stale lock; the store now clears it automatically on the
  next start, *only* if no live process still holds it.) For real concurrency, run a Qdrant server and set
  `QDRANT_URL` — the code switches automatically and then uses a true HNSW ANN index (the embedded store does
  exact KNN, which is fast and exact at this corpus size).
- **Ingestion time:** embeddings run on CPU (onnxruntime) at ~12 chunks/s, so the full corpus (~14k chunks) takes
  ~25 min — a one-time offline step. Use `--subset` / `--no-ocr` to speed it up.
- **OCR** triggers automatically when a page's native text is sparse *and* the page contains an image; the
  Internet-Archive scans in the corpus exercise this path.
- A capable GPU is present but **not required** — the pipeline is CPU-only by design (onnxruntime) to stay within
  Smart App Control constraints.

---

## Author

**Devdutt S** — Kochi, Kerala, India
[GitHub](https://github.com/0DevDutt0) · [LinkedIn](https://www.linkedin.com/in/devdutts/) · devduttshoji123@gmail.com
