r"""Streamlit demo for the PDF RAG chatbot.

    streamlit run app.py

Shows the three things the brief asks a demo to show:
  1. the ingestion pipeline (PDF -> extract/OCR -> chunk -> embed -> Qdrant),
  2. retrieval visualization (top retrieved chunks with scores), and
  3. the final, streamed answer with [filename p.PAGE] citations + latency.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag import config, llm, pipeline          # noqa: E402
from rag.embed import get_embedder              # noqa: E402
from rag.ingest import ingest_pdf               # noqa: E402
from rag.store import get_store                 # noqa: E402

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading embedding + reranker models (first run downloads them)…")
def _bootstrap() -> bool:
    pipeline.warmup()
    return True


_bootstrap()

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def render_latency(t: dict) -> None:
    c = st.columns(5)
    c[0].metric("Embed", f"{t.get('embed_ms', 0):.0f} ms")
    c[1].metric("Search", f"{t.get('search_ms', 0):.0f} ms")
    c[2].metric("Rerank", f"{t.get('rerank_ms', 0):.0f} ms")
    c[3].metric("LLM", f"{t.get('llm_ms', 0):.0f} ms")
    total = t.get("total_ms", 0)
    c[4].metric("Total", f"{total / 1000:.2f} s",
                delta="within 5s" if total <= config.LATENCY_BUDGET_MAX_S * 1000 else "over budget",
                delta_color="normal" if total <= config.LATENCY_BUDGET_MAX_S * 1000 else "inverse")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.markdown("**Sources**")
    for s in sources:
        rr = f" · rerank {s['rerank_score']}" if s.get("rerank_score") is not None else ""
        st.caption(f"📄 `{s['filename']}` **{s['page_label']}** — similarity {s['score']}{rr}")


def render_retrieved(candidates: list[dict]) -> None:
    if not candidates:
        return
    with st.expander(f"🔍 Retrieved chunks ({len(candidates)}) — retrieval visualization"):
        df = pd.DataFrame(candidates)
        cols = [c for c in ["filename", "page_label", "score", "rerank_score", "snippet"] if c in df]
        st.dataframe(df[cols], width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Sidebar — status + retrieval settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📚 PDF RAG")
    store = get_store()
    chunk_count = store.count()
    st.metric("Chunks indexed", f"{chunk_count:,}")
    st.caption(f"**Vector DB:** Qdrant — {store.mode}")
    st.caption(f"**Embedder:** {config.EMBEDDING_MODEL} ({config.EMBEDDING_DIM}-d)")
    st.caption(f"**Reranker:** {config.RERANKER_MODEL}")
    st.caption("**LLM:** Groq (open-source Llama)")

    st.divider()
    st.subheader("Settings")
    model = st.selectbox("LLM model (Groq)", [config.GROQ_MODEL, config.GROQ_MODEL_QUALITY], index=0,
                         help="8b-instant: faster + more rate-limit resilient. "
                              "70b-versatile: higher-quality synthesis.")
    top_k = st.slider("Candidates (top-K)", 5, 30, config.TOP_K_RETRIEVE)
    final_k = st.slider("Context chunks", 1, 10, config.TOP_K_RERANK)
    use_rr = st.toggle("Use reranker", value=True)
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()


tab_chat, tab_ingest, tab_about = st.tabs(["💬 Chat", "📥 Ingestion pipeline", "ℹ️ About"])

# --------------------------------------------------------------------------- #
# Chat tab
# --------------------------------------------------------------------------- #
with tab_chat:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if m.get("timings"):
                    render_latency(m["timings"])
                render_sources(m.get("sources", []))
                render_retrieved(m.get("candidates", []))

    if prompt := st.chat_input("Ask a question about the PDFs…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if get_store().count() == 0:
                st.warning("No documents indexed yet. Go to the **Ingestion pipeline** tab and ingest PDFs first.")
                st.stop()

            with st.status("Retrieving relevant passages…", expanded=False) as status:
                r = pipeline.retrieve(prompt, top_k=top_k, final_k=final_k, use_reranker=use_rr)
                status.update(label=f"Retrieved {len(r['candidates'])} candidates "
                                    f"→ {len(r['final'])} after rerank", state="complete")

            t0 = time.perf_counter()
            answer = st.write_stream(llm.stream_answer(prompt, r["contexts"], model=model))
            llm_ms = round((time.perf_counter() - t0) * 1000, 1)

            timings = dict(r["timings"])
            timings["llm_ms"] = llm_ms
            timings["total_ms"] = round(
                timings["embed_ms"] + timings["search_ms"] + timings["rerank_ms"] + llm_ms, 1)

            sources = [pipeline._view(c) for c in r["final"]]
            candidates = [pipeline._view(c) for c in r["candidates"]]
            render_latency(timings)
            render_sources(sources)
            render_retrieved(candidates)

        st.session_state.messages.append({
            "role": "assistant", "content": answer,
            "timings": timings, "sources": sources, "candidates": candidates,
        })

# --------------------------------------------------------------------------- #
# Ingestion tab
# --------------------------------------------------------------------------- #
with tab_ingest:
    st.subheader("Document ingestion pipeline")
    st.markdown(
        "```\nPDF  →  extract text (PyMuPDF) + OCR scanned pages (RapidOCR)  →  "
        "clean / de-hyphenate / strip headers  →  chunk (480 tok, 80 overlap)  →  "
        "embed (bge-base-en-v1.5)  →  upsert into Qdrant (HNSW, cosine)\n```"
    )

    pdf_dir = config.PDF_DIR
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    ingested = get_store().list_files()

    colA, colB = st.columns(2)
    with colA:
        st.metric("PDFs on disk", len(pdfs))
    with colB:
        st.metric("Files indexed", len(ingested))

    if pdfs:
        rows = [{"file": p.name, "size_MB": round(p.stat().st_size / 1e6, 1),
                 "chunks_indexed": ingested.get(p.name, 0)} for p in pdfs]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No PDFs found in `data/pdfs`. Run `python -m scripts.download_corpus` first.")

    st.divider()
    selected = st.multiselect("PDFs to ingest", [p.name for p in pdfs],
                              default=[p.name for p in pdfs])
    recreate = st.checkbox("Rebuild collection from scratch (clears the index)", value=False)
    st.caption("Tip: large textbooks (200-470 MB) take a few minutes to embed on CPU. "
               "For the full corpus, `python -m scripts.ingest` from a terminal is faster and "
               "won't block the UI.")

    if st.button("🚀 Ingest selected", type="primary", disabled=not selected):
        emb = get_embedder()
        store = get_store()
        store.ensure_collection(dim=emb.dim, recreate=recreate)
        prog = st.progress(0.0)
        logbox = st.empty()
        for i, name in enumerate(selected, 1):
            if not recreate and store.filename_exists(name):
                logbox.info(f"{name}: already ingested — skipping")
            else:
                logbox.info(f"Ingesting **{name}** … (extract → chunk → embed)")
                s = ingest_pdf(pdf_dir / name, emb, store)
                logbox.success(f"{name}: {s['pages']} pages, {s['ocr_pages']} OCR'd, "
                               f"{s['chunks']} chunks in {s['seconds']}s")
            prog.progress(i / len(selected))
        st.success(f"Done. The store now holds {store.count():,} chunks.")
        st.rerun()

# --------------------------------------------------------------------------- #
# About tab
# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("How it works")
    st.markdown(
        f"""
This is a Retrieval-Augmented Generation (RAG) chatbot over a private PDF corpus.

**Open-source stack (no paid components for retrieval):**
- **Embeddings:** `{config.EMBEDDING_MODEL}` (BGE, {config.EMBEDDING_DIM}-d) via **fastembed / onnxruntime**
- **Vector DB:** **Qdrant** (embedded, HNSW index, cosine) — metadata payload + scores per chunk
- **Reranker:** `{config.RERANKER_MODEL}` (cross-encoder)
- **OCR:** **RapidOCR** (onnxruntime) for scanned / image-only pages
- **LLM (answer synthesis):** **Groq** serving open-source Llama (`{config.GROQ_MODEL}`)

**Per query:** embed → Qdrant ANN search (top-{config.TOP_K_RETRIEVE}) → cross-encoder rerank
→ top-{config.TOP_K_RERANK} chunks → grounded, cited answer. Target latency **2–5 s**.

**Provenance:** every chunk keeps its `filename`, `page` and text bbox, and every answer cites
sources as `[filename p.PAGE]`.

> Note: on this machine, Windows **Smart App Control** blocks SciPy's DLLs, so the embedder/reranker
> run on **onnxruntime** (fastembed) instead of PyTorch — same models, no SciPy.
"""
    )
