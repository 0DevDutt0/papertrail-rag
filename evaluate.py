r"""Evaluation harness for the RAG chatbot.

Measures, over a small labeled QA set (``data/eval/qa_testset.json``):
  - latency  : p50 / p95 / max of end-to-end answer time, % within the 5 s budget
  - retrieval: Recall@k and MRR (a retrieved chunk counts as relevant if it
               contains the question's answer keyword(s))
  - citations: citation accuracy (does the answer cite a page that actually
               contains the answer?)
  - grounding: answer-keyword accuracy, abstention rate, and an optional
               LLM-as-judge hallucination check (--judge)

Run:  python evaluate.py            (fast, deterministic)
      python evaluate.py --judge    (adds Groq LLM-as-judge grounding check)
      python evaluate.py --save     (also writes data/eval/results/eval_report.md)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from rag import llm, pipeline
from rag.config import EVAL_DIR, LATENCY_BUDGET_MAX_S, TOP_K_RERANK, TOP_K_RETRIEVE
from rag.llm import ABSTAIN

CITE_RE = re.compile(r"p\.\s*(\d+)(?:\s*[-–]\s*(\d+))?")


def kw_match(text: str, kws: list[str], min_match: int = 1) -> bool:
    t = (text or "").lower()
    return sum(1 for k in kws if k.lower() in t) >= min_match


def cited_pages(answer: str) -> set[int]:
    pages: set[int] = set()
    for m in CITE_RE.finditer(answer):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        pages.update(range(a, b + 1))
    return pages


def judge_grounded(question: str, contexts: list[dict], answer: str) -> bool:
    """LLM-as-judge: is the answer fully supported by the context? (best-effort)."""
    ctx = "\n\n".join(f"[{c.get('filename')} {c.get('page_label')}]\n{c.get('text','')}"
                      for c in contexts)
    prompt = [
        {"role": "system", "content": "You are a strict grader. Reply with only 'yes' or 'no'."},
        {"role": "user", "content": (
            f"Context:\n{ctx}\n\nAnswer:\n{answer}\n\n"
            "Is every factual claim in the Answer supported by the Context? Reply yes or no.")},
    ]
    try:
        from rag.llm import _groq_client
        from rag.config import GROQ_MODEL
        resp = _groq_client().chat.completions.create(
            model=GROQ_MODEL, messages=prompt, temperature=0.0, max_tokens=4)
        return resp.choices[0].message.content.strip().lower().startswith("y")
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    ap.add_argument("--testset", default=str(EVAL_DIR / "qa_testset.json"))
    ap.add_argument("--top-k", type=int, default=TOP_K_RETRIEVE)
    ap.add_argument("--final-k", type=int, default=TOP_K_RERANK)
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge grounding check")
    ap.add_argument("--save", action="store_true", help="write a markdown report")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="seconds to pause between questions to avoid LLM free-tier "
                         "rate-limiting inflating the measured latency (not counted in latency)")
    args = ap.parse_args()

    items = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    print(f"Loaded {len(items)} questions. Warming models…")
    pipeline.warmup()

    latencies: list[float] = []          # end-to-end (incl. external LLM)
    ret_latencies: list[float] = []      # retrieval only (embed+search+rerank)
    recall_hits = rr_sum = cite_ok = ans_ok = abstain = overbudget = 0
    judged_ok = judged_total = 0
    detail = []

    for it in items:
        q = it["question"]
        kws = it.get("answer_keywords", [])

        r = pipeline.retrieve(q, top_k=args.top_k, final_k=args.final_k)
        rt = r["timings"]
        ret_ms = rt["embed_ms"] + rt["search_ms"] + rt["rerank_ms"]
        t1 = time.perf_counter()
        ans = llm.answer(q, r["contexts"])
        llm_ms = (time.perf_counter() - t1) * 1000.0
        total_ms = ret_ms + llm_ms
        ret_latencies.append(ret_ms)
        latencies.append(total_ms)

        cands = r["candidates"]
        rels = [kw_match(c.get("text", ""), kws) for c in cands]
        hit = any(rels)
        recall_hits += int(hit)
        rank = next((i + 1 for i, rel in enumerate(rels) if rel), 0)
        rr_sum += (1.0 / rank) if rank else 0.0

        relevant_pages = {p for c, rel in zip(cands, rels) if rel for p in c.get("pages", [])}
        cpages = cited_pages(ans)
        cite_hit = bool(cpages & relevant_pages)
        cite_ok += int(cite_hit)

        is_abstain = ABSTAIN.lower()[:30] in ans.lower()
        abstain += int(is_abstain)
        a_ok = kw_match(ans, kws)
        ans_ok += int(a_ok)
        overbudget += int(total_ms > LATENCY_BUDGET_MAX_S * 1000)

        if args.judge:
            judged_total += 1
            judged_ok += int(judge_grounded(q, r["contexts"], ans))

        detail.append((q[:54], round(total_ms), hit, round(1.0 / rank, 2) if rank else 0.0,
                       cite_hit, a_ok))
        print(f"  · {q[:60]:60s} {total_ms:6.0f}ms  R@k={int(hit)} cite={int(cite_hit)} kw={int(a_ok)}")
        if args.delay:
            time.sleep(args.delay)  # space requests to respect LLM free-tier limits

    n = len(items)
    lat = np.array(latencies)
    ret = np.array(ret_latencies)
    lines = [
        "# RAG evaluation report",
        "",
        f"- Questions: **{n}**  |  retrieve top-k=**{args.top_k}**, context=**{args.final_k}**",
        "",
        "## Latency",
        f"- **Retrieval pipeline** (embed+search+rerank): p50 **{np.percentile(ret,50)/1000:.2f} s** | "
        f"p95 **{np.percentile(ret,95)/1000:.2f} s**",
        f"- **End-to-end** (incl. Groq LLM): p50 **{np.percentile(lat,50)/1000:.2f} s** | "
        f"p95 **{np.percentile(lat,95)/1000:.2f} s** | max **{lat.max()/1000:.2f} s**",
        f"- within {LATENCY_BUDGET_MAX_S:.0f}s budget: **{(n-overbudget)}/{n}** "
        f"({100*(n-overbudget)/n:.0f}%)  _(end-to-end is subject to Groq free-tier "
        "rate-limiting on sustained bursts; a single/spaced query is ~2-3 s)_",
        "",
        "## Retrieval",
        f"- Recall@{args.top_k}: **{recall_hits/n:.2f}**",
        f"- MRR: **{rr_sum/n:.2f}**",
        "",
        "## Answer quality",
        f"- Citation accuracy (cited a relevant page): **{cite_ok/n:.2f}**",
        f"- Answer-keyword accuracy: **{ans_ok/n:.2f}**",
        f"- Abstention rate: **{abstain/n:.2f}**",
    ]
    if args.judge and judged_total:
        lines.append(f"- LLM-judge grounded (no hallucination): **{judged_ok/judged_total:.2f}**")

    report = "\n".join(lines)
    print("\n" + report)

    if args.save:
        out_dir = EVAL_DIR / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_report.md").write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved report to {out_dir / 'eval_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
