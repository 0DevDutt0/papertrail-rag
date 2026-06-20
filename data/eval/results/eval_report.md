# RAG evaluation report

- Questions: **8**  |  retrieve top-k=**15**, context=**5**

## Latency
- **Retrieval pipeline** (embed+search+rerank): p50 **0.98 s** | p95 **1.11 s**
- **End-to-end** (incl. Groq LLM): p50 **1.84 s** | p95 **2.06 s** | max **2.07 s**
- within 5s budget: **8/8** (100%)  _(end-to-end is subject to Groq free-tier rate-limiting on sustained bursts; a single/spaced query is ~2-3 s)_

## Retrieval
- Recall@15: **1.00**
- MRR: **0.84**

## Answer quality
- Citation accuracy (cited a relevant page): **0.88**
- Answer-keyword accuracy: **1.00**
- Abstention rate: **0.00**
