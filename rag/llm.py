r"""Grounded, cited answer synthesis via Groq (hosted open-source Llama models).

The system prompt forces the model to answer only from the supplied excerpts,
cite every claim as ``[filename p.PAGE]``, and abstain when the context is
insufficient. Answers stream token-by-token. On error we fall back from the
primary Groq model to the higher-quality one, then to GLM (Zhipu).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterator

from .config import (GLM_API_KEY, GLM_BASE_URL, GLM_MODEL, GROQ_API_KEY, GROQ_MODEL,
                     GROQ_MODEL_QUALITY, LLM_MAX_TOKENS, LLM_TEMPERATURE)

SYSTEM_PROMPT = (
    "You are a precise assistant that answers strictly from the provided document "
    "excerpts.\n"
    "Rules:\n"
    "- Use ONLY the information in the excerpts. Do not use outside knowledge.\n"
    "- After each claim, cite the source by copying the exact bracketed tag shown "
    "above the excerpt you used, e.g. [report.pdf p.12].\n"
    '- If the excerpts do not contain the answer, reply exactly: "I don\'t have enough '
    'information in the documents to answer that."\n'
    "- Be concise; quote names, numbers and terms exactly as written."
)

ABSTAIN = "I don't have enough information in the documents to answer that."


def build_messages(question: str, contexts: list[dict]) -> list[dict]:
    blocks = []
    for c in contexts:
        tag = f"[{c.get('filename', '?')} {c.get('page_label', '')}]".replace("  ", " ")
        blocks.append(f"{tag}\n{c.get('text', '')}")
    context_str = "\n\n".join(blocks) if blocks else "(no excerpts retrieved)"
    user = (
        "Document excerpts (each preceded by its citation tag in brackets):\n\n"
        f"{context_str}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above. After each claim, cite the exact "
        "bracketed tag(s) you used."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


@lru_cache(maxsize=1)
def _groq_client():
    from groq import Groq

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    # Fail fast on rate-limit (no long internal back-off) so we fall back to the
    # next model / GLM quickly instead of stalling the demo for ~20 s.
    return Groq(api_key=GROQ_API_KEY, max_retries=0, timeout=30.0)


def stream_answer(question: str, contexts: list[dict], *, model: str | None = None) -> Iterator[str]:
    """Yield answer tokens. Falls back across models/providers on failure."""
    messages = build_messages(question, contexts)
    models = [model] if model else [GROQ_MODEL, GROQ_MODEL_QUALITY]
    last_err: Exception | None = None
    for m in models:
        try:
            client = _groq_client()
            stream = client.chat.completions.create(
                model=m, messages=messages, temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS, stream=True,
            )
            produced = False
            for event in stream:
                delta = event.choices[0].delta.content
                if delta:
                    produced = True
                    yield delta
            if produced:
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    # Secondary provider: GLM / Zhipu (non-streaming, yielded as one block)
    try:
        yield _glm_answer(messages)
        return
    except Exception as e:  # noqa: BLE001
        last_err = e
    yield f"\n\n[LLM unavailable: {last_err}]"


def answer(question: str, contexts: list[dict], *, model: str | None = None) -> str:
    return "".join(stream_answer(question, contexts, model=model))


def _glm_answer(messages: list[dict]) -> str:
    """Fallback synthesis via GLM (Zhipu) OpenAI-compatible REST endpoint."""
    import requests

    if not GLM_API_KEY:
        raise RuntimeError("GLM_API_KEY is not set")
    resp = requests.post(
        GLM_BASE_URL,
        headers={"Authorization": f"Bearer {GLM_API_KEY}", "Content-Type": "application/json"},
        json={"model": GLM_MODEL, "messages": messages,
              "temperature": LLM_TEMPERATURE, "max_tokens": LLM_MAX_TOKENS},
        timeout=40,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
