"""LLM integration: document Q&A (retrieval-augmented) and field extraction.

Supports Anthropic and OpenAI via their REST APIs. With no API key set,
Q&A falls back to extractive answering (returns the most relevant passages),
so the platform remains usable offline.
"""
from __future__ import annotations

import json
import re

import requests

from .config import Config


# ---------- retrieval ----------
def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def top_chunks(question: str, text: str, k: int = 4) -> list[str]:
    chunks = chunk_text(text)
    if len(chunks) <= k:
        return chunks
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(stop_words="english").fit(chunks + [question])
    sims = cosine_similarity(vec.transform([question]), vec.transform(chunks))[0]
    order = sims.argsort()[::-1][:k]
    return [chunks[i] for i in sorted(order)]


# ---------- providers ----------
class LLMError(Exception):
    pass


def llm_available() -> bool:
    p = Config.LLM_PROVIDER.lower()
    return (p == "anthropic" and bool(Config.ANTHROPIC_API_KEY)) or (p == "openai" and bool(Config.OPENAI_API_KEY))


def call_llm(system: str, user: str, max_tokens: int = 800) -> str:
    p = Config.LLM_PROVIDER.lower()
    try:
        if p == "anthropic":
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": Config.ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": Config.LLM_MODEL,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
                timeout=Config.LLM_TIMEOUT,
            )
            r.raise_for_status()
            return "".join(b.get("text", "") for b in r.json().get("content", []))
        if p == "openai":
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {Config.OPENAI_API_KEY}"},
                json={
                    "model": Config.LLM_MODEL,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                },
                timeout=Config.LLM_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        raise LLMError(f"LLM request failed: {e}") from e
    raise LLMError(f"Unknown LLM provider '{Config.LLM_PROVIDER}'")


# ---------- features ----------
QA_SYSTEM = (
    "You are NEXUS AI, an enterprise document assistant. Answer ONLY from the "
    "document excerpts provided. If the answer is not in them, say you cannot "
    "find it in the document. Be concise and quote exact figures."
)


def answer_question(question: str, text: str, doc_meta: dict | None = None) -> dict:
    context = top_chunks(question, text)
    if not context:
        return {"answer": "This document has no readable text.", "sources": [], "mode": "none"}

    if not llm_available():
        return {
            "answer": "LLM not configured — showing the most relevant passages from the document.",
            "sources": context,
            "mode": "extractive",
        }

    meta = json.dumps(doc_meta or {}, default=str)[:2000]
    excerpts = "\n\n".join(f"[Excerpt {i + 1}]\n{c}" for i, c in enumerate(context))
    prompt = f"Document metadata: {meta}\n\n{excerpts}\n\nQuestion: {question}"
    answer = call_llm(QA_SYSTEM, prompt)
    return {"answer": answer.strip(), "sources": context, "mode": "llm"}


EXTRACT_SYSTEM = (
    "You extract structured data from business documents. Reply with a single "
    "JSON object only, no prose. Use null for values you cannot find. Dates "
    "must be ISO format YYYY-MM-DD; amounts must be plain numbers."
)


def llm_extract_fields(text: str, doc_type: str, wanted: list[str]) -> dict:
    """Ask the LLM to fill specific missing fields. Returns {} on any failure."""
    if not wanted or not llm_available():
        return {}
    prompt = f"Document type: {doc_type}\nFields to extract: {', '.join(wanted)}\n\nDocument:\n{text[:12000]}"
    try:
        raw = call_llm(EXTRACT_SYSTEM, prompt, max_tokens=600)
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
        return {k: data.get(k) for k in wanted if data.get(k) not in (None, "")}
    except (LLMError, json.JSONDecodeError, AttributeError):
        return {}
