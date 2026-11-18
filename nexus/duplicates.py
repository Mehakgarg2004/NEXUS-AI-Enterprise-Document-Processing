"""Duplicate detection: exact (file hash), near-duplicate (text shingling +
Jaccard similarity) and business-key duplicates (e.g. same invoice number
from the same vendor)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass
class DuplicateMatch:
    document_id: int
    score: float  # 1.0 = identical
    kind: str  # "exact" | "near" | "business_key"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shingles(text: str, k: int = 5) -> set[int]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < k:
        return {hash(" ".join(words))} if words else set()
    return {hash(" ".join(words[i : i + k])) for i in range(len(words) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


BUSINESS_KEYS = {
    "invoice": ("invoice_number", "vendor_name"),
    "purchase_order": ("po_number", "vendor_name"),
    "receipt": ("receipt_number", "merchant"),
}


def _norm(v) -> str:
    return re.sub(r"\W+", "", str(v).lower()) if v is not None else ""


def find_duplicate(
    file_hash: str,
    text: str,
    doc_type: str,
    fields: dict,
    candidates: list,  # existing Document rows
    threshold: float = 0.85,
) -> DuplicateMatch | None:
    # 1) exact binary duplicate
    for c in candidates:
        if c.sha256 == file_hash:
            return DuplicateMatch(c.id, 1.0, "exact")

    # 2) business-key duplicate (same invoice no. + vendor, re-scanned or re-sent)
    keys = BUSINESS_KEYS.get(doc_type)
    if keys and all(fields.get(k) for k in keys):
        target = tuple(_norm(fields[k]) for k in keys)
        for c in candidates:
            if c.doc_type == doc_type and c.fields and tuple(_norm(c.fields.get(k)) for k in keys) == target:
                return DuplicateMatch(c.id, 1.0, "business_key")

    # 3) near-duplicate text (same document, different scan/format)
    sh = shingles(text)
    best: DuplicateMatch | None = None
    for c in candidates:
        if not c.text:
            continue
        s = jaccard(sh, shingles(c.text))
        if s >= threshold and (best is None or s > best.score):
            best = DuplicateMatch(c.id, round(s, 4), "near")
    return best
