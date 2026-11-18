"""Anomaly identification.

For each document type we build a numeric feature vector and compare the
new document against historical documents of the same type using:
  * an Isolation Forest (multivariate outliers), and
  * a robust z-score (median/MAD) on the main amount (explainable reason).
Below a minimum history size, only simple rule checks are applied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

AMOUNT_FIELD = {
    "invoice": "total",
    "receipt": "total",
    "purchase_order": "total",
    "bank_statement": "closing_balance",
    "contract": "contract_value",
}


@dataclass
class AnomalyResult:
    is_anomaly: bool
    score: float  # 0..1, higher = more anomalous
    reasons: list[str] = field(default_factory=list)


def _d(v):
    try:
        return date.fromisoformat(v) if v else None
    except (TypeError, ValueError):
        return None


def features(doc_type: str, fields: dict) -> list[float] | None:
    amt = fields.get(AMOUNT_FIELD.get(doc_type, ""), None)
    if not isinstance(amt, (int, float)):
        return None
    f = [math.log1p(max(amt, 0))]
    if doc_type == "invoice":
        tax = fields.get("tax") or 0.0
        f.append(tax / amt if amt else 0.0)
        a, b = _d(fields.get("invoice_date")), _d(fields.get("due_date"))
        f.append(float((b - a).days) if a and b else 30.0)
    if doc_type == "bank_statement":
        ob = fields.get("opening_balance") or 0.0
        f.append(math.log1p(abs(amt - ob)))
    return f


def robust_z(value: float, history: list[float]) -> float:
    arr = np.asarray(history, dtype=float)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med)) or (np.std(arr) or 1.0)
    return float(0.6745 * (value - med) / mad)


def detect(doc_type: str, fields: dict, history_fields: list[dict], min_history: int = 8) -> AnomalyResult:
    reasons: list[str] = []
    amt_key = AMOUNT_FIELD.get(doc_type)
    amt = fields.get(amt_key) if amt_key else None

    # Rule checks that apply even with no history
    if isinstance(amt, (int, float)) and amt > 0 and amt == round(amt, -4) and amt >= 50000:
        reasons.append(f"Suspiciously round {amt_key} ({amt:,.0f}).")
    if doc_type == "invoice" and isinstance(amt, (int, float)) and fields.get("tax") is not None and amt:
        ratio = fields["tax"] / amt
        if ratio > 0.3:
            reasons.append(f"Tax is {ratio:.0%} of total — unusually high.")

    x = features(doc_type, fields)
    hist = [h for h in (features(doc_type, f) for f in history_fields) if h is not None and len(h) == len(x or [])]

    if x is None or len(hist) < min_history:
        score = min(1.0, 0.4 * len(reasons))
        return AnomalyResult(bool(reasons) and score >= 0.4, round(score, 4), reasons)

    from sklearn.ensemble import IsolationForest

    X = np.asarray(hist, dtype=float)
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=42).fit(X)
    raw = -float(model.score_samples(np.asarray([x]))[0])  # ~0.3 normal, >0.6 outlier
    iso_score = max(0.0, min(1.0, (raw - 0.4) / 0.35))

    hist_amounts = [math.expm1(h[0]) for h in hist]
    z = robust_z(amt, hist_amounts)
    if abs(z) > 3.5:
        direction = "higher" if z > 0 else "lower"
        reasons.append(f"{amt_key} {amt:,.2f} is far {direction} than usual (median {np.median(hist_amounts):,.2f}, robust z={z:.1f}).")
    if iso_score > 0.6:
        reasons.append(f"Isolation Forest flags this {doc_type} as an outlier vs {len(hist)} past documents.")

    score = max(iso_score, min(1.0, abs(z) / 7), 0.4 * bool(reasons))
    return AnomalyResult(score >= 0.5 and bool(reasons), round(score, 4), reasons)
