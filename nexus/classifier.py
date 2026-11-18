"""Document type classifier.

Primary: a TensorFlow/Keras text model (TextVectorization + embeddings +
dense layers) trained by `training/train_classifier.py` and tracked in MLflow.

Fallback: a transparent keyword-scoring classifier, used when no trained
model is present yet (e.g. fresh clone) so the platform still works.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

DOC_TYPES = ["invoice", "receipt", "purchase_order", "bank_statement", "contract", "resume"]

KEYWORDS: dict[str, list[str]] = {
    "invoice": ["invoice", "bill to", "invoice no", "due date", "gstin", "tax invoice", "amount due", "payment terms"],
    "receipt": ["receipt", "thank you", "paid", "cash", "change", "cashier", "store", "visit again", "card ending"],
    "purchase_order": ["purchase order", "po number", "po no", "vendor", "ship to", "delivery date", "ordered by", "requisition"],
    "bank_statement": ["statement", "account number", "opening balance", "closing balance", "withdrawal", "deposit", "ifsc", "transaction"],
    "contract": ["agreement", "party", "parties", "hereinafter", "terms and conditions", "witness", "governing law", "termination", "clause"],
    "resume": ["resume", "curriculum vitae", "experience", "education", "skills", "projects", "linkedin", "objective", "certifications"],
}


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class KeywordClassifier:
    name = "keyword"

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        t = text.lower()
        scores = {label: float(sum(t.count(k) for k in kws)) for label, kws in KEYWORDS.items()}
        total = sum(scores.values())
        if total == 0:
            probs = {k: 1 / len(DOC_TYPES) for k in DOC_TYPES}
            return "unknown", 0.0, probs
        probs = {k: v / total for k, v in scores.items()}
        best = max(probs, key=probs.get)
        return best, round(probs[best], 4), probs


class TensorFlowClassifier:
    name = "tensorflow"

    def __init__(self, model_dir: str):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf  # imported lazily: heavy

        self.tf = tf
        model_dir = Path(model_dir)
        self.model = tf.keras.models.load_model(model_dir / "model.keras")
        self.labels = json.loads((model_dir / "labels.json").read_text())

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        x = self.tf.constant([_clean(text)[:20000]])
        probs = self.model(x, training=False).numpy()[0]
        mapping = {label: float(round(p, 4)) for label, p in zip(self.labels, probs)}
        best = max(mapping, key=mapping.get)
        return best, mapping[best], mapping


_lock = threading.Lock()
_instance = None


def get_classifier(model_dir: str):
    """Load the TF model once; fall back to keywords if it's not trained yet."""
    global _instance
    with _lock:
        if _instance is None:
            if (Path(model_dir) / "model.keras").exists():
                try:
                    _instance = TensorFlowClassifier(model_dir)
                except Exception:  # noqa: BLE001 - corrupt model / TF missing
                    _instance = KeywordClassifier()
            else:
                _instance = KeywordClassifier()
        return _instance


def reset_classifier():
    global _instance
    with _lock:
        _instance = None


def classify(text: str, model_dir: str, min_confidence: float = 0.35):
    clf = get_classifier(model_dir)
    label, conf, probs = clf.predict(text)
    if conf < min_confidence:
        label = "unknown"
    return label, conf, clf.name, probs
