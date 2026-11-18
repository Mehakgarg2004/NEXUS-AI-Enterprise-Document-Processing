import random
from types import SimpleNamespace

from nexus.anomaly import detect
from nexus.duplicates import find_duplicate, jaccard, sha256, shingles
from nexus.extraction import extract_fields
from training.synthetic import add_noise, invoice


def doc(id, text, data=b"", doc_type="invoice", fields=None):
    return SimpleNamespace(id=id, text=text, sha256=sha256(data or text.encode()), doc_type=doc_type, fields=fields or {})


def test_exact_duplicate():
    t = "hello world invoice"
    m = find_duplicate(sha256(t.encode()), t, "invoice", {}, [doc(1, t)])
    assert m.kind == "exact" and m.document_id == 1


def test_near_duplicate_survives_ocr_noise():
    rng = random.Random(5)
    original = invoice(rng)
    noisy = add_noise(random.Random(9), original, 0.01)
    assert jaccard(shingles(original), shingles(noisy)) > 0.6
    m = find_duplicate(sha256(noisy.encode()), noisy, "unknown", {}, [doc(7, original)], threshold=0.6)
    assert m and m.kind == "near" and m.document_id == 7


def test_business_key_duplicate():
    f = {"invoice_number": "INV-1", "vendor_name": "Acme Ltd"}
    m = find_duplicate("x", "totally different text", "invoice", {"invoice_number": "inv-1", "vendor_name": "ACME LTD"},
                       [doc(3, "other", fields=f)])
    assert m.kind == "business_key"


def test_not_duplicate():
    rng = random.Random(1)
    a, b = invoice(rng), invoice(rng)
    assert find_duplicate(sha256(b.encode()), b, "invoice", {}, [doc(1, a)]) is None


def _history(n=40):
    rng = random.Random(11)
    return [extract_fields(invoice(rng), "invoice") for _ in range(n)]


def test_normal_invoice_not_anomalous():
    hist = _history()
    rng = random.Random(99)
    flagged = sum(detect("invoice", extract_fields(invoice(rng), "invoice"), hist).is_anomaly for _ in range(20))
    assert flagged <= 2  # allow a small false-positive rate


def test_huge_invoice_is_anomalous():
    hist = _history()
    f = {"total": 9.5e9, "tax": 1.4e9, "subtotal": 8.1e9, "invoice_date": "2025-01-01", "due_date": "2025-01-31"}
    r = detect("invoice", f, hist)
    assert r.is_anomaly and r.score >= 0.5 and r.reasons


def test_rules_without_history():
    r = detect("invoice", {"total": 500000.0, "tax": 200000.0}, [])
    assert r.is_anomaly and len(r.reasons) == 2
