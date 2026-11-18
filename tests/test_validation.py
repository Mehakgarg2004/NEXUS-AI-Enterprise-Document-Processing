import random
from datetime import date

from nexus.validation import gstin_checksum_ok, validate
from training.synthetic import make_gstin


def codes(issues):
    return {i.code for i in issues}


def test_gstin_checksum():
    assert gstin_checksum_ok("27AAPFU0939F1ZV")
    assert not gstin_checksum_ok("27AAPFU0939F1ZX")
    rng = random.Random(0)
    assert all(gstin_checksum_ok(make_gstin(rng)) for _ in range(50))


def test_valid_invoice_has_no_errors():
    f = {"invoice_number": "A1", "invoice_date": "2025-01-01", "due_date": "2025-01-31",
         "subtotal": 100.0, "tax": 18.0, "total": 118.0, "gstin": "27AAPFU0939F1ZV"}
    assert not [i for i in validate("invoice", f, today=date(2025, 6, 1)) if i.severity == "error"]


def test_invoice_errors():
    f = {"invoice_number": None, "invoice_date": "2025-02-01", "due_date": "2025-01-01",
         "subtotal": 100.0, "tax": 18.0, "total": 500.0, "gstin": "27AAPFU0939F1ZX"}
    c = codes(validate("invoice", f, today=date(2025, 6, 1)))
    assert {"MISSING_FIELD", "TOTAL_MISMATCH", "DUE_BEFORE_ISSUE", "INVALID_GSTIN"} <= c


def test_future_date_and_low_ocr():
    c = codes(validate("receipt", {"date": "2099-01-01", "total": 10.0}, ocr_confidence=40, today=date(2025, 1, 1)))
    assert {"FUTURE_DATE", "LOW_OCR_CONFIDENCE"} <= c


def test_bank_balance_mismatch():
    f = {"account_number": "X1", "opening_balance": 100.0, "closing_balance": 999.0, "total_credits": 50.0, "total_debits": 20.0}
    assert "BALANCE_MISMATCH" in codes(validate("bank_statement", f))


def test_unknown_type_flagged():
    assert "UNKNOWN_TYPE" in codes(validate("unknown", {}))
