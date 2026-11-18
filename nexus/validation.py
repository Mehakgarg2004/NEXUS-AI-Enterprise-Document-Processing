"""Automated validation rules for extracted document data."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .extraction import GSTIN_RE, IFSC_RE

REQUIRED_FIELDS = {
    "invoice": ["invoice_number", "invoice_date", "total"],
    "receipt": ["date", "total"],
    "purchase_order": ["po_number", "order_date", "total"],
    "bank_statement": ["account_number", "opening_balance", "closing_balance"],
    "contract": ["party_a", "party_b", "effective_date"],
    "resume": ["name", "email"],
}

GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Issue:
    severity: str  # "error" | "warning" | "info"
    code: str
    message: str


def gstin_checksum_ok(gstin: str) -> bool:
    """Validate the 15th (check) character of an Indian GSTIN."""
    if not GSTIN_RE.fullmatch(gstin or ""):
        return False
    total = 0
    for i, ch in enumerate(gstin[:14]):
        v = GSTIN_CHARS.index(ch) * (2 if i % 2 else 1)
        total += v // 36 + v % 36
    check = GSTIN_CHARS[(36 - total % 36) % 36]
    return check == gstin[14]


def _date(v):
    try:
        return date.fromisoformat(v) if v else None
    except (TypeError, ValueError):
        return None


def validate(doc_type: str, fields: dict, ocr_confidence: float | None = None, today: date | None = None) -> list[Issue]:
    today = today or date.today()
    issues: list[Issue] = []

    if doc_type == "unknown":
        issues.append(Issue("warning", "UNKNOWN_TYPE", "Document type could not be determined confidently; manual review needed."))

    if ocr_confidence is not None and ocr_confidence < 60:
        issues.append(Issue("warning", "LOW_OCR_CONFIDENCE", f"OCR confidence is low ({ocr_confidence:.0f}%). Scan quality may affect accuracy."))

    for f in REQUIRED_FIELDS.get(doc_type, []):
        if fields.get(f) in (None, "", []):
            issues.append(Issue("error", "MISSING_FIELD", f"Required field '{f}' was not found."))

    # Amount sanity
    for k, v in fields.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("years_experience",):
            if v < 0 and "balance" not in k:
                issues.append(Issue("error", "NEGATIVE_AMOUNT", f"'{k}' is negative ({v})."))

    # Date sanity
    for k, v in fields.items():
        if (k.endswith("date") or k in ("date", "period_from", "period_to")) and v:
            d = _date(v)
            if d is None:
                issues.append(Issue("warning", "BAD_DATE", f"'{k}' is not a valid date ({v})."))
            elif d.year < 1990:
                issues.append(Issue("warning", "OLD_DATE", f"'{k}' ({v}) looks implausibly old."))
            elif d > today and k not in ("due_date", "delivery_date"):
                issues.append(Issue("warning", "FUTURE_DATE", f"'{k}' ({v}) is in the future."))

    if doc_type == "invoice":
        sub, tax, total = fields.get("subtotal"), fields.get("tax"), fields.get("total")
        if sub is not None and tax is not None and total is not None and abs(sub + tax - total) > max(1.0, 0.01 * total):
            issues.append(Issue("error", "TOTAL_MISMATCH", f"Subtotal ({sub}) + tax ({tax}) = {sub + tax:.2f} does not match total ({total})."))
        inv, due = _date(fields.get("invoice_date")), _date(fields.get("due_date"))
        if inv and due and due < inv:
            issues.append(Issue("error", "DUE_BEFORE_ISSUE", "Due date is before the invoice date."))
        g = fields.get("gstin")
        if g and not gstin_checksum_ok(g):
            issues.append(Issue("error", "INVALID_GSTIN", f"GSTIN {g} fails the checksum."))

    if doc_type == "purchase_order":
        o, d = _date(fields.get("order_date")), _date(fields.get("delivery_date"))
        if o and d and d < o:
            issues.append(Issue("error", "DELIVERY_BEFORE_ORDER", "Delivery date is before the order date."))

    if doc_type == "bank_statement":
        ob, cb = fields.get("opening_balance"), fields.get("closing_balance")
        cr, dr = fields.get("total_credits"), fields.get("total_debits")
        if None not in (ob, cb, cr, dr) and abs(ob + cr - dr - cb) > 1.0:
            issues.append(Issue("error", "BALANCE_MISMATCH", f"Opening {ob} + credits {cr} - debits {dr} != closing {cb}."))
        ifsc = fields.get("ifsc")
        if ifsc and not IFSC_RE.fullmatch(ifsc):
            issues.append(Issue("error", "INVALID_IFSC", f"IFSC {ifsc} is malformed."))

    if doc_type == "resume":
        email = fields.get("email")
        if email and "@" not in email:
            issues.append(Issue("error", "INVALID_EMAIL", f"Email {email} is malformed."))

    return issues
