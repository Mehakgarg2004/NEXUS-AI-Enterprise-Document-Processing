"""NLP information extraction.

Rule-based NLP (regex + label proximity + normalisation) extracts structured
fields per document type. When an LLM API key is configured, the LLM is
asked to fill any fields the rules missed (see `llm.llm_extract_fields`).
"""
from __future__ import annotations

import re
from datetime import datetime

# ---------- generic entity patterns ----------
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)|(?<!\d)\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}(?!\d)")
GSTIN_RE = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
IFSC_RE = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
URL_RE = re.compile(r"\bhttps?://[^\s|]+|\bwww\.[^\s|]+|\b(?:[a-z0-9-]+\.)?(?:linkedin|github)\.com/[^\s|]+", re.I)
AMOUNT_RE = re.compile(r"(?:₹|rs\.?|inr|\$|usd|€|eur)?\s*(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", re.I)

DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), "%Y-%m-%d"),
    (re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b"), "%d/%m/%Y"),
    (re.compile(r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*,?\s+(\d{4})\b", re.I), "%d %b %Y"),
    (re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})\b", re.I), "%b %d %Y"),
]


def parse_amount(s: str) -> float | None:
    m = AMOUNT_RE.search(s or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def normalise_date(raw: str) -> str | None:
    raw = raw.strip()
    for rx, fmt in DATE_PATTERNS:
        m = rx.search(raw)
        if not m:
            continue
        s = " ".join(m.groups()).replace("Sept", "Sep")
        f = fmt.replace("/", " ").replace("-", " ").replace(",", "")
        try:
            return datetime.strptime(s, f).date().isoformat()
        except ValueError:
            # try month-first for ambiguous numeric dates (US style)
            if fmt == "%d/%m/%Y":
                try:
                    return datetime.strptime(s, "%m %d %Y").date().isoformat()
                except ValueError:
                    pass
    return None


def find_dates(text: str) -> list[str]:
    out = []
    for rx, _ in DATE_PATTERNS:
        for m in rx.finditer(text):
            d = normalise_date(m.group(0))
            if d and d not in out:
                out.append(d)
    return out


def _labelled(text: str, labels: list[str], value: str = r"([^\n]{1,80})", require_sep: bool = False) -> str | None:
    """Find 'Label: value' (or 'Label value') on a single line."""
    sep = r"[ \t]*[:#\-][ \t]*" if require_sep else r"[ \t]*[:#\-]?[ \t]*"
    for label in labels:
        rx = re.compile(rf"(?im)^[^\n]*?\b{label}\b[ \t]*(?:no\.?|number|#)?{sep}{value}")
        for m in rx.finditer(text):
            v = m.group(1).strip(" :#-\t")
            if v and re.search(r"[A-Za-z0-9]", v):
                return v
    return None


PERCENT_RE = re.compile(r"\(?\s*\d+(?:\.\d+)?\s*%\s*\)?")
LINE_AMOUNT_RE = re.compile(r"(?:₹|rs\.?|inr|\$|usd|€|eur)?\s*(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", re.I)


def _labelled_amount(text: str, labels: list[str]) -> float | None:
    """Amount on a line that starts with/contains the label. Percentages
    such as 'GST (18%)' are ignored; the last number on the line wins."""
    for label in labels:
        rx = re.compile(rf"(?i)^\s*(?:\d{{1,2}}[.)]\s*)?[^\n\d]*\b{label}\b")
        hits = []
        for line in text.splitlines():
            if rx.search(line):
                rest = PERCENT_RE.sub(" ", line[rx.search(line).end():])
                nums = LINE_AMOUNT_RE.findall(rest)
                if nums:
                    hits.append(nums[-1])
        if hits:
            try:
                return float(hits[-1].replace(",", ""))  # last occurrence = usually the final total
            except ValueError:
                continue
    return None


# OCR often confuses letters/digits; GSTIN has fixed letter/digit positions.
_TO_LETTER = str.maketrans("01258", "OIZSB")
_TO_DIGIT = str.maketrans("OIZSBL", "012581")
GSTIN_LOOSE_RE = re.compile(r"\b[0-9OIZSBL]{2}[A-Z0-9]{5}[0-9OIZSBL]{4}[A-Z0-9][A-Z0-9]Z[A-Z0-9]\b")


def find_gstins(text: str) -> list[str]:
    out = []
    for raw in GSTIN_LOOSE_RE.findall(text.upper()):
        fixed = (raw[:2].translate(_TO_DIGIT) + raw[2:7].translate(_TO_LETTER) + raw[7:11].translate(_TO_DIGIT)
                 + raw[11].translate(_TO_LETTER) + raw[12:])
        if GSTIN_RE.fullmatch(fixed) and fixed not in out:
            out.append(fixed)
    return out


def _labelled_date(text: str, labels: list[str]) -> str | None:
    v = _labelled(text, labels)
    return normalise_date(v) if v else None


def _id_token(v: str | None) -> str | None:
    if not v:
        return None
    m = re.match(r"([A-Za-z0-9][A-Za-z0-9/_\-.]*)", v.strip())
    return m.group(1) if m else None


def _first_line(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 2 and not re.search(r"invoice|receipt|statement|purchase order|resume|curriculum", line, re.I):
            return line[:120]
    return None


def detect_currency(text: str) -> str | None:
    t = text.lower()
    if "₹" in text or re.search(r"\binr\b|\brs\.?\s*\d", t):
        return "INR"
    if "$" in text or "usd" in t:
        return "USD"
    if "€" in text or "eur" in t:
        return "EUR"
    return None


# ---------- per-type extractors ----------
def extract_invoice(text: str) -> dict:
    return {
        "invoice_number": _id_token(_labelled(text, ["invoice no", "invoice number", "invoice #", "inv no", "invoice"], require_sep=True)),
        "invoice_date": _labelled_date(text, ["invoice date", "date of issue", "date"]),
        "due_date": _labelled_date(text, ["due date", "payment due"]),
        "vendor_name": _labelled(text, ["from", "vendor", "seller", "supplier"]) or _first_line(text),
        "customer_name": _labelled(text, ["bill to", "billed to", "customer"]),
        "subtotal": _labelled_amount(text, ["subtotal", "sub total", "taxable value"]),
        "tax": _labelled_amount(text, ["gst", "tax", "vat", "igst", "cgst"]),
        "total": _labelled_amount(text, ["grand total", "total amount", "amount due", "total"]),
        "gstin": (find_gstins(text) or [None])[0],
        "currency": detect_currency(text),
    }


def extract_receipt(text: str) -> dict:
    return {
        "merchant": _first_line(text),
        "receipt_number": _id_token(_labelled(text, ["receipt no", "receipt number", "receipt #", "bill no", "txn id", "transaction id"], require_sep=True)),
        "date": _labelled_date(text, ["date"]) or (find_dates(text) or [None])[0],
        "total": _labelled_amount(text, ["grand total", "total", "amount paid", "paid"]),
        "payment_method": _labelled(text, ["payment method", "paid by", "payment mode"]),
        "currency": detect_currency(text),
    }


def extract_purchase_order(text: str) -> dict:
    return {
        "po_number": _id_token(_labelled(text, ["po number", "po no", "po #", "purchase order no"], require_sep=True)),
        "order_date": _labelled_date(text, ["order date", "po date", "date"]),
        "delivery_date": _labelled_date(text, ["delivery date", "deliver by", "required by"]),
        "vendor_name": _labelled(text, ["vendor", "supplier"]),
        "ship_to": _labelled(text, ["ship to", "deliver to"]),
        "total": _labelled_amount(text, ["grand total", "total amount", "total"]),
        "currency": detect_currency(text),
    }


def extract_bank_statement(text: str) -> dict:
    acct = _labelled(text, ["account number", "account no", "a/c no"], r"([Xx\d*][Xx\d* -]{3,23})")
    return {
        "bank_name": _first_line(text),
        "account_holder": _labelled(text, ["account holder", "customer name", "name"]),
        "account_number": acct.replace(" ", "") if acct else None,
        "ifsc": (IFSC_RE.findall(text) or [None])[0],
        "period_from": _labelled_date(text, ["from", "period from", "statement from"]),
        "period_to": _labelled_date(text, ["to", "period to"]),
        "opening_balance": _labelled_amount(text, ["opening balance"]),
        "closing_balance": _labelled_amount(text, ["closing balance"]),
        "total_credits": _labelled_amount(text, ["total credits", "total deposits"]),
        "total_debits": _labelled_amount(text, ["total debits", "total withdrawals"]),
        "currency": detect_currency(text),
    }


def extract_contract(text: str) -> dict:
    parties = re.findall(r"(?i)between\s+(.+?)\s+(?:\(.*?\)\s*)?and\s+(.+?)(?:\s*\(|[.,\n])", text)
    return {
        "title": _first_line(text),
        "party_a": parties[0][0].strip() if parties else None,
        "party_b": parties[0][1].strip() if parties else None,
        "effective_date": _labelled_date(text, ["effective date", "dated", "commencement date"]) or (find_dates(text) or [None])[0],
        "term": (re.search(r"(?i)\bterm of ([^.,\n]{2,40})", text) or [None, None])[1] or _labelled(text, ["duration"], require_sep=True),
        "governing_law": _labelled(text, ["governing law", "governed by the laws of"]),
        "contract_value": _labelled_amount(text, ["contract value", "consideration", "fee", "total value"]),
    }


def extract_resume(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name = None
    for line in lines[:5]:
        if 1 < len(line.split()) <= 4 and not re.search(r"@|\d|resume|curriculum", line, re.I):
            name = line
            break
    skills_line = _labelled(text, ["technical skills", "skills"], r"([^\n]{1,300})", require_sep=True)
    skills = [s.strip() for s in re.split(r"[,|•;]", skills_line)] if skills_line else []
    years = re.findall(r"(\d{1,2})\+?\s+years", text, re.I)
    return {
        "name": name,
        "email": (EMAIL_RE.findall(text) or [None])[0],
        "phone": (PHONE_RE.findall(text) or [None])[0],
        "skills": [s for s in skills if s][:30],
        "years_experience": max(map(int, years)) if years else None,
        "linkedin": next((u for u in URL_RE.findall(text) if "linkedin" in u.lower()), None),
        "github": next((u for u in URL_RE.findall(text) if "github" in u.lower()), None),
    }


EXTRACTORS = {
    "invoice": extract_invoice,
    "receipt": extract_receipt,
    "purchase_order": extract_purchase_order,
    "bank_statement": extract_bank_statement,
    "contract": extract_contract,
    "resume": extract_resume,
}


def extract_entities(text: str) -> dict:
    """Generic entities found in any document."""
    text = normalise_ocr_text(text)
    return {
        "emails": sorted(set(EMAIL_RE.findall(text)))[:20],
        "phones": sorted(set(p.strip() for p in PHONE_RE.findall(text)))[:20],
        "dates": find_dates(text)[:20],
        "gstins": find_gstins(text),
        "pans": sorted(set(PAN_RE.findall(text)) - {g[2:12] for g in find_gstins(text)}),
        "urls": sorted(set(URL_RE.findall(text)))[:20],
    }


def normalise_ocr_text(text: str) -> str:
    """Repair common OCR artefacts before extraction."""
    text = re.sub(r"(\d),\s+(\d{2,3}\b)", r"\1,\2", text)  # "110, 466.62" -> "110,466.62"
    text = re.sub(r"(\d)\s+\.\s*(\d{2})\b", r"\1.\2", text)  # "466 .62" -> "466.62"
    text = text.replace("₹ ", "₹")
    text = re.sub(r"(\w)\.\s+(\w[\w.+-]*@)", r"\1.\2", text)  # "rohan. gupta@x.com"
    text = re.sub(r"(\w)[ \t]*@[ \t]*(\w)", r"\1@\2", text)  # "name @ example.com"
    text = re.sub(r"\.[ \t]+(com|in|org|net|io)\b", r".\1", text)  # "linkedin. com"
    return text


def extract_fields(text: str, doc_type: str) -> dict:
    fn = EXTRACTORS.get(doc_type)
    return fn(normalise_ocr_text(text)) if fn else {}
