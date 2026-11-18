import random

from nexus.extraction import extract_entities, extract_fields, find_gstins, normalise_date
from training.synthetic import invoice, make_gstin

INVOICE = """Acme Technologies Pvt Ltd
12 MG Road, Pune
GSTIN: 27AAPFU0939F1ZV

TAX INVOICE
Invoice No: INV-2025-0042
Invoice Date: 05/03/2025
Due Date: 04 Apr 2025
Bill To: Zenith Logistics

Subtotal: ₹1,00,000.00
GST (18%): ₹18,000.00
Total Amount: ₹1,18,000.00
"""


def test_invoice_fields():
    f = extract_fields(INVOICE, "invoice")
    assert f["invoice_number"] == "INV-2025-0042"
    assert f["invoice_date"] == "2025-03-05"
    assert f["due_date"] == "2025-04-04"
    assert f["customer_name"] == "Zenith Logistics"
    assert f["subtotal"] == 100000.0
    assert f["tax"] == 18000.0  # not 18 from "(18%)"
    assert f["total"] == 118000.0
    assert f["gstin"] == "27AAPFU0939F1ZV"
    assert f["currency"] == "INR"


def test_ocr_spacing_in_amounts_is_repaired():
    f = extract_fields("Invoice No: A1\nSubtotal: Rs. 1,000.00\nGST (18%): Rs.180, 000.00\nTotal: 1,81,000.00", "invoice")
    assert f["tax"] == 180000.0


def test_generated_invoices_extract_consistently():
    rng = random.Random(1)
    for _ in range(25):
        f = extract_fields(invoice(rng), "invoice")
        assert f["invoice_number"].startswith("INV-")
        assert abs(f["subtotal"] + f["tax"] - f["total"]) < 1.0


def test_gstin_ocr_confusions_fixed():
    g = make_gstin(random.Random(3))
    garbled = g[:2] + g[2:7].replace("O", "0").replace("S", "5") + g[7:]
    assert g in find_gstins(f"GSTIN: {garbled}")


def test_dates():
    assert normalise_date("12/08/2024") == "2024-08-12"
    assert normalise_date("Mar 5, 2025") == "2025-03-05"
    assert normalise_date("2025-01-31") == "2025-01-31"
    assert normalise_date("13/25/2024") is None


def test_resume_and_entities():
    text = "Priya Sharma\npriya. sharma@example.com | +91 9876543210 | linkedin.com/in/priya\nSKILLS: Python, SQL, AWS\n5 years experience"
    f = extract_fields(text, "resume")
    assert f["name"] == "Priya Sharma"
    assert f["email"] == "priya.sharma@example.com"
    assert f["skills"] == ["Python", "SQL", "AWS"]
    assert f["years_experience"] == 5
    assert f["linkedin"].startswith("linkedin.com")
    e = extract_entities(text)
    assert "priya.sharma@example.com" in e["emails"]


def test_bank_statement_and_contract():
    bank = "HDFC Bank\nAccount Number: XXXX1234\nIFSC: HDFC0001234\nOpening Balance: 10,000.00\nTotal Credits: 5,000.00\nTotal Debits: 2,000.00\nClosing Balance: 13,000.00"
    f = extract_fields(bank, "bank_statement")
    assert f["account_number"] == "XXXX1234" and f["ifsc"] == "HDFC0001234" and f["closing_balance"] == 13000.0
    c = extract_fields("SERVICE AGREEMENT\nThis Agreement is made between Foo Ltd (the Client) and Bar LLP (the Provider).\n"
                       "Effective Date: 01/02/2025\n3. Consideration. Total contract value of ₹5,00,000.00.", "contract")
    assert c["party_a"] == "Foo Ltd" and c["party_b"] == "Bar LLP" and c["contract_value"] == 500000.0
