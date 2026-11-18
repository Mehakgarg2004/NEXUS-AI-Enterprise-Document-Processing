"""Synthetic enterprise document generator.

Produces realistic-looking text for each document type, with optional
OCR-style noise (character swaps, dropped words, shuffled lines), so the
classifier learns robust features rather than exact templates.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

COMPANIES = ["Acme Technologies Pvt Ltd", "Zenith Logistics", "BlueOcean Retail", "Sharma Traders", "Infotech Solutions LLP",
             "Global Foods Co", "Northwind Supplies", "Vertex Pharma", "Orbit Electronics", "GreenLeaf Organics",
             "Kaveri Textiles", "Pinnacle Consulting", "Metro Hardware", "Sunrise Motors", "Delta Analytics"]
PEOPLE = ["Aarav Mehta", "Priya Sharma", "Rohan Gupta", "Ananya Iyer", "Vikram Singh", "Neha Kapoor", "Arjun Nair",
          "Sneha Reddy", "Karan Malhotra", "Divya Joshi", "Rahul Verma", "Meera Pillai", "Mehak Garg", "Ishaan Bose"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Chennai", "Hyderabad", "Kolkata", "Jaipur", "Ahmedabad", "Chandigarh"]
ITEMS = ["Laptop", "Office Chair", "Printer Paper A4", "LED Monitor", "Wireless Mouse", "Cloud Hosting (monthly)",
         "Consulting Hours", "Steel Rods", "Cotton Fabric", "Packaging Boxes", "Software License", "Diesel", "Coffee Beans"]
BANKS = [("State Bank of India", "SBIN"), ("HDFC Bank", "HDFC"), ("ICICI Bank", "ICIC"), ("Axis Bank", "UTIB"), ("Kotak Mahindra Bank", "KKBK")]
SKILLS = ["Python", "TensorFlow", "SQL", "Docker", "AWS", "React", "Java", "NLP", "Flask", "PostgreSQL", "Kubernetes", "Excel", "Tableau", "Git"]
GST_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _rand_date(rng: random.Random, start=date(2023, 1, 1), days=1000) -> date:
    return start + timedelta(days=rng.randint(0, days))


def _fmt_date(rng: random.Random, d: date) -> str:
    return rng.choice([d.strftime("%d/%m/%Y"), d.isoformat(), d.strftime("%d %b %Y"), d.strftime("%b %d, %Y")])


def _money(rng: random.Random, v: float) -> str:
    s = f"{v:,.2f}"
    return rng.choice([f"₹{s}", f"Rs. {s}", f"INR {s}", s])


def make_gstin(rng: random.Random) -> str:
    state = f"{rng.randint(1, 37):02d}"
    pan = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)) + f"{rng.randint(0, 9999):04d}" + rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    body = state + pan + str(rng.randint(1, 9)) + "Z"
    total = 0
    for i, ch in enumerate(body):
        v = GST_CHARS.index(ch) * (2 if i % 2 else 1)
        total += v // 36 + v % 36
    return body + GST_CHARS[(36 - total % 36) % 36]


def invoice(rng: random.Random, **over) -> str:
    vendor, cust = rng.sample(COMPANIES, 2)
    d = over.get("date") or _rand_date(rng)
    lines, sub = [], 0.0
    for _ in range(rng.randint(1, 5)):
        q, p = rng.randint(1, 20), round(rng.uniform(100, 15000), 2)
        sub += q * p
        lines.append(f"{rng.choice(ITEMS):<28} {q:>3} x {p:>10,.2f} = {q * p:>12,.2f}")
    sub = round(over.get("subtotal", sub), 2)
    tax = round(over.get("tax", sub * 0.18), 2)
    total = round(over.get("total", sub + tax), 2)
    return "\n".join([
        vendor, f"{rng.randint(1, 300)} MG Road, {rng.choice(CITIES)}", f"GSTIN: {make_gstin(rng)}", "",
        rng.choice(["TAX INVOICE", "INVOICE", "Invoice"]),
        f"Invoice No: {over.get('number', f'INV-{d.year}-{rng.randint(1000, 99999)}')}",
        f"Invoice Date: {_fmt_date(rng, d)}",
        f"Due Date: {_fmt_date(rng, d + timedelta(days=rng.choice([15, 30, 45])))}",
        f"Bill To: {cust}", "", "Description                  Qty   Unit Price        Amount", *lines, "",
        f"Subtotal: {_money(rng, sub)}", f"GST (18%): {_money(rng, tax)}", f"Total Amount: {_money(rng, total)}",
        f"Payment Terms: Net {rng.choice([15, 30, 45])}", "Thank you for your business.",
    ])


def receipt(rng: random.Random, **over) -> str:
    store = rng.choice(["Reliance Fresh", "DMart", "Cafe Coffee Day", "Big Bazaar", "Croma", "Starbucks", "Apollo Pharmacy", "More Supermarket"])
    d = _rand_date(rng)
    items, tot = [], 0.0
    for _ in range(rng.randint(1, 6)):
        p = round(rng.uniform(20, 2500), 2)
        tot += p
        items.append(f"{rng.choice(['Milk 1L', 'Bread', 'Cappuccino', 'Rice 5kg', 'Headphones', 'Paracetamol', 'Eggs (12)', 'Notebook', 'Shampoo']):<20}{p:>10,.2f}")
    tot = round(over.get("total", tot), 2)
    return "\n".join([
        store, f"{rng.choice(CITIES)} Store #{rng.randint(10, 999)}", f"Receipt No: R{rng.randint(100000, 999999)}",
        f"Date: {_fmt_date(rng, d)}  Time: {rng.randint(8, 22)}:{rng.randint(0, 59):02d}", f"Cashier: {rng.choice(PEOPLE).split()[0]}",
        "-" * 30, *items, "-" * 30, f"TOTAL {_money(rng, tot)}",
        f"Payment Method: {rng.choice(['Cash', 'UPI', 'Card ending 4421', 'Credit Card'])}",
        rng.choice(["Thank you! Visit again.", "Thank you for shopping with us", "No returns without receipt"]),
    ])


def purchase_order(rng: random.Random, **over) -> str:
    buyer, vendor = rng.sample(COMPANIES, 2)
    d = _rand_date(rng)
    lines, tot = [], 0.0
    for _ in range(rng.randint(1, 5)):
        q, p = rng.randint(5, 500), round(rng.uniform(50, 8000), 2)
        tot += q * p
        lines.append(f"{rng.choice(ITEMS):<28} Qty {q:<5} @ {p:,.2f}")
    return "\n".join([
        buyer, "PURCHASE ORDER", f"PO Number: PO-{rng.randint(10000, 99999)}", f"Order Date: {_fmt_date(rng, d)}",
        f"Delivery Date: {_fmt_date(rng, d + timedelta(days=rng.randint(7, 60)))}", f"Vendor: {vendor}",
        f"Ship To: {buyer}, Warehouse {rng.randint(1, 9)}, {rng.choice(CITIES)}", f"Ordered By: {rng.choice(PEOPLE)}", "",
        *lines, "", f"Total: {_money(rng, round(over.get('total', tot), 2))}", "Please confirm receipt of this purchase order.",
        "Terms: Delivery must match requisition specifications.",
    ])


def bank_statement(rng: random.Random, **over) -> str:
    bank, code = rng.choice(BANKS)
    start = _rand_date(rng)
    end = start + timedelta(days=30)
    bal = opening = round(rng.uniform(5000, 500000), 2)
    rows, cr, dr = [], 0.0, 0.0
    for i in range(rng.randint(4, 12)):
        amt = round(rng.uniform(100, 60000), 2)
        if rng.random() < 0.45:
            bal += amt; cr += amt
            rows.append(f"{(start + timedelta(days=i * 2)).strftime('%d/%m/%Y')}  NEFT CREDIT {rng.choice(COMPANIES)[:18]:<18} Deposit    {amt:>12,.2f}  {bal:>12,.2f}")
        else:
            bal -= amt; dr += amt
            rows.append(f"{(start + timedelta(days=i * 2)).strftime('%d/%m/%Y')}  UPI/{rng.choice(['AMAZON', 'SWIGGY', 'RENT', 'ELECTRICITY', 'ATM WDL'])}  Withdrawal {amt:>12,.2f}  {bal:>12,.2f}")
    return "\n".join([
        bank, "ACCOUNT STATEMENT", f"Account Holder: {rng.choice(PEOPLE)}",
        f"Account Number: XXXXXX{rng.randint(1000, 9999)}", f"IFSC: {code}0{rng.randint(100000, 999999)}",
        f"Statement From: {start.strftime('%d/%m/%Y')} To: {end.strftime('%d/%m/%Y')}",
        f"Opening Balance: {opening:,.2f}", "", "Date        Transaction Details                      Type        Amount       Balance", *rows, "",
        f"Total Credits: {cr:,.2f}", f"Total Debits: {dr:,.2f}", f"Closing Balance: {round(over.get('closing', bal), 2):,.2f}",
    ])


def contract(rng: random.Random, **over) -> str:
    a, b = rng.sample(COMPANIES, 2)
    kind = rng.choice(["SERVICE AGREEMENT", "NON-DISCLOSURE AGREEMENT", "MASTER SERVICES AGREEMENT", "LEASE AGREEMENT", "CONSULTING AGREEMENT"])
    d = _rand_date(rng)
    return "\n".join([
        kind, "",
        f"This Agreement is made and entered into between {a} (hereinafter referred to as the \"Client\") and {b} (hereinafter referred to as the \"Service Provider\").",
        f"Effective Date: {_fmt_date(rng, d)}", "",
        "1. Scope of Services. The Service Provider shall provide the services described in Schedule A.",
        f"2. Term. This agreement shall remain in force for a term of {rng.choice([6, 12, 24, 36])} months unless terminated earlier.",
        f"3. Consideration. The Client shall pay a total contract value of {_money(rng, rng.uniform(50000, 5000000))}.",
        "4. Confidentiality. Each party shall keep the other party's confidential information strictly confidential.",
        "5. Termination. Either party may terminate this agreement with 30 days written notice.",
        f"6. Governing Law: This agreement shall be governed by the laws of India, courts at {rng.choice(CITIES)}.", "",
        "IN WITNESS WHEREOF the parties have executed this agreement.", f"For {a}: ____________   For {b}: ____________",
    ])


def resume(rng: random.Random, **over) -> str:
    name = rng.choice(PEOPLE)
    first, last = name.lower().split()
    skills = rng.sample(SKILLS, rng.randint(4, 9))
    yrs = rng.randint(0, 12)
    return "\n".join([
        name, f"{first}.{last}@example.com | +91 9{rng.randint(100000000, 999999999)} | linkedin.com/in/{first}{last}", "",
        "OBJECTIVE", "Motivated professional seeking a challenging role to apply my skills.", "",
        "EXPERIENCE", f"{rng.choice(['Software Engineer', 'Data Analyst', 'ML Engineer', 'Accountant'])} — {rng.choice(COMPANIES)} ({yrs} years)",
        "- Built and maintained production systems; improved performance by 30%.", "",
        "EDUCATION", f"B.Tech in Computer Science, {rng.choice(['IIT Delhi', 'NIT Trichy', 'Delhi University', 'VIT Vellore'])}", "",
        f"SKILLS: {', '.join(skills)}", "", "PROJECTS", "- Document processing platform using OCR and NLP.",
        "CERTIFICATIONS", "- AWS Certified Cloud Practitioner",
    ])


GENERATORS = {
    "invoice": invoice, "receipt": receipt, "purchase_order": purchase_order,
    "bank_statement": bank_statement, "contract": contract, "resume": resume,
}


def add_noise(rng: random.Random, text: str, level: float = 0.03) -> str:
    """Simulate OCR errors: char substitutions, dropped words, shuffled lines."""
    subs = {"0": "O", "1": "l", "5": "S", "8": "B", "e": "c", "a": "o", "i": "1", "m": "rn"}
    out = []
    for word in text.split(" "):
        if rng.random() < level:
            continue
        if rng.random() < level * 2:
            word = "".join(subs.get(c, c) if rng.random() < 0.3 else c for c in word)
        out.append(word)
    lines = " ".join(out).split("\n")
    if rng.random() < 0.3 and len(lines) > 4:
        i = rng.randint(1, len(lines) - 2)
        lines[i], lines[i + 1] = lines[i + 1], lines[i]
    if rng.random() < 0.2:  # partial page / truncated scan
        lines = lines[: max(4, int(len(lines) * rng.uniform(0.5, 0.9)))]
    return "\n".join(lines)


def generate(n_per_class: int = 400, seed: int = 7, noise: float = 0.04) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    data = []
    for label, fn in GENERATORS.items():
        for _ in range(n_per_class):
            data.append((add_noise(rng, fn(rng), noise), label))
    rng.shuffle(data)
    return data
