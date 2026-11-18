import random

from conftest import upload

from nexus.config import Config
from training.synthetic import invoice, receipt


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json["status"] == "ok"


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200 and b"NEXUS" in r.data


def test_upload_process_and_query(client):
    text = invoice(random.Random(1))
    r = upload(client, "inv.txt", text)
    assert r.status_code == 201
    d = r.json["documents"][0]
    assert d["status"] == "processed"
    assert d["doc_type"] == "invoice"
    assert d["fields"]["invoice_number"].startswith("INV-")
    assert d["is_valid"] is True

    got = client.get(f"/api/documents/{d['id']}?text=true").json
    assert "Invoice" in got["text"] or "INVOICE" in got["text"]

    ans = client.post(f"/api/documents/{d['id']}/ask", json={"question": "What is the total amount?"}).json
    assert ans["mode"] == "extractive" and ans["sources"]

    lst = client.get("/api/documents?type=invoice").json
    assert lst["total"] == 1


def test_duplicate_flagged_on_reupload(client):
    text = receipt(random.Random(2))
    first = upload(client, "r.txt", text).json["documents"][0]
    second = upload(client, "r-copy.txt", text).json["documents"][0]
    assert second["duplicate_of_id"] == first["id"]
    assert any(i["code"].startswith("DUPLICATE") for i in second["issues"])
    assert client.get("/api/documents?duplicates=true").json["total"] == 1


def test_anomaly_flagged_after_history(client):
    rng = random.Random(3)
    for i in range(12):
        upload(client, f"inv{i}.txt", invoice(rng))
    bad = invoice(rng, subtotal=8_000_000_000.0, tax=1_440_000_000.0, total=9_440_000_000.0, number="INV-HUGE-1")
    d = upload(client, "huge.txt", bad).json["documents"][0]
    assert d["is_anomaly"] is True
    s = client.get("/api/stats").json
    assert s["anomalies"] >= 1 and s["total"] == 13


def test_rejects_unsupported_and_missing(client):
    assert upload(client, "malware.exe", b"MZ").status_code == 400
    assert client.post("/api/documents").status_code == 400
    assert client.get("/api/documents/999").status_code == 404
    assert client.post("/api/documents/999/ask", json={"question": "x"}).status_code == 404


def test_delete(client):
    d = upload(client, "a.txt", invoice(random.Random(4))).json["documents"][0]
    assert client.delete(f"/api/documents/{d['id']}").status_code == 204
    assert client.get(f"/api/documents/{d['id']}").status_code == 404


def test_api_key_required(client, monkeypatch):
    monkeypatch.setattr(Config, "API_KEY", "s3cret")
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/documents", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/api/health").status_code == 200  # health stays open for load balancers
