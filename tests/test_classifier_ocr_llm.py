import random
import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from nexus import classifier, llm
from nexus.config import Config
from nexus.ocr import OCRError, extract_text
from training.synthetic import GENERATORS, add_noise

MODEL = Path(Config.MODEL_DIR)


def test_keyword_classifier():
    clf = classifier.KeywordClassifier()
    rng = random.Random(4)
    correct = sum(clf.predict(fn(rng))[0] == label for label, fn in GENERATORS.items() for _ in range(10))
    assert correct >= 55  # >= ~90% on 60 docs
    assert clf.predict("lorem ipsum")[0] == "unknown"


@pytest.mark.skipif(not (MODEL / "model.keras").exists(), reason="TF model not trained (run python -m training.train_classifier)")
def test_tensorflow_classifier_on_unseen_noisy_docs():
    clf = classifier.TensorFlowClassifier(str(MODEL))
    rng = random.Random(12345)  # different seed than training
    total = correct = 0
    for label, fn in GENERATORS.items():
        for _ in range(15):
            total += 1
            correct += clf.predict(add_noise(rng, fn(rng), 0.08))[0] == label
    assert correct / total >= 0.9


def _render(text):
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    img = Image.new("L", (1100, 60 + 40 * len(text.splitlines())), 255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(text.splitlines()):
        d.text((30, 30 + i * 40), line, fill=0, font=font)
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_ocr_png():
    png = _render("INVOICE\nInvoice No: INV-7781\nTotal Amount: 4,250.00")
    r = extract_text(png, "scan.png")
    assert r.method == "tesseract" and "INV-7781" in r.text and r.confidence > 60


def test_ocr_text_and_unsupported():
    assert extract_text(b"hello", "a.txt").text == "hello"
    with pytest.raises(OCRError):
        extract_text(b"x", "evil.exe")


def test_chunking_and_retrieval():
    text = ("Filler sentence about nothing. " * 60) + "The penalty for late payment is 2% per month. " + ("More filler text. " * 60)
    top = llm.top_chunks("What is the late payment penalty?", text, k=1)
    assert "late payment" in top[0]


def test_llm_answer_with_mocked_anthropic(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(Config, "ANTHROPIC_API_KEY", "test-key")

    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"content": [{"type": "text", "text": "The total is ₹1,18,000."}]}

    calls = {}

    def fake_post(url, headers, json, timeout):
        calls["url"], calls["body"] = url, json
        return Resp()

    monkeypatch.setattr(llm.requests, "post", fake_post)
    r = llm.answer_question("What is the total?", "Invoice total: 1,18,000")
    assert r["mode"] == "llm" and "1,18,000" in r["answer"]
    assert calls["url"].endswith("/v1/messages") and calls["body"]["model"] == Config.LLM_MODEL


def test_llm_extract_fields_parses_json(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(Config, "OPENAI_API_KEY", "k")
    monkeypatch.setattr(llm, "call_llm", lambda *a, **k: 'Sure: {"due_date": "2025-02-01", "vendor_name": null}')
    assert llm.llm_extract_fields("doc", "invoice", ["due_date", "vendor_name"]) == {"due_date": "2025-02-01"}


def test_extractive_fallback_without_key(monkeypatch):
    monkeypatch.setattr(Config, "LLM_PROVIDER", "none")
    r = llm.answer_question("total?", "Total: 50")
    assert r["mode"] == "extractive" and r["sources"]
