"""OCR layer: turns images, PDFs and text files into plain text.

- Images (png/jpg/tiff/bmp/webp): Tesseract OCR with light pre-processing.
- PDFs: use the embedded text layer when present (fast, exact); otherwise
  rasterise each page and run Tesseract on it (scanned PDFs).
- Text files: read directly.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageFilter, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TEXT_EXTS = {".txt", ".md", ".csv"}
SUPPORTED_EXTS = IMAGE_EXTS | TEXT_EXTS | {".pdf"}


@dataclass
class OCRResult:
    text: str
    confidence: float | None  # mean word confidence 0-100, None if not OCR'd
    page_count: int
    method: str  # "tesseract" | "pdf-text" | "pdf-ocr" | "plain-text"


class OCRError(Exception):
    pass


def _tesseract():
    import pytesseract

    from .config import Config

    if Config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
    return pytesseract


def preprocess(img: Image.Image) -> Image.Image:
    """Greyscale, upscale small images, denoise and boost contrast."""
    img = ImageOps.exif_transpose(img).convert("L")
    if img.width < 1200:
        scale = 1200 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = ImageOps.autocontrast(img)
    return img


def ocr_image(img: Image.Image, lang: str = "eng") -> tuple[str, float | None]:
    pytesseract = _tesseract()
    img = preprocess(img)
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    confs = [float(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and float(c) >= 0]
    text = pytesseract.image_to_string(img, lang=lang)
    conf = round(sum(confs) / len(confs), 2) if confs else None
    return text, conf


def extract_text(data: bytes, filename: str, lang: str = "eng") -> OCRResult:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTS:
        raise OCRError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTS))}")

    if ext in TEXT_EXTS:
        return OCRResult(data.decode("utf-8", errors="replace"), None, 1, "plain-text")

    if ext in IMAGE_EXTS:
        try:
            img = Image.open(io.BytesIO(data))
        except Exception as e:  # noqa: BLE001
            raise OCRError(f"Could not open image: {e}") from e
        frames, confs = [], []
        for i in range(getattr(img, "n_frames", 1)):  # multi-page TIFF support
            img.seek(i)
            text, conf = ocr_image(img.copy(), lang)
            frames.append(text)
            if conf is not None:
                confs.append(conf)
        conf = round(sum(confs) / len(confs), 2) if confs else None
        return OCRResult("\n\f\n".join(frames), conf, len(frames), "tesseract")

    # PDF
    import pdfplumber

    try:
        pdf = pdfplumber.open(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        raise OCRError(f"Could not open PDF: {e}") from e
    with pdf:
        pages = pdf.pages
        embedded = [p.extract_text() or "" for p in pages]
        if sum(len(t.strip()) for t in embedded) >= 20 * max(1, len(pages)):
            return OCRResult("\n\f\n".join(embedded), None, len(pages), "pdf-text")
        # Scanned PDF: rasterise and OCR each page.
        texts, confs = [], []
        for p in pages:
            img = p.to_image(resolution=300).original
            text, conf = ocr_image(img, lang)
            texts.append(text)
            if conf is not None:
                confs.append(conf)
        conf = round(sum(confs) / len(confs), 2) if confs else None
        return OCRResult("\n\f\n".join(texts), conf, len(pages), "pdf-ocr")
