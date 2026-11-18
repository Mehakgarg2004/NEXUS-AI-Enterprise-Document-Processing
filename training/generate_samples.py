"""Write sample documents (text + scanned-style PNG images) to sample_docs/
so you can try the platform immediately.

    python -m training.generate_samples
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from training.synthetic import GENERATORS  # noqa: E402

OUT = ROOT / "sample_docs"


def render_png(text: str, path: Path):
    """Render text onto a white page so it can be OCR'd like a scan."""
    font = None
    for f in ["DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "consola.ttf", "Courier New.ttf"]:
        try:
            font = ImageFont.truetype(f, 22)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    lines = text.replace("₹", "Rs.").splitlines()
    w = 60 + max(int(font.getlength(l)) for l in lines if l) if lines else 800
    h = 60 + 32 * len(lines)
    img = Image.new("L", (max(w, 900), h), 255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((30, 30 + i * 32), line, fill=0, font=font)
    img.save(path)


def main():
    OUT.mkdir(exist_ok=True)
    rng = random.Random(2026)
    for label, fn in GENERATORS.items():
        text = fn(rng)
        (OUT / f"{label}.txt").write_text(text, encoding="utf-8")
        render_png(text, OUT / f"{label}_scan.png")
    print(f"Wrote samples to {OUT}")


if __name__ == "__main__":
    main()
