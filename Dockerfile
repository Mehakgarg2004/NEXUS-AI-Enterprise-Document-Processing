FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Tesseract OCR engine + fonts (for sample generation)
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng fonts-dejavu-core libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Train the document classifier at build time so the image ships with a model.
# (Tracked in a local MLflow SQLite store inside the image; point
#  MLFLOW_TRACKING_URI at your MLflow server to train against it instead.)
RUN python -m training.train_classifier --epochs 12 \
 && useradd --create-home nexus \
 && mkdir -p /app/uploads \
 && chown -R nexus:nexus /app

USER nexus
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "180", "wsgi:app"]
