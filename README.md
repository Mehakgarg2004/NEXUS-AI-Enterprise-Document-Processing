# NEXUS AI – Enterprise Intelligent Document Processing Platform

NEXUS AI is an AI-powered platform that **extracts, classifies and validates enterprise documents**. You upload a scanned invoice, receipt, purchase order, bank statement, contract or resume. NEXUS runs OCR, works out what kind of document it is, pulls out the important fields, checks them, flags duplicates and anomalies, and lets you **ask questions about the document in plain English** through an LLM.

**Tech stack:** Python · TensorFlow · Tesseract OCR · NLP · LLM APIs (Anthropic / OpenAI) · Flask · PostgreSQL · Docker · AWS (S3, ECS, RDS) · MLflow

---

## Features

| Area | What it does |
|---|---|
| **OCR** | Tesseract with image pre-processing (greyscale, upscaling, denoising, contrast). Handles PNG/JPG/TIFF (multi-page), PDFs with embedded text, and scanned PDFs (rasterised page by page). Reports OCR confidence. |
| **Classification** | A TensorFlow/Keras text classifier (TextVectorization → Embedding → Dense) sorts documents into 6 types. Training is tracked in **MLflow**. If no model has been trained yet, a keyword classifier takes over. |
| **Information extraction** | Rule-based NLP extractors for each document type: invoice numbers, dates, amounts, GSTIN, IFSC, parties, skills and more. Corrects common OCR mistakes (`0`↔`O`, `110, 466` → `110,466`). Any field the rules miss is **filled in by the LLM**. |
| **LLM document Q&A** | Retrieval-augmented Q&A: TF-IDF finds the most relevant chunks of the document and the LLM answers from them. With no API key, it falls back to showing the relevant passages. |
| **Validation** | Required-field checks, subtotal + tax = total, due date after issue date, GSTIN checksum, IFSC format, bank balance reconciliation, future or implausible dates, and low OCR confidence. |
| **Duplicate detection** | Exact duplicates (SHA-256), near-duplicates (5-word shingles + Jaccard similarity, which catches a re-scan of the same paper), and business-key duplicates (same invoice number from the same vendor). |
| **Anomaly detection** | An Isolation Forest, plus robust z-scores (median/MAD), compares each document against past documents of the same type. Rule checks catch suspiciously round amounts and an unusually high tax ratio. Every flag comes with a reason you can read. |
| **REST API + dashboard** | Flask REST API and a single-page web dashboard: drag-and-drop upload, filters, extracted fields, validation results and Q&A. |
| **Production ready** | PostgreSQL via SQLAlchemy, local or S3 storage, Docker + docker-compose (app + Postgres + MLflow), gunicorn, optional API-key auth, health checks, GitHub Actions CI and an AWS ECS deployment guide. |

## Architecture

```
            ┌──────────┐   ┌────────┐   ┌──────────────┐   ┌────────────────┐
 upload ──▶ │ Storage  │──▶│  OCR   │──▶│ Classifier   │──▶│ NLP extraction │
            │ local/S3 │   │Tesser- │   │ TensorFlow   │   │ rules + LLM    │
            └──────────┘   │ act    │   │ (MLflow)     │   └───────┬────────┘
                           └────────┘   └──────────────┘           │
                                                                   ▼
   ┌──────────────┐    ┌──────────────────┐    ┌───────────────────────────┐
   │ PostgreSQL   │◀───│ Anomaly detection│◀───│ Validation + duplicate    │
   │ (SQLAlchemy) │    │ IsolationForest  │    │ detection                 │
   └──────┬───────┘    └──────────────────┘    └───────────────────────────┘
          │
          ▼
   Flask REST API  ──▶  Web dashboard  ·  LLM document Q&A (RAG)
```

## Project structure

```
NEXUS-AI-Enterprise-Document-Processing/
├── nexus/
│   ├── __init__.py        # Flask app factory
│   ├── config.py          # Settings from environment variables
│   ├── models.py          # SQLAlchemy models (Document, ValidationIssue)
│   ├── ocr.py             # Tesseract OCR + PDF handling
│   ├── classifier.py      # TensorFlow classifier (+ keyword fallback)
│   ├── extraction.py      # NLP field + entity extraction
│   ├── validation.py      # Validation rules (GSTIN checksum, totals, dates…)
│   ├── duplicates.py      # Exact / near / business-key duplicate detection
│   ├── anomaly.py         # Isolation Forest + robust z-score anomalies
│   ├── llm.py             # LLM Q&A (RAG) and LLM-assisted extraction
│   ├── storage.py         # Local disk or AWS S3
│   ├── pipeline.py        # Orchestrates the full processing pipeline
│   ├── api.py             # REST API
│   ├── web.py             # Dashboard route
│   └── templates/index.html
├── training/
│   ├── synthetic.py       # Synthetic document generator with OCR-style noise
│   ├── train_classifier.py# TensorFlow training + MLflow tracking
│   └── generate_samples.py# Creates sample docs (text + scanned PNGs)
├── tests/                 # 36 pytest tests
├── sample_docs/           # Example documents
├── deploy/ecs-task-definition.json
├── docs/aws-deployment.md
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/Mehakgarg2004/NEXUS-AI-Enterprise-Document-Processing.git
cd NEXUS-AI-Enterprise-Document-Processing
```

### Option A: Docker (recommended)

```bash
cp .env.example .env          # optionally add ANTHROPIC_API_KEY or OPENAI_API_KEY
docker compose up --build
```

- Dashboard: http://localhost:5000
- MLflow UI: http://localhost:5001

### Option B: run locally

Requirements: Python 3.10–3.12 and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (`sudo apt install tesseract-ocr` / `brew install tesseract` / [Windows installer](https://github.com/UB-Mannheim/tesseract/wiki)).

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m training.train_classifier     # trains the TensorFlow model, logs to MLflow
python -m training.generate_samples     # creates sample scans in sample_docs/
python wsgi.py                           # http://localhost:5000
```

SQLite is used automatically unless you set `DATABASE_URL` to a PostgreSQL URL.

View training runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Status, classifier in use, whether the LLM is enabled |
| `POST` | `/api/documents` | Upload one or more files (`multipart/form-data`, field `file`) |
| `GET` | `/api/documents` | List documents. Filters: `type`, `anomalies=true`, `duplicates=true`, `valid=true/false`, `page`, `per_page` |
| `GET` | `/api/documents/<id>` | Document details (`?text=true` includes the OCR text) |
| `DELETE` | `/api/documents/<id>` | Delete a document |
| `POST` | `/api/documents/<id>/ask` | Ask a question: `{"question": "What is the total?"}` |
| `GET` | `/api/stats` | Totals, counts by type, duplicates, anomalies, average processing time |

```bash
curl -F "file=@sample_docs/invoice_scan.png" http://localhost:5000/api/documents
curl -X POST -H "Content-Type: application/json" \
     -d '{"question":"When is this invoice due?"}' http://localhost:5000/api/documents/1/ask
```

If `NEXUS_API_KEY` is set, send it as the `X-API-Key` header.

## Example output

A scanned invoice (`invoice_scan.png`) produces:

```json
{
  "doc_type": "invoice",
  "classification_confidence": 0.9999,
  "classifier_used": "tensorflow",
  "ocr_confidence": 94.01,
  "fields": {
    "invoice_number": "INV-2024-46568",
    "invoice_date": "2024-05-29",
    "due_date": "2024-06-28",
    "vendor_name": "Zenith Logistics",
    "customer_name": "Global Foods Co",
    "subtotal": 613703.47,
    "tax": 110466.62,
    "total": 724170.09,
    "gstin": "19DOAVP5148G7Z8",
    "currency": "INR"
  },
  "is_valid": true,
  "issues": []
}
```

Uploading the same invoice again as a `.txt` file is flagged as `DUPLICATE_BUSINESS_KEY` of document #1.

## Testing

```bash
python -m pytest -q
```

The 36 tests cover extraction, validation rules, GSTIN checksums, duplicate and anomaly detection, OCR on a rendered image, the TensorFlow classifier on unseen noisy documents, LLM calls (mocked), and the full REST API.

## Model training and MLflow

`training/train_classifier.py` trains on synthetic enterprise documents with OCR-style noise (character swaps, dropped words, truncated pages), or on your own labelled CSV:

```bash
python -m training.train_classifier --data my_labelled_docs.csv --epochs 20   # CSV columns: text,label
```

Each run logs its hyperparameters, per-epoch loss and accuracy, test accuracy, macro-F1, per-class F1, a confusion matrix, a classification report, and the model artifact to MLflow.

> The bundled model is trained on synthetic data. For production accuracy on your own documents, retrain it on real labelled samples.

## Deploying to AWS

See [docs/aws-deployment.md](docs/aws-deployment.md) for the ECR → ECS Fargate + RDS PostgreSQL + S3 + Secrets Manager setup, and how to host MLflow on AWS.

## License

MIT
