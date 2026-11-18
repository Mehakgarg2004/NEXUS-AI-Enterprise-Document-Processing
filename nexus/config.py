"""Application configuration, loaded from environment variables."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    # Database: PostgreSQL in production, SQLite fallback for local dev/tests.
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'nexus.db'}")

    # Storage: "local" or "s3"
    STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))
    S3_BUCKET = os.getenv("S3_BUCKET", "")
    AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

    # ML model
    MODEL_DIR = os.getenv("MODEL_DIR", str(BASE_DIR / "models" / "doc_classifier"))
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{BASE_DIR / 'mlflow.db'}")

    # OCR
    TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")  # leave blank to use PATH
    OCR_LANG = os.getenv("OCR_LANG", "eng")

    # LLM (document Q&A + extraction). Provider: "anthropic", "openai" or "none".
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
    LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

    # Pipeline tuning
    NEAR_DUPLICATE_THRESHOLD = float(os.getenv("NEAR_DUPLICATE_THRESHOLD", "0.85"))
    ANOMALY_MIN_HISTORY = int(os.getenv("ANOMALY_MIN_HISTORY", "8"))
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))

    # API security: if set, requests must send header "X-API-Key: <value>"
    API_KEY = os.getenv("NEXUS_API_KEY", "")
