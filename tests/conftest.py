import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexus import classifier, create_app  # noqa: E402
from nexus.config import Config  # noqa: E402


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(Config, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(Config, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(Config, "LLM_PROVIDER", "none")
    monkeypatch.setattr(Config, "API_KEY", "")
    monkeypatch.setattr(Config, "ANOMALY_MIN_HISTORY", 8)
    classifier.reset_classifier()
    yield create_app()
    classifier.reset_classifier()


@pytest.fixture
def client(app):
    return app.test_client()


def upload(client, name: str, content: str | bytes, headers=None):
    data = content.encode() if isinstance(content, str) else content
    import io

    return client.post(
        "/api/documents",
        data={"file": (io.BytesIO(data), name)},
        content_type="multipart/form-data",
        headers=headers or {},
    )
