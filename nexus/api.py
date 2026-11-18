"""REST API (all routes under /api)."""
from __future__ import annotations

import hmac
from functools import wraps

from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import func, select

from . import __version__, llm
from .classifier import get_classifier
from .config import Config
from .models import Document
from .ocr import SUPPORTED_EXTS
from .pipeline import process_document

bp = Blueprint("api", __name__, url_prefix="/api")


def _session():
    return current_app.extensions["nexus_session"]()


def require_key(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if Config.API_KEY and not hmac.compare_digest(request.headers.get("X-API-Key", ""), Config.API_KEY):
            return jsonify(error="Invalid or missing API key"), 401
        return fn(*a, **kw)

    return wrapper


@bp.errorhandler(413)
def too_large(_):
    return jsonify(error=f"File too large (max {Config.MAX_UPLOAD_MB} MB)"), 413


@bp.get("/health")
def health():
    with _session() as s:
        s.execute(select(1))
    return jsonify(
        status="ok",
        version=__version__,
        classifier=get_classifier(Config.MODEL_DIR).name,
        llm_enabled=llm.llm_available(),
        storage=Config.STORAGE_BACKEND,
    )


@bp.post("/documents")
@require_key
def upload():
    files = request.files.getlist("file") or request.files.getlist("files")
    if not files:
        return jsonify(error="Send one or more files in form field 'file'"), 400
    results = []
    with _session() as s:
        for f in files:
            name = f.filename or "upload"
            ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext not in SUPPORTED_EXTS:
                results.append({"filename": name, "status": "rejected", "error": f"Unsupported type '{ext}'"})
                continue
            doc = process_document(s, current_app.extensions["nexus_storage"], f.read(), name, f.mimetype or "")
            results.append(doc.to_dict())
    code = 201 if any(r.get("id") for r in results) else 400
    return jsonify(documents=results), code


@bp.get("/documents")
@require_key
def list_documents():
    page = max(1, request.args.get("page", 1, type=int))
    per_page = min(100, max(1, request.args.get("per_page", 20, type=int)))
    q = select(Document)
    if t := request.args.get("type"):
        q = q.where(Document.doc_type == t)
    if request.args.get("anomalies") == "true":
        q = q.where(Document.is_anomaly.is_(True))
    if request.args.get("duplicates") == "true":
        q = q.where(Document.duplicate_of_id.is_not(None))
    if (v := request.args.get("valid")) in ("true", "false"):
        q = q.where(Document.is_valid.is_(v == "true"))
    with _session() as s:
        total = s.scalar(select(func.count()).select_from(q.subquery()))
        docs = s.scalars(q.order_by(Document.id.desc()).offset((page - 1) * per_page).limit(per_page)).all()
        return jsonify(total=total, page=page, per_page=per_page, documents=[d.to_dict() for d in docs])


@bp.get("/documents/<int:doc_id>")
@require_key
def get_document(doc_id):
    with _session() as s:
        doc = s.get(Document, doc_id)
        if not doc:
            return jsonify(error="Not found"), 404
        return jsonify(doc.to_dict(include_text=request.args.get("text") == "true"))


@bp.delete("/documents/<int:doc_id>")
@require_key
def delete_document(doc_id):
    with _session() as s:
        doc = s.get(Document, doc_id)
        if not doc:
            return jsonify(error="Not found"), 404
        for d in s.scalars(select(Document).where(Document.duplicate_of_id == doc_id)):
            d.duplicate_of_id = None
        current_app.extensions["nexus_storage"].delete(doc.storage_key)
        s.delete(doc)
        s.commit()
    return "", 204


@bp.post("/documents/<int:doc_id>/ask")
@require_key
def ask(doc_id):
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify(error="JSON body must include 'question'"), 400
    with _session() as s:
        doc = s.get(Document, doc_id)
        if not doc:
            return jsonify(error="Not found"), 404
        try:
            result = llm.answer_question(question, doc.text, {"type": doc.doc_type, "fields": doc.fields})
        except llm.LLMError as e:
            return jsonify(error=str(e)), 502
    return jsonify(question=question, **result)


@bp.get("/stats")
@require_key
def stats():
    with _session() as s:
        by_type = dict(s.execute(select(Document.doc_type, func.count()).group_by(Document.doc_type)).all())
        return jsonify(
            total=s.scalar(select(func.count(Document.id))) or 0,
            processed=s.scalar(select(func.count()).where(Document.status == "processed")) or 0,
            failed=s.scalar(select(func.count()).where(Document.status == "failed")) or 0,
            valid=s.scalar(select(func.count()).where(Document.is_valid.is_(True))) or 0,
            duplicates=s.scalar(select(func.count()).where(Document.duplicate_of_id.is_not(None))) or 0,
            anomalies=s.scalar(select(func.count()).where(Document.is_anomaly.is_(True))) or 0,
            by_type={k or "unknown": v for k, v in by_type.items()},
            avg_processing_ms=round(s.scalar(select(func.avg(Document.processing_ms))) or 0),
        )
