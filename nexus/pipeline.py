"""End-to-end document processing pipeline.

upload -> store -> OCR -> classify -> extract (rules + optional LLM)
       -> validate -> duplicate check -> anomaly check -> persist
"""
from __future__ import annotations

import logging
import time

from sqlalchemy import select

from . import anomaly, classifier, duplicates, extraction, llm, ocr, validation
from .config import Config
from .models import Document, ValidationIssue

log = logging.getLogger("nexus.pipeline")

HISTORY_LIMIT = 500


def process_document(session, storage, data: bytes, filename: str, content_type: str = "") -> Document:
    t0 = time.perf_counter()
    doc = Document(
        filename=filename,
        content_type=content_type,
        sha256=duplicates.sha256(data),
        storage_key=storage.save(data, filename),
        status="processing",
    )
    session.add(doc)
    session.flush()  # assign id

    try:
        # 1. OCR
        res = ocr.extract_text(data, filename, Config.OCR_LANG)
        doc.text, doc.ocr_confidence, doc.page_count = res.text, res.confidence, res.page_count

        # 2. Classification
        label, conf, used, _ = classifier.classify(res.text, Config.MODEL_DIR)
        doc.doc_type, doc.classification_confidence, doc.classifier_used = label, conf, used

        # 3. Extraction (rules first, LLM fills the gaps)
        fields = extraction.extract_fields(res.text, label)
        missing = [k for k, v in fields.items() if v in (None, "", [])]
        if missing:
            fields.update(llm.llm_extract_fields(res.text, label, missing))
        doc.fields = fields
        doc.entities = extraction.extract_entities(res.text)

        # 4. Validation
        issues = validation.validate(label, fields, res.confidence)

        # 5. Duplicate detection (against recent history)
        candidates = session.scalars(
            select(Document)
            .where(Document.id != doc.id, Document.status == "processed")
            .order_by(Document.id.desc())
            .limit(HISTORY_LIMIT)
        ).all()
        dup = duplicates.find_duplicate(doc.sha256, res.text, label, fields, candidates, Config.NEAR_DUPLICATE_THRESHOLD)
        if dup:
            doc.duplicate_of_id, doc.duplicate_score = dup.document_id, dup.score
            issues.append(validation.Issue("warning", f"DUPLICATE_{dup.kind.upper()}",
                                           f"Looks like a duplicate of document #{dup.document_id} ({dup.kind}, similarity {dup.score:.0%})."))

        # 6. Anomaly detection (same type, excluding duplicates)
        history = [c.fields for c in candidates if c.doc_type == label and c.fields and c.duplicate_of_id is None]
        an = anomaly.detect(label, fields, history, Config.ANOMALY_MIN_HISTORY)
        doc.is_anomaly, doc.anomaly_score = an.is_anomaly, an.score
        for r in an.reasons:
            issues.append(validation.Issue("warning", "ANOMALY", r))

        doc.issues = [ValidationIssue(severity=i.severity, code=i.code, message=i.message) for i in issues]
        doc.is_valid = not any(i.severity == "error" for i in issues)
        doc.status = "processed"
    except ocr.OCRError as e:
        doc.status, doc.error = "failed", str(e)
    except Exception as e:  # noqa: BLE001 - never lose the upload record
        log.exception("Processing failed for %s", filename)
        doc.status, doc.error = "failed", f"{type(e).__name__}: {e}"

    doc.processing_ms = int((time.perf_counter() - t0) * 1000)
    session.commit()
    return doc
