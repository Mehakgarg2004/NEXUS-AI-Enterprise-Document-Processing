"""Database models (SQLAlchemy 2.x). Works with PostgreSQL and SQLite."""
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(100), default="")
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="processing")  # processing|processed|failed

    text: Mapped[str] = mapped_column(Text, default="")
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, default=1)

    doc_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    classification_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    classifier_used: Mapped[str | None] = mapped_column(String(32), nullable=True)

    fields: Mapped[dict] = mapped_column(JSON, default=dict)
    entities: Mapped[dict] = mapped_column(JSON, default=dict)

    is_valid: Mapped[bool] = mapped_column(Boolean, default=False)
    duplicate_of_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id"), nullable=True)
    duplicate_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    issues: Mapped[list["ValidationIssue"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    def to_dict(self, include_text: bool = False) -> dict:
        data = {
            "id": self.id,
            "filename": self.filename,
            "content_type": self.content_type,
            "sha256": self.sha256,
            "status": self.status,
            "page_count": self.page_count,
            "ocr_confidence": self.ocr_confidence,
            "doc_type": self.doc_type,
            "classification_confidence": self.classification_confidence,
            "classifier_used": self.classifier_used,
            "fields": self.fields or {},
            "entities": self.entities or {},
            "is_valid": self.is_valid,
            "duplicate_of_id": self.duplicate_of_id,
            "duplicate_score": self.duplicate_score,
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "issues": [i.to_dict() for i in self.issues],
            "error": self.error,
            "processing_ms": self.processing_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_text:
            data["text"] = self.text
        return data


class ValidationIssue(Base):
    __tablename__ = "validation_issues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    severity: Mapped[str] = mapped_column(String(16))  # error|warning|info
    code: Mapped[str] = mapped_column(String(64))
    message: Mapped[str] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="issues")

    def to_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code, "message": self.message}


def make_session_factory(database_url: str):
    kwargs = {}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(database_url, pool_pre_ping=True, **kwargs)
    with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            # Several gunicorn workers boot at once; serialise schema creation.
            conn.execute(text("SELECT pg_advisory_xact_lock(424242)"))
        Base.metadata.create_all(conn)
    return sessionmaker(bind=engine, expire_on_commit=False)
