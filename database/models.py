"""SQLAlchemy 2.0 ORM models for the local persistence layer.

Entities
--------
User         -- an account. A seeded "developer" user enables no-login local use.
Subject      -- a normalized analysis subject (e.g. "vitamin d"). Reports are
                classified by subject.
Report       -- one analysis run (medication/procedure/diagnostic/factcheck).
                Holds metadata + cost; the generated artifacts live on disk and
                are referenced by ReportFile rows (files-first design).
ReportFile   -- one artifact produced by a run (markdown/pdf/json), by path.
PatientData  -- structured patient information parsed from attached medical
                reports. Schema is created now; population arrives with the
                later medical-parsing spec.

All primary keys are UUID4 strings for global uniqueness and safe exposure.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.types import JSON


def _uuid() -> str:
    """Generate a new UUID4 string primary key."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp (DTZ-safe)."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class User(Base):
    """An application user."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    # Nullable: the seeded developer user has no password (no-login local mode).
    # The future login spec populates this for real accounts.
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    reports: Mapped[list["Report"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    patient_data: Mapped[list["PatientData"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<User id={self.id!r} username={self.username!r}>"


class Subject(Base):
    """A normalized subject used to classify reports."""

    __tablename__ = "subjects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # Normalized key for dedup/grouping (lowercased, trimmed).
    name: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    # Original-cased label as first seen, for display.
    display_name: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    reports: Mapped[list["Report"]] = relationship(back_populates="subject")
    patient_data: Mapped[list["PatientData"]] = relationship(
        back_populates="subject"
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Subject id={self.id!r} name={self.name!r}>"


class Report(Base):
    """A single analysis run and its metadata."""

    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subject_id: Mapped[str] = mapped_column(
        ForeignKey("subjects.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    # Secondary classification: medication / procedure / diagnostic / factcheck.
    agent_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Raw subject string as supplied to the run (pre-normalization).
    subject_text: Mapped[str] = mapped_column(String(512), nullable=False)
    # Provider/model and run config snapshot (free-form JSON).
    llm_provider: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    implementation: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # Cost summary as produced by cost_tracker.get_summary() (JSON).
    cost_summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    total_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )

    user: Mapped["User"] = relationship(back_populates="reports")
    subject: Mapped["Subject"] = relationship(back_populates="reports")
    files: Mapped[list["ReportFile"]] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )
    patient_data: Mapped[list["PatientData"]] = relationship(
        back_populates="report"
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<Report id={self.id!r} agent={self.agent_type!r} "
            f"subject={self.subject_text!r}>"
        )


class ReportFile(Base):
    """An on-disk artifact produced by a report run, referenced by path."""

    __tablename__ = "report_files"
    __table_args__ = (
        UniqueConstraint("report_id", "file_type", name="uq_report_file_type"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    report_id: Mapped[str] = mapped_column(
        ForeignKey("reports.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Logical key from the orchestrator files dict, e.g. "practitioner_report",
    # "patient_report_pdf", "session", "cost", "summary", "audit".
    file_type: Mapped[str] = mapped_column(String(64), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    report: Mapped["Report"] = relationship(back_populates="files")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<ReportFile id={self.id!r} type={self.file_type!r}>"


class PatientData(Base):
    """Structured patient information parsed from attached medical reports.

    Schema only for now (flexible shape). Populated by the later
    medical-parsing spec. Linked to a user; optionally to the report/subject
    that produced or referenced it.
    """

    __tablename__ = "patient_data"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    report_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("reports.id", ondelete="SET NULL"), nullable=True, index=True
    )
    subject_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("subjects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # Where the data came from: "pdf", "image", "manual", etc.
    source_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # Original filename / source reference of the attached medical report.
    source_reference: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # When the underlying clinical data was recorded (from the document), if known.
    recorded_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Flexible parsed payload (labs, vitals, diagnoses, meds, demographics...).
    data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="patient_data")
    report: Mapped[Optional["Report"]] = relationship(back_populates="patient_data")
    subject: Mapped[Optional["Subject"]] = relationship(back_populates="patient_data")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<PatientData id={self.id!r} source={self.source_type!r}>"
