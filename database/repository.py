"""Data-access functions for reports, subjects, and patient data.

The persistence entry point is :func:`persist_report`, called best-effort by the
orchestrator after files are written. Everything here is a thin, testable layer
over the ORM models.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import PatientData, Report, ReportFile, Subject, User
from .users import get_current_user

logger = logging.getLogger(__name__)


def normalize_subject(text: str) -> str:
    """Normalize a subject string for dedup/grouping (lowercase, collapse ws)."""
    return " ".join((text or "").strip().lower().split())


def get_or_create_subject(session: Session, subject_text: str) -> Subject:
    """Return the Subject for ``subject_text``, creating it if needed."""
    name = normalize_subject(subject_text)
    subject = session.scalar(select(Subject).where(Subject.name == name))
    if subject is not None:
        return subject

    subject = Subject(name=name, display_name=(subject_text or "").strip())
    session.add(subject)
    session.flush()
    return subject


def _extract_total_cost(cost_summary: Optional[dict]) -> Optional[float]:
    """Best-effort extraction of a numeric total cost from a cost summary dict."""
    if not isinstance(cost_summary, dict):
        return None
    for key in ("total_cost", "total", "cost_usd", "grand_total"):
        value = cost_summary.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def persist_report(
    *,
    session: Session,
    agent_type: str,
    subject_text: str,
    files: dict[str, str],
    user: Optional[User] = None,
    llm_provider: Optional[str] = None,
    implementation: Optional[str] = None,
    cost_summary: Optional[dict] = None,
) -> Report:
    """Create a Report (+ Subject + ReportFile rows) within ``session``.

    The caller's session controls the transaction boundary. ``files`` is the
    orchestrator's {logical_type: path} dict; each entry becomes a ReportFile.
    """
    if user is None:
        user = get_current_user(session)
    if user is None:
        raise ValueError("No acting user available to associate the report with.")

    subject = get_or_create_subject(session, subject_text)

    report = Report(
        user_id=user.id,
        subject_id=subject.id,
        agent_type=agent_type,
        subject_text=(subject_text or "").strip(),
        llm_provider=llm_provider,
        implementation=implementation,
        cost_summary=cost_summary,
        total_cost=_extract_total_cost(cost_summary),
    )
    session.add(report)
    session.flush()

    for file_type, file_path in (files or {}).items():
        if not file_path:
            continue
        session.add(
            ReportFile(
                report_id=report.id,
                file_type=str(file_type),
                file_path=str(file_path),
            )
        )
    session.flush()
    logger.info(
        "Persisted report id=%s agent=%s subject=%r (%d files)",
        report.id,
        agent_type,
        report.subject_text,
        len(files or {}),
    )
    return report


# ── Read helpers ─────────────────────────────────────────────────────────────


def list_reports(
    session: Session,
    *,
    user_id: Optional[str] = None,
    subject_id: Optional[str] = None,
    agent_type: Optional[str] = None,
    limit: int = 100,
) -> list[Report]:
    """List reports, optionally filtered, newest first."""
    stmt = select(Report).order_by(Report.created_at.desc())
    if user_id:
        stmt = stmt.where(Report.user_id == user_id)
    if subject_id:
        stmt = stmt.where(Report.subject_id == subject_id)
    if agent_type:
        stmt = stmt.where(Report.agent_type == agent_type)
    stmt = stmt.limit(limit)
    return list(session.scalars(stmt))


def get_report(session: Session, report_id: str) -> Optional[Report]:
    """Fetch a single report by id."""
    return session.get(Report, report_id)


def list_subjects(session: Session, *, limit: int = 500) -> list[Subject]:
    """List all subjects, newest first."""
    stmt = select(Subject).order_by(Subject.created_at.desc()).limit(limit)
    return list(session.scalars(stmt))


def list_reports_by_subject_name(
    session: Session, subject_text: str, *, limit: int = 100
) -> list[Report]:
    """List reports for a subject given its (un-normalized) name."""
    name = normalize_subject(subject_text)
    subject = session.scalar(select(Subject).where(Subject.name == name))
    if subject is None:
        return []
    return list_reports(session, subject_id=subject.id, limit=limit)


# ── Patient data helpers (schema-level; full population in parsing spec) ──────


def create_patient_data(
    *,
    session: Session,
    user: User,
    data: Optional[dict[str, Any]] = None,
    source_type: Optional[str] = None,
    source_reference: Optional[str] = None,
    recorded_at: Optional[datetime] = None,
    report_id: Optional[str] = None,
    subject_id: Optional[str] = None,
) -> PatientData:
    """Create a PatientData row. Provided now for testability; the medical
    parsing spec supplies the extraction that populates ``data``."""
    record = PatientData(
        user_id=user.id,
        report_id=report_id,
        subject_id=subject_id,
        source_type=source_type,
        source_reference=source_reference,
        recorded_at=recorded_at,
        data=data,
    )
    session.add(record)
    session.flush()
    return record


def list_patient_data(
    session: Session, *, user_id: str, limit: int = 100
) -> list[PatientData]:
    """List patient data rows for a user, newest first."""
    stmt = (
        select(PatientData)
        .where(PatientData.user_id == user_id)
        .order_by(PatientData.created_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))
