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

import os
from .models import Conversation, Patient, PatientData, Report, ReportFile, Subject, User
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
    report_id: Optional[str] = None,
) -> Report:
    """Create a Report (+ Subject + ReportFile rows) within ``session``.

    The caller's session controls the transaction boundary. ``files`` is the
    orchestrator's {logical_type: path} dict; each entry becomes a ReportFile.

    When ``report_id`` is provided (e.g. the async job UUID), it is used as the
    Report primary key so conversation delete can cascade by the same id.
    """
    if user is None:
        user = get_current_user(session)
    if user is None:
        raise ValueError("No acting user available to associate the report with.")

    subject = get_or_create_subject(session, subject_text)

    kwargs: dict[str, Any] = {
        "user_id": user.id,
        "subject_id": subject.id,
        "agent_type": agent_type,
        "subject_text": (subject_text or "").strip(),
        "llm_provider": llm_provider,
        "implementation": implementation,
        "cost_summary": cost_summary,
        "total_cost": _extract_total_cost(cost_summary),
    }
    if report_id:
        kwargs["id"] = report_id
    report = Report(**kwargs)
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


# ── Patient CRUD Helpers ──────────────────────────────────────────────────────


def create_patient(
    session: Session,
    *,
    user: Optional[User] = None,
    name: str,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    primary_condition: Optional[str] = None,
    contact_email: Optional[str] = None,
    contact_phone: Optional[str] = None,
    metadata_json: Optional[dict[str, Any]] = None,
    clinical_data: Optional[dict[str, Any]] = None,
) -> Patient:
    """Create and persist a new Patient entity."""
    if user is None:
        user = get_current_user(session)
    if user is None:
        raise ValueError("No acting user available.")

    patient = Patient(
        user_id=user.id,
        name=name.strip(),
        age=age,
        gender=gender,
        primary_condition=primary_condition,
        contact_email=contact_email,
        contact_phone=contact_phone,
        metadata_json=metadata_json or {},
        clinical_data=clinical_data or {},
    )
    session.add(patient)
    session.flush()
    return patient


def list_patients(
    session: Session, *, user_id: Optional[str] = None, limit: int = 200
) -> list[Patient]:
    """List patients, newest first."""
    stmt = select(Patient).order_by(Patient.created_at.desc())
    if user_id:
        stmt = stmt.where(Patient.user_id == user_id)
    stmt = stmt.limit(limit)
    return list(session.scalars(stmt))


def get_patient(session: Session, patient_id: str) -> Optional[Patient]:
    """Fetch a single Patient by primary key ID."""
    return session.get(Patient, patient_id)


def update_patient(
    session: Session,
    patient_id: str,
    *,
    name: Optional[str] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    primary_condition: Optional[str] = None,
    contact_email: Optional[str] = None,
    contact_phone: Optional[str] = None,
    metadata_json: Optional[dict[str, Any]] = None,
    clinical_data: Optional[dict[str, Any]] = None,
) -> Optional[Patient]:
    """Update existing Patient record."""
    patient = session.get(Patient, patient_id)
    if patient is None:
        return None

    if name is not None:
        patient.name = name.strip()
    if age is not None:
        patient.age = age
    if gender is not None:
        patient.gender = gender
    if primary_condition is not None:
        patient.primary_condition = primary_condition
    if contact_email is not None:
        patient.contact_email = contact_email
    if contact_phone is not None:
        patient.contact_phone = contact_phone
    if metadata_json is not None:
        patient.metadata_json = metadata_json
    if clinical_data is not None:
        patient.clinical_data = clinical_data

    session.flush()
    return patient


def delete_patient(session: Session, patient_id: str) -> bool:
    """Delete a patient record."""
    patient = session.get(Patient, patient_id)
    if patient is None:
        return False
    session.delete(patient)
    session.flush()
    return True


def delete_report_and_artifacts(session: Session, report_id: str) -> list[str]:
    """Delete a report database record and remove all its on-disk files.
    
    Returns a list of removed file paths.
    """
    report = session.get(Report, report_id)
    if report is None:
        return []

    removed_paths = []
    for report_file in report.files:
        path = report_file.file_path
        if path and os.path.exists(path):
            try:
                os.remove(path)
                removed_paths.append(path)
            except Exception as e:
                logger.warning("Failed to remove file %s: %s", path, e)

    session.delete(report)
    session.flush()
    return removed_paths


# ── Conversation (persistent jobs) helpers ────────────────────────────────────


def _conversation_has_docs(files: Optional[dict]) -> bool:
    """True when at least one documentation artifact path is present."""
    if not isinstance(files, dict) or not files:
        return False
    doc_keys = (
        "patient_report",
        "practitioner_report",
        "summary",
        "medication_summary",
        "medication_detailed",
        "diagnostic_report",
        "patient_report_pdf",
        "practitioner_report_pdf",
        "summary_pdf",
        "context_report",
    )
    for key in doc_keys:
        path = files.get(key)
        if path:
            return True
    # Fallback: any .md/.pdf path counts as documentation.
    for path in files.values():
        if isinstance(path, str) and (
            path.endswith(".md") or path.endswith(".pdf")
        ):
            return True
    return False


def conversation_to_job_dict(conv: Conversation) -> dict[str, Any]:
    """Serialize a Conversation row to the API job dict shape used by the UI."""
    files = conv.files or {}
    return {
        "id": conv.id,
        "query": conv.query,
        "agent_id": conv.agent_id,
        "status": conv.status,
        "model": conv.model,
        "implementation": conv.implementation,
        "error": conv.error,
        "files": files if files else None,
        "result": conv.result,
        "parent_job_id": conv.parent_job_id,
        "report_id": conv.report_id,
        "patient_id": conv.patient_id,
        "has_docs": _conversation_has_docs(files),
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
    }


def create_conversation(
    session: Session,
    *,
    conversation_id: str,
    query: str,
    agent_id: Optional[str] = None,
    status: str = "pending",
    model: Optional[str] = None,
    implementation: Optional[str] = None,
    parent_job_id: Optional[str] = None,
    patient_id: Optional[str] = None,
) -> Conversation:
    """Insert a new Conversation row for a background job."""
    conv = Conversation(
        id=conversation_id,
        query=(query or "").strip(),
        agent_id=agent_id,
        status=status,
        model=model,
        implementation=implementation,
        parent_job_id=parent_job_id,
        patient_id=patient_id,
    )
    session.add(conv)
    session.flush()
    return conv


def update_conversation(
    session: Session,
    conversation_id: str,
    *,
    status: Optional[str] = None,
    agent_id: Optional[str] = None,
    error: Optional[str] = None,
    files: Optional[dict] = None,
    result: Optional[dict] = None,
    report_id: Optional[str] = None,
    model: Optional[str] = None,
    implementation: Optional[str] = None,
    patient_id: Optional[str] = None,
) -> Optional[Conversation]:
    """Patch fields on an existing Conversation. Returns None if missing."""
    conv = session.get(Conversation, conversation_id)
    if conv is None:
        return None
    if status is not None:
        conv.status = status
    if agent_id is not None:
        conv.agent_id = agent_id
    if patient_id is not None:
        conv.patient_id = patient_id
        conv.agent_id = agent_id
    if error is not None:
        conv.error = error
    if files is not None:
        conv.files = files
    if result is not None:
        conv.result = result
    if report_id is not None:
        conv.report_id = report_id
    if model is not None:
        conv.model = model
    if implementation is not None:
        conv.implementation = implementation
    from datetime import timezone

    conv.updated_at = datetime.now(timezone.utc)
    session.flush()
    return conv


def get_conversation(session: Session, conversation_id: str) -> Optional[Conversation]:
    """Fetch a Conversation by id."""
    return session.get(Conversation, conversation_id)


def list_conversations(session: Session, *, limit: int = 200) -> list[Conversation]:
    """List conversations newest first."""
    stmt = select(Conversation).order_by(Conversation.created_at.desc()).limit(limit)
    return list(session.scalars(stmt))


def delete_conversation_and_artifacts(
    session: Session, conversation_id: str
) -> list[str]:
    """Delete a conversation, its linked report, and all on-disk artifacts.

    Returns the list of removed file paths.
    """
    removed: list[str] = []
    conv = session.get(Conversation, conversation_id)

    # Paths stored on the conversation itself.
    if conv and isinstance(conv.files, dict):
        for path in conv.files.values():
            if path and isinstance(path, str) and os.path.exists(path):
                try:
                    os.remove(path)
                    removed.append(path)
                except Exception as e:
                    logger.warning("Failed to remove conversation file %s: %s", path, e)

    report_ids: set[str] = set()
    if conv and conv.report_id:
        report_ids.add(conv.report_id)
    report_ids.add(conversation_id)

    for rid in report_ids:
        removed.extend(delete_report_and_artifacts(session, rid))

    if conv is not None:
        session.delete(conv)
        session.flush()

    return list(dict.fromkeys(removed))  # dedupe, preserve order


def backfill_conversations_from_reports(session: Session) -> int:
    """Create Conversation rows for Report records that have none yet.

    Used so pre-existing DB reports appear in the conversations UI after
    the persistence layer is introduced. Returns number of rows created.
    """
    existing_ids = set(session.scalars(select(Conversation.id)))
    reports = list_reports(session, limit=500)
    created = 0
    for report in reports:
        if report.id in existing_ids:
            continue
        files = {rf.file_type: rf.file_path for rf in report.files}
        session.add(
            Conversation(
                id=report.id,
                query=report.subject_text,
                agent_id=report.agent_type,
                status="completed",
                model=report.llm_provider,
                implementation=report.implementation,
                files=files or None,
                report_id=report.id,
                created_at=report.created_at,
                updated_at=report.created_at,
            )
        )
        created += 1
    if created:
        session.flush()
        logger.info("Backfilled %d conversations from existing reports", created)
    return created

