"""Local persistence layer for the medical multi-agent system.

Additive and optional: file output (outputs/) never depends on this package.
Report persistence is best-effort and can be disabled for file-only debugging
via ``DB_PERSISTENCE_ENABLED=false``.

Public surface
--------------
- config:     get_database_url, is_db_enabled, get_engine, get_session_factory
- session:    init_db, ensure_initialized, session_scope
- models:     Base, User, Subject, Report, ReportFile, PatientData
- users:      get_current_user
- seed:       seed_developer_user, DEVELOPER_USER_ID, DEVELOPER_USERNAME
- repository: persist_report and read/create helpers
"""

from __future__ import annotations

from .config import (
    get_database_url,
    get_engine,
    get_session_factory,
    is_db_enabled,
    reset_engine_cache,
)
from .models import Base, Conversation, PatientData, Report, ReportFile, Subject, User
from .repository import (
    backfill_conversations_from_reports,
    conversation_to_job_dict,
    create_conversation,
    create_patient_data,
    delete_conversation_and_artifacts,
    get_conversation,
    get_or_create_subject,
    get_report,
    list_conversations,
    list_patient_data,
    list_reports,
    list_reports_by_subject_name,
    list_subjects,
    normalize_subject,
    persist_report,
    update_conversation,
)
from .seed import DEVELOPER_USER_ID, DEVELOPER_USERNAME, seed_developer_user
from .session import (
    ensure_initialized,
    init_db,
    reset_initialized_flag,
    session_scope,
)
from .users import get_current_user

__all__ = [
    "DEVELOPER_USERNAME",
    "DEVELOPER_USER_ID",
    "Base",
    "Conversation",
    "PatientData",
    "Report",
    "ReportFile",
    "Subject",
    "User",
    "backfill_conversations_from_reports",
    "conversation_to_job_dict",
    "create_conversation",
    "create_patient_data",
    "delete_conversation_and_artifacts",
    "ensure_initialized",
    "get_conversation",
    "get_current_user",
    "get_database_url",
    "get_engine",
    "get_or_create_subject",
    "get_report",
    "get_session_factory",
    "init_db",
    "is_db_enabled",
    "list_conversations",
    "list_patient_data",
    "list_reports",
    "list_reports_by_subject_name",
    "list_subjects",
    "normalize_subject",
    "persist_report",
    "reset_engine_cache",
    "reset_initialized_flag",
    "seed_developer_user",
    "session_scope",
    "update_conversation",
]
