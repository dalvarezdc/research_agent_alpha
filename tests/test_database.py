#!/usr/bin/env python3
"""Tests for the local persistence layer (database package)."""

import importlib

import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Isolated SQLite database per test.

    Points DATABASE_URL at a temp file, resets the cached engine/session
    factory and the init guard, then initializes a fresh schema + seed.
    """
    db_path = tmp_path / "test_app.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.delenv("DB_PERSISTENCE_ENABLED", raising=False)
    monkeypatch.setenv("APP_ENV", "local")

    import database

    database.reset_engine_cache()
    database.reset_initialized_flag()
    database.init_db(seed=True)

    yield database

    database.reset_engine_cache()
    database.reset_initialized_flag()


def test_developer_user_seeded(db):
    with db.session_scope() as session:
        user = session.get(db.User, db.DEVELOPER_USER_ID)
        assert user is not None
        assert user.username == db.DEVELOPER_USERNAME
        assert user.password_hash is None  # no-login local account
        assert user.is_active is True


def test_seed_is_idempotent(db):
    with db.session_scope() as session:
        db.seed_developer_user(session)
        db.seed_developer_user(session)
    with db.session_scope() as session:
        users = session.query(db.User).filter_by(
            username=db.DEVELOPER_USERNAME
        ).all()
        assert len(users) == 1


def test_get_current_user_returns_developer_in_local_mode(db):
    with db.session_scope() as session:
        user = db.get_current_user(session)
        assert user is not None
        assert user.id == db.DEVELOPER_USER_ID


def test_get_current_user_none_when_not_local(db, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    with db.session_scope() as session:
        assert db.get_current_user(session) is None


def test_uuid_primary_keys(db):
    with db.session_scope() as session:
        subject = db.get_or_create_subject(session, "Vitamin D")
        assert isinstance(subject.id, str)
        assert len(subject.id) == 36  # UUID4 string


def test_subject_normalization_and_dedup(db):
    with db.session_scope() as session:
        s1 = db.get_or_create_subject(session, "  Vitamin   D ")
        s2 = db.get_or_create_subject(session, "vitamin d")
        assert s1.id == s2.id
        assert s1.name == "vitamin d"
        assert s1.display_name == "Vitamin   D"  # original-cased, trimmed


def test_persist_report_creates_rows(db):
    files = {
        "session": "/tmp/x_session.json",
        "patient_report": "/tmp/x_patient.md",
        "patient_report_pdf": "/tmp/x_patient.pdf",
    }
    with db.session_scope() as session:
        user = db.get_current_user(session)
        report = db.persist_report(
            session=session,
            agent_type="factcheck",
            subject_text="Vitamin D",
            files=files,
            user=user,
            llm_provider="grok-4.3",
            implementation="langchain",
            cost_summary={"total_cost": 0.42},
        )
        report_id = report.id

    with db.session_scope() as session:
        report = db.get_report(session, report_id)
        assert report is not None
        assert report.agent_type == "factcheck"
        assert report.subject_text == "Vitamin D"
        assert report.total_cost == pytest.approx(0.42)
        assert report.user_id == db.DEVELOPER_USER_ID
        assert {f.file_type for f in report.files} == set(files.keys())
        assert report.subject.name == "vitamin d"


def test_list_reports_by_subject_name(db):
    with db.session_scope() as session:
        user = db.get_current_user(session)
        for agent in ("factcheck", "medication"):
            db.persist_report(
                session=session,
                agent_type=agent,
                subject_text="Aspirin",
                files={"summary": f"/tmp/{agent}.md"},
                user=user,
            )
        db.persist_report(
            session=session,
            agent_type="factcheck",
            subject_text="Ibuprofen",
            files={"summary": "/tmp/ibu.md"},
            user=user,
        )

    with db.session_scope() as session:
        aspirin_reports = db.list_reports_by_subject_name(session, "aspirin")
        assert len(aspirin_reports) == 2
        all_reports = db.list_reports(session)
        assert len(all_reports) == 3
        factcheck_only = db.list_reports(session, agent_type="factcheck")
        assert len(factcheck_only) == 2


def test_persist_report_without_user_raises(db, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    with db.session_scope() as session, pytest.raises(ValueError):
        db.persist_report(
            session=session,
            agent_type="factcheck",
            subject_text="X",
            files={},
            user=None,
        )


def test_patient_data_schema(db):
    with db.session_scope() as session:
        user = db.get_current_user(session)
        record = db.create_patient_data(
            session=session,
            user=user,
            data={"labs": {"vitamin_d": "18 ng/mL"}},
            source_type="pdf",
            source_reference="bloodwork.pdf",
        )
        record_id = record.id

    with db.session_scope() as session:
        rows = db.list_patient_data(session, user_id=db.DEVELOPER_USER_ID)
        assert len(rows) == 1
        assert rows[0].id == record_id
        assert rows[0].source_type == "pdf"
        assert rows[0].data["labs"]["vitamin_d"] == "18 ng/mL"


def test_is_db_enabled_toggle(monkeypatch):
    import database

    monkeypatch.delenv("DB_PERSISTENCE_ENABLED", raising=False)
    assert database.is_db_enabled() is True
    monkeypatch.setenv("DB_PERSISTENCE_ENABLED", "false")
    assert database.is_db_enabled() is False
    monkeypatch.setenv("DB_PERSISTENCE_ENABLED", "0")
    assert database.is_db_enabled() is False
    monkeypatch.setenv("DB_PERSISTENCE_ENABLED", "true")
    assert database.is_db_enabled() is True


def test_default_database_url_is_local_sqlite(monkeypatch):
    import database

    monkeypatch.delenv("DATABASE_URL", raising=False)
    url = database.get_database_url()
    assert url.startswith("sqlite:///")
    assert url.endswith("app.db")
