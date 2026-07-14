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


# ── Cost extraction (_extract_total_cost) ────────────────────────────────────
# Locks down the cost-tracking contract: guards against silent drift between
# cost_tracker.get_summary() and what persist_report writes to Report.total_cost.


def test_extract_total_cost_from_realistic_cost_summary():
    """The real cost_tracker.get_summary() shape must yield total_cost."""
    from database.repository import _extract_total_cost

    # Mirrors CostTracker.get_summary() (cost_tracker.py) exactly.
    summary = {
        "total_cost": 1.2345,
        "total_duration": 42.0,
        "phases": [
            {"phase": "Phase 1", "cost": 0.5, "duration": 20.0},
            {"phase": "Phase 2", "cost": 0.7345, "duration": 22.0},
        ],
        "most_expensive": [
            {"phase": "Phase 2", "cost": 0.7345, "duration": 22.0},
        ],
    }
    assert _extract_total_cost(summary) == pytest.approx(1.2345)


@pytest.mark.parametrize(
    "summary,expected",
    [
        ({"total_cost": 3.5}, 3.5),          # primary key
        ({"total": 2.0}, 2.0),               # fallback: total
        ({"cost_usd": 1.5}, 1.5),            # fallback: cost_usd
        ({"grand_total": 9.9}, 9.9),         # fallback: grand_total
        ({"total_cost": 7}, 7.0),            # int is coerced to float
        ({"total_cost": True}, 1.0),         # bool is an int subclass (edge)
    ],
)
def test_extract_total_cost_key_variants(summary, expected):
    from database.repository import _extract_total_cost

    result = _extract_total_cost(summary)
    assert result == pytest.approx(expected)
    assert isinstance(result, float)


def test_extract_total_cost_prefers_primary_key_over_fallbacks():
    """total_cost wins even when fallback keys are also present."""
    from database.repository import _extract_total_cost

    summary = {"total_cost": 1.0, "total": 99.0, "grand_total": 77.0}
    assert _extract_total_cost(summary) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "summary",
    [
        None,                       # not a dict
        "not-a-dict",               # not a dict
        123,                        # not a dict
        {},                         # empty dict
        {"phases": [], "note": 1},  # no recognized cost key
        {"total_cost": "1.23"},     # value present but non-numeric (string)
        {"total_cost": None},       # value present but None
    ],
)
def test_extract_total_cost_returns_none_when_unavailable(summary):
    from database.repository import _extract_total_cost

    assert _extract_total_cost(summary) is None


def test_persist_report_with_no_cost_summary_leaves_total_cost_none(db):
    """End-to-end: a run with no usable cost summary stores NULL total_cost."""
    with db.session_scope() as session:
        user = db.get_current_user(session)
        report = db.persist_report(
            session=session,
            agent_type="diagnostic",
            subject_text="fatigue",
            files={"analysis": "/tmp/dx.json"},
            user=user,
            cost_summary=None,
        )
        report_id = report.id

    with db.session_scope() as session:
        report = db.get_report(session, report_id)
        assert report is not None
        assert report.total_cost is None


# ── Developer username accessor ──────────────────────────────────────────────


def test_get_developer_username_matches_seed_constant():
    from database.seed import DEVELOPER_USERNAME
    from database.users import get_developer_username

    assert get_developer_username() == DEVELOPER_USERNAME
    assert get_developer_username() == "developer"


# ── persist_report / list edge behaviors ─────────────────────────────────────


def test_persist_report_skips_empty_file_paths(db):
    """Files with empty/blank paths must not create ReportFile rows."""
    with db.session_scope() as session:
        user = db.get_current_user(session)
        report = db.persist_report(
            session=session,
            agent_type="medication",
            subject_text="Metformin",
            files={"summary": "/tmp/ok.md", "pdf": "", "audit": None},
            user=user,
        )
        report_id = report.id

    with db.session_scope() as session:
        report = db.get_report(session, report_id)
        # Only the non-empty "summary" entry becomes a row.
        assert {f.file_type for f in report.files} == {"summary"}


def test_list_reports_by_subject_name_unknown_returns_empty(db):
    """Querying an unseen subject returns an empty list, not an error."""
    with db.session_scope() as session:
        assert db.list_reports_by_subject_name(session, "never-analyzed") == []


def test_list_subjects_returns_created_subjects(db):
    """list_subjects returns subjects created via persist_report."""
    with db.session_scope() as session:
        user = db.get_current_user(session)
        for name in ("Vitamin D", "Magnesium"):
            db.persist_report(
                session=session,
                agent_type="factcheck",
                subject_text=name,
                files={"summary": f"/tmp/{name}.md"},
                user=user,
            )

    with db.session_scope() as session:
        subjects = db.list_subjects(session)
        names = {s.name for s in subjects}
        assert {"vitamin d", "magnesium"} <= names
