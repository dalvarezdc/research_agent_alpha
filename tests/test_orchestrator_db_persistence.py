#!/usr/bin/env python3
"""Tests for AgentOrchestrator best-effort DB persistence wiring.

These verify the integration seam without invoking real agents/LLMs:
files-first behavior, gating by DB-enabled flag, and never-raises semantics.
"""

import pytest


@pytest.fixture()
def db_env(tmp_path, monkeypatch):
    """Isolated DB pointed at a temp SQLite file."""
    db_path = tmp_path / "orch_app.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.delenv("DB_PERSISTENCE_ENABLED", raising=False)
    monkeypatch.setenv("APP_ENV", "local")

    import database

    database.reset_engine_cache()
    database.reset_initialized_flag()
    yield database
    database.reset_engine_cache()
    database.reset_initialized_flag()


def _make_orchestrator(tmp_path):
    from run_analysis import AgentOrchestrator

    return AgentOrchestrator(output_dir=str(tmp_path / "outputs"))


def test_persist_helper_writes_report(db_env, tmp_path):
    orch = _make_orchestrator(tmp_path)
    files = {"summary": str(tmp_path / "s.md"), "cost": str(tmp_path / "c.json")}

    orch._persist_report_to_db(
        agent_type="medication",
        subject_text="Metformin",
        files=files,
        llm_provider="claude-sonnet",
        implementation="langchain",
        cost_summary={"total_cost": 1.23},
    )

    with db_env.session_scope() as session:
        reports = db_env.list_reports(session)
        assert len(reports) == 1
        report = reports[0]
        assert report.agent_type == "medication"
        assert report.subject_text == "Metformin"
        assert report.user_id == db_env.DEVELOPER_USER_ID
        assert {f.file_type for f in report.files} == {"summary", "cost"}


def test_persist_disabled_is_noop(db_env, tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PERSISTENCE_ENABLED", "false")
    orch = _make_orchestrator(tmp_path)

    # Must not raise and must not create the schema/rows.
    orch._persist_report_to_db(
        agent_type="factcheck",
        subject_text="Vitamin C",
        files={"summary": str(tmp_path / "s.md")},
    )

    # Re-enable and confirm nothing was persisted while disabled.
    monkeypatch.delenv("DB_PERSISTENCE_ENABLED", raising=False)
    db_env.reset_engine_cache()
    db_env.reset_initialized_flag()
    db_env.init_db(seed=True)
    with db_env.session_scope() as session:
        assert db_env.list_reports(session) == []


def test_persist_never_raises_on_internal_error(db_env, tmp_path, monkeypatch):
    """Even if persistence blows up internally, the run must continue."""
    orch = _make_orchestrator(tmp_path)

    import database

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated DB failure")

    # The orchestrator does `from database import persist_report`, so patch the
    # name on the database package namespace it resolves at call time.
    monkeypatch.setattr(database, "persist_report", _boom)

    # Should swallow the error (best-effort) and return normally.
    orch._persist_report_to_db(
        agent_type="procedure",
        subject_text="Appendectomy",
        files={"summary": str(tmp_path / "s.md")},
    )
