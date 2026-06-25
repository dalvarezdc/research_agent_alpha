# Database — Local Persistence Layer

SQLAlchemy 2.0 persistence for generated reports, classified by subject and
associated with users. Local-first (SQLite), swappable to Postgres via a single
connection URL.

## Design principle: additive and optional

The database layer **never** sits in the critical path of report generation.
Agents always write their artifacts to `outputs/` first; persistence happens
afterward as a **best-effort** step. If the DB is disabled, missing, or errors,
the run logs a warning and continues. This keeps local, file-only debugging
fully functional with no DB connection.

```
run_*  →  files written to outputs/  →  _persist_report_to_db()  (best-effort, gated)
                  (source of truth)            (optional metadata)
```

## Quick start

```python
import database

# Create tables (idempotent) and seed the default 'developer' user.
database.init_db()

with database.session_scope() as session:
    user = database.get_current_user(session)          # 'developer' locally
    report = database.persist_report(
        session=session,
        agent_type="factcheck",
        subject_text="Vitamin D",
        files={"summary": "outputs/vitamin_d_summary.md"},
        user=user,
        llm_provider="grok-4.3",
        implementation="langchain",
        cost_summary={"total_cost": 0.42},
    )
    print(report.id)  # UUID4 string
```

## Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `sqlite:///data/app.db` | SQLAlchemy connection URL. Set to e.g. `postgresql+psycopg://user:pass@host:5432/db` for Postgres. |
| `DB_PERSISTENCE_ENABLED` | enabled | Set to `false`/`0`/`no`/`off` for pure file-only mode (no DB writes). The single switch for local debugging. |
| `APP_ENV` | `local` | `local`/`dev`/`development` → the acting user is the seeded `developer`. Any other value → `get_current_user()` returns `None` until the login spec wires real auth. |

The default SQLite file lives at `<repo>/data/app.db` (gitignored); the `data/`
directory is created on demand.

## Schema

All primary keys are **UUID4 strings** for global uniqueness and safe exposure.

| Model | Table | Purpose |
|-------|-------|---------|
| `User` | `users` | Accounts. The seeded `developer` user has no `password_hash` (no-login local account). |
| `Subject` | `subjects` | Normalized subject (`name` = lowercased/trimmed, `display_name` = original). Reports are classified by subject. |
| `Report` | `reports` | One analysis run: `agent_type`, `subject_text`, `llm_provider`, `implementation`, `cost_summary` (JSON), `total_cost`, FK to user + subject. |
| `ReportFile` | `report_files` | One on-disk artifact per run, by path (`file_type` → `file_path`). Unique per `(report_id, file_type)`. |
| `PatientData` | `patient_data` | Structured patient info parsed from attached medical reports. Flexible JSON `data` column + typed fields (`source_type`, `source_reference`, `recorded_at`). Linked to user; optionally to report/subject. |

Relationships: `User 1─* Report 1─* ReportFile`, `Subject 1─* Report`,
`User 1─* PatientData` (optionally `→ Report`/`Subject`).

> **Note:** `PatientData` is schema-only for now. Automatic population from
> attached PDFs/images arrives with the medical-parsing spec.

## The `developer` user (no-login local mode)

On `init_db()` a developer account is seeded with a fixed, well-known id:

```python
DEVELOPER_USER_ID  = "00000000-0000-0000-0000-000000000001"
DEVELOPER_USERNAME = "developer"
```

`get_current_user(session)` is the **seam** the future login spec replaces:
in local mode it returns the developer user; otherwise it returns `None`.

## Module map

| File | Responsibility |
|------|----------------|
| `config.py` | URL resolution, singleton engine/sessionmaker, `is_db_enabled()`, `reset_engine_cache()` |
| `models.py` | SQLAlchemy 2.0 ORM models + declarative `Base` |
| `session.py` | `init_db()`, `ensure_initialized()`, transactional `session_scope()` |
| `seed.py` | `seed_developer_user()` and developer id/username constants |
| `users.py` | `get_current_user()` resolver (env-gated) |
| `repository.py` | `persist_report()` and read/create helpers |
| `__init__.py` | Public API surface |

## Public API

```python
# lifecycle
init_db(seed=True); ensure_initialized(); session_scope()
# config
get_database_url(); is_db_enabled(); get_engine(); get_session_factory()
# users / seed
get_current_user(session); seed_developer_user(session)
# write
persist_report(session=..., agent_type=..., subject_text=..., files=..., user=...,
               llm_provider=..., implementation=..., cost_summary=...)
create_patient_data(session=..., user=..., data=..., source_type=..., ...)
# read
list_reports(session, user_id=..., subject_id=..., agent_type=..., limit=100)
get_report(session, report_id)
list_subjects(session, limit=500)
list_reports_by_subject_name(session, "vitamin d")
list_patient_data(session, user_id=..., limit=100)
get_or_create_subject(session, "Vitamin D"); normalize_subject("  Vitamin D ")
```

## Migrations (Alembic)

Local development can rely on `init_db()` (`create_all`). For managed/Postgres
environments use Alembic — `alembic/env.py` is wired to the app's models and
`DATABASE_URL`.

```bash
uv run alembic revision --autogenerate -m "describe change"
uv run alembic upgrade head
```

## Testing

```bash
uv run python -m pytest tests/test_database.py tests/test_orchestrator_db_persistence.py -v
```

Tests point `DATABASE_URL` at a temp SQLite file and call `reset_engine_cache()`
+ `reset_initialized_flag()` for isolation.
