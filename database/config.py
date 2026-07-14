"""Database configuration: URL resolution, engine/session factory, enablement.

The DB layer is *additive and optional*. File output (outputs/) never depends on
it. Persistence is best-effort and can be turned off entirely for local,
file-only debugging by setting ``DB_PERSISTENCE_ENABLED=false``.

Environment variables
---------------------
DATABASE_URL
    SQLAlchemy connection URL. Defaults to a local SQLite file at
    ``<repo>/data/app.db``. Swap to Postgres later by setting e.g.
    ``postgresql+psycopg://user:pass@host:5432/dbname``.
DB_PERSISTENCE_ENABLED
    "false"/"0"/"no" disables all DB persistence (file-only mode). Anything
    else (or unset) leaves persistence enabled.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

# Repo root = parent of this file's package directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"
_DEFAULT_SQLITE_PATH = _DEFAULT_DATA_DIR / "app.db"

_FALSEY = {"false", "0", "no", "off", ""}


def get_database_url() -> str:
    """Resolve the SQLAlchemy database URL.

    Defaults to a local SQLite file. The data directory is created on demand so
    a fresh checkout works without manual setup.
    """
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return url

    # Default local SQLite. Ensure the data directory exists.
    _DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{_DEFAULT_SQLITE_PATH}"


def is_db_enabled() -> bool:
    """Return True if DB persistence is enabled.

    Disabled explicitly via ``DB_PERSISTENCE_ENABLED`` being falsey. This is the
    single switch that lets the existing file-only flow run untouched.
    """
    flag = os.getenv("DB_PERSISTENCE_ENABLED")
    if flag is not None and flag.strip().lower() in _FALSEY:
        return False
    return True


def _create_engine(url: str) -> Engine:
    """Create an Engine with sensible defaults for SQLite and others."""
    connect_args = {}
    engine_kwargs = {"future": True, "pool_pre_ping": True}

    if url.startswith("sqlite"):
        # check_same_thread=False allows the engine to be shared across the
        # orchestrator's threads (api.py uses background threads).
        connect_args["check_same_thread"] = False

    return create_engine(url, connect_args=connect_args, **engine_kwargs)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a process-wide singleton Engine for the resolved DATABASE_URL."""
    url = get_database_url()
    logger.debug("Creating SQLAlchemy engine for %s", url)
    return _create_engine(url)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker:
    """Return a process-wide singleton sessionmaker bound to the engine."""
    return sessionmaker(
        bind=get_engine(),
        autoflush=False,
        expire_on_commit=False,
        class_=Session,
        future=True,
    )


def reset_engine_cache() -> None:
    """Dispose and clear cached engine/session factory.

    Used by tests (and when switching DATABASE_URL at runtime) so a new URL
    takes effect cleanly.
    """
    # Dispose the currently cached engine if one exists.
    if get_engine.cache_info().currsize:  # type: ignore[attr-defined]
        existing: Optional[Engine] = get_engine()
        if existing is not None:
            existing.dispose()
    get_engine.cache_clear()
    get_session_factory.cache_clear()
