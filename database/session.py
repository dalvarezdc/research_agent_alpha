"""Session lifecycle and schema initialization."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy.orm import Session

from .config import get_engine, get_session_factory
from .models import Base

logger = logging.getLogger(__name__)

_initialized = False


def init_db(*, seed: bool = True) -> None:
    """Create all tables (idempotent) and optionally seed the developer user.

    Safe to call multiple times. Uses ``create_all`` for local SQLite; Alembic
    migrations are available for managed/Postgres environments.
    """
    global _initialized
    engine = get_engine()
    Base.metadata.create_all(engine)
    _initialized = True
    logger.debug("Database schema ensured for %s", engine.url)

    if seed:
        # Imported lazily to avoid a circular import (seed -> session).
        from .seed import seed_developer_user

        with session_scope() as session:
            seed_developer_user(session)


def ensure_initialized() -> None:
    """Initialize the schema once per process if not already done."""
    if not _initialized:
        init_db(seed=True)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope around a series of operations.

    Commits on success, rolls back on exception, and always closes.
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_initialized_flag() -> None:
    """Reset the one-time init guard (test support)."""
    global _initialized
    _initialized = False
