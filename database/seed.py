"""Seed data: the default 'developer' user for no-login local use."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import User

logger = logging.getLogger(__name__)

# Fixed, well-known UUID so the developer user is stable across environments and
# can be referenced deterministically (e.g. by the future login spec / tests).
DEVELOPER_USER_ID = "00000000-0000-0000-0000-000000000001"
DEVELOPER_USERNAME = "developer"


def seed_developer_user(session: Session) -> User:
    """Create the developer user if missing; return the existing/created row.

    The developer user has no password_hash, marking it as the no-login local
    account. Idempotent.
    """
    user = session.get(User, DEVELOPER_USER_ID)
    if user is not None:
        return user

    # Guard against a pre-existing row with the same username but different id.
    existing = session.scalar(
        select(User).where(User.username == DEVELOPER_USERNAME)
    )
    if existing is not None:
        return existing

    user = User(
        id=DEVELOPER_USER_ID,
        username=DEVELOPER_USERNAME,
        email=None,
        password_hash=None,
        is_active=True,
    )
    session.add(user)
    session.flush()  # assign/persist within the caller's transaction
    logger.info("Seeded default developer user (id=%s)", DEVELOPER_USER_ID)
    return user
