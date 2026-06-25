"""Current-user resolution.

This is the seam the future login spec will replace. For now, when running
locally (no auth configured), the acting user is the seeded ``developer``
account. Real authentication will later override ``get_current_user`` to return
the logged-in account instead.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from sqlalchemy.orm import Session

from .models import User
from .seed import DEVELOPER_USERNAME, seed_developer_user

logger = logging.getLogger(__name__)


def _is_local_mode() -> bool:
    """True when running in the local/no-auth profile.

    Defaults to local. Set ``APP_ENV`` to anything other than "local"/"dev"/
    "development" to opt out (the login spec will gate real auth on this).
    """
    env = os.getenv("APP_ENV", "local").strip().lower()
    return env in {"local", "dev", "development", ""}


def get_current_user(session: Session) -> Optional[User]:
    """Resolve the acting user for the current operation.

    Local mode -> the seeded developer user (created on demand if absent).
    Non-local mode -> None for now; the login spec will supply the real user.
    """
    if _is_local_mode():
        return seed_developer_user(session)

    logger.debug(
        "Non-local APP_ENV and no auth wired yet; no current user resolved."
    )
    return None


def get_developer_username() -> str:
    """Expose the developer username for callers/tests."""
    return DEVELOPER_USERNAME
