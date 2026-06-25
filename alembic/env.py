"""Alembic migration environment.

Resolves the database URL and target metadata from the application's database
package so migrations stay in sync with the ORM models. Use this for managed /
Postgres environments; local SQLite can also use ``database.init_db`` directly.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Application database package.
from database.config import get_database_url
from database.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Always use the app's resolved URL (env-driven), overriding the ini placeholder.
config.set_main_option("sqlalchemy.url", get_database_url())

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (emit SQL)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=url.startswith("sqlite"),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        is_sqlite = connection.dialect.name == "sqlite"
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=is_sqlite,  # batch mode for SQLite ALTERs
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
