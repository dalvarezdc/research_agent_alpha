"""Add durable chats and link existing analysis history without changing artifacts."""
from alembic import op
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

revision = "20260914_saved_chats"
down_revision = "845c57de9b5a"
branch_labels = None
depends_on = None


def upgrade():
    from database.models import Base
    from database.chats import migrate_history
    connection = op.get_bind()
    Base.metadata.create_all(connection)
    columns = {column["name"] for column in inspect(connection).get_columns("conversations")}
    if "patient_id" not in columns:
        connection.execute(text("ALTER TABLE conversations ADD COLUMN patient_id VARCHAR(36)"))
    if "chat_id" not in columns:
        connection.execute(text("ALTER TABLE conversations ADD COLUMN chat_id VARCHAR(36) REFERENCES saved_chats(id)"))
    with Session(bind=connection) as session:
        migrate_history(session)
        session.commit()


def downgrade():
    raise RuntimeError("Saved chats contain user history; export it before performing a manual downgrade.")
