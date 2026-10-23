"""Initial schema and admin seed

Revision ID: 0001
Revises:
Create Date: 2026-06-30

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # users
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("username", sa.String(100), nullable=False, unique=True, index=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("is_admin", sa.Boolean(), default=False),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
    )

    # sessions
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("title", sa.String(255)),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), default=sa.func.now()),
    )

    # messages
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(36),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
    )

    # user_documents
    op.create_table(
        "user_documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("document_id", sa.String(100), nullable=False, index=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(), default=sa.func.now()),
    )

    # settings
    op.create_table(
        "settings",
        sa.Column("key", sa.String(100), primary_key=True),
        sa.Column("value", sa.Text()),
        sa.Column("default_value", sa.Text()),
        sa.Column("category", sa.String(50), index=True),
        sa.Column("is_sensitive", sa.Boolean(), default=False),
        sa.Column("description", sa.Text()),
    )

    # user_summary_files
    op.create_table(
        "user_summary_files",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
    )

    # Admin-Seed: admin / admin
    op.get_bind().execute(
        sa.text(
            "INSERT INTO users (username, password_hash, is_admin) "
            "VALUES ('admin', :pw, :yes)"
        ),
        {"pw": _hash("admin"), "yes": True},
    )


def downgrade() -> None:
    op.drop_table("user_summary_files")
    op.drop_table("settings")
    op.drop_table("user_documents")
    op.drop_table("messages")
    op.drop_table("sessions")
    op.drop_table("users")


def _hash(password: str) -> str:
    from werkzeug.security import generate_password_hash

    return generate_password_hash(password)
