"""Benutzer mit Rollen, Chats pro Benutzer

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-12

- users: Benutzername, Passwort-Hash, Rolle (admin|editor|user), Aktiv-Flag.
- sessions.user_id: Besitzer eines Chats.

Der Standard-Admin wird zur Laufzeit gesaeet (``app.users.ensure_admin_user``),
weil das Passwort aus APP_PASSWORD_HASH stammt, das bei frischen Installationen
erst dann in der settings-Tabelle steht. Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "users" not in set(insp.get_table_names()):
        op.create_table(
            "users",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("username", sa.String(80), nullable=False, unique=True),
            sa.Column("display_name", sa.String(120)),
            sa.Column("password_hash", sa.String(255), nullable=False),
            sa.Column("role", sa.String(20), nullable=False, server_default="user"),
            sa.Column(
                "is_active", sa.Boolean, nullable=False, server_default=sa.true()
            ),
            sa.Column("created_at", sa.DateTime),
        )

    if "user_id" not in {c["name"] for c in insp.get_columns("sessions")}:
        with op.batch_alter_table("sessions") as batch_op:
            batch_op.add_column(sa.Column("user_id", sa.Integer, nullable=True))
            batch_op.create_index("ix_sessions_user_id", ["user_id"])
            batch_op.create_foreign_key(
                "fk_sessions_user_id_users", "users", ["user_id"], ["id"]
            )


def downgrade() -> None:
    with op.batch_alter_table("sessions") as batch_op:
        batch_op.drop_constraint("fk_sessions_user_id_users", type_="foreignkey")
        batch_op.drop_index("ix_sessions_user_id")
        batch_op.drop_column("user_id")
    op.drop_table("users")
