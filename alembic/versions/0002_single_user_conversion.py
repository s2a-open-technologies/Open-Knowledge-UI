"""Single-user conversion

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-09

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Admin-Passwort übernehmen, damit das Login-Passwort gleich bleibt.
    admin_pw = conn.execute(
        sa.text("SELECT password_hash FROM users WHERE is_admin = :yes LIMIT 1"),
        {"yes": True},
    ).scalar()
    if admin_pw:
        conn.execute(
            sa.text(
                "INSERT INTO settings (key, value, default_value, category, is_sensitive, description) "
                "VALUES ('APP_PASSWORD_HASH', :pw, '', 'auth', :yes, 'Gehashtes App-Passwort') "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            ),
            {"pw": admin_pw, "yes": True},
        )

    # user_id FKs und Spalten entfernen (SQLite-kompatibel via batch_alter_table)
    with op.batch_alter_table("sessions") as batch_op:
        batch_op.drop_index("ix_sessions_user_id")
        batch_op.drop_column("user_id")

    with op.batch_alter_table("user_documents") as batch_op:
        batch_op.drop_index("ix_user_documents_user_id")
        batch_op.drop_column("user_id")

    with op.batch_alter_table("user_summary_files") as batch_op:
        batch_op.drop_index("ix_user_summary_files_user_id")
        batch_op.drop_column("user_id")

    op.drop_table("users")

    # Neue STT-Settings vorbelegen (falls noch nicht vorhanden)
    conn.execute(
        sa.text(
            "INSERT INTO settings (key, value, default_value, category, is_sensitive, description) "
            "VALUES "
            "('STT_MODEL', 'nemo-parakeet-tdt-0.6b-v3', 'nemo-parakeet-tdt-0.6b-v3', 'stt', false, 'STT Modell'), "
            "('STT_BASEURL', 'http://parakeet:5001/v1', 'http://parakeet:5001/v1', 'stt', false, 'STT API URL'), "
            "('STT_API_KEY', 'ollama', 'ollama', 'stt', true, 'STT API Key') "
            "ON CONFLICT(key) DO NOTHING"
        )
    )


def downgrade() -> None:
    conn = op.get_bind()

    # users-Tabelle wiederherstellen
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("username", sa.String(100), nullable=False, unique=True, index=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("is_admin", sa.Boolean(), default=False),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
    )

    # Default admin user
    admin_id = conn.execute(
        sa.text(
            "INSERT INTO users (username, password_hash, is_admin) "
            "VALUES ('admin', :pw, :yes) RETURNING id"
        ),
        {"pw": _hash("admin"), "yes": True},
    ).scalar()

    with op.batch_alter_table("sessions") as batch_op:
        batch_op.add_column(sa.Column("user_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_sessions_user_id_users",
            "users",
            ["user_id"],
            ["id"],
            ondelete="CASCADE",
        )
    conn.execute(sa.text("UPDATE sessions SET user_id = :uid"), {"uid": admin_id})
    with op.batch_alter_table("sessions") as batch_op:
        batch_op.alter_column("user_id", nullable=False)

    with op.batch_alter_table("user_documents") as batch_op:
        batch_op.add_column(sa.Column("user_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_user_documents_user_id_users",
            "users",
            ["user_id"],
            ["id"],
            ondelete="CASCADE",
        )
    conn.execute(sa.text("UPDATE user_documents SET user_id = :uid"), {"uid": admin_id})
    with op.batch_alter_table("user_documents") as batch_op:
        batch_op.alter_column("user_id", nullable=False)

    with op.batch_alter_table("user_summary_files") as batch_op:
        batch_op.add_column(sa.Column("user_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_user_summary_files_user_id_users",
            "users",
            ["user_id"],
            ["id"],
            ondelete="CASCADE",
        )
    conn.execute(
        sa.text("UPDATE user_summary_files SET user_id = :uid"), {"uid": admin_id}
    )
    with op.batch_alter_table("user_summary_files") as batch_op:
        batch_op.alter_column("user_id", nullable=False)


def _hash(password: str) -> str:
    from werkzeug.security import generate_password_hash

    return generate_password_hash(password)
