"""Audit-Log-Tabelle fuer sicherheitsrelevante Ereignisse

Revision ID: 0036
Revises: 0035

Ein Anhang (append) auf ``audit_log``: Anmeldung (ok/falsch/gesperrt), Abmeldung,
Kontoverwaltung (Anlegen/Aendern/Loeschen, Rolle/Status/Passwort), Gruppen und
Freigaben, Passwortwechsel und der persoenliche Datenexport. Best-effort beim
Schreiben (siehe ``app/auth/audit.py``) — die Tabelle ist bewusst ohne
UNIQUE/NOT NULL-Zwänge angelegt, damit ein unvollstaendiger Eintrag nie den
eigentlichen Vorgang blockiert.
"""

import sqlalchemy as sa

from alembic import op

revision = "0036"
down_revision = "0035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column(
            "actor_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True
        ),
        sa.Column("actor_name", sa.String(80), nullable=True),
        sa.Column("ip", sa.String(64), nullable=True),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("object_type", sa.String(64), nullable=True),
        sa.Column("object_id", sa.String(100), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
    )
    op.create_index("ix_audit_log_created_at", "audit_log", ["created_at"])
    op.create_index("ix_audit_log_actor_user_id", "audit_log", ["actor_user_id"])
    op.create_index("ix_audit_log_action", "audit_log", ["action"])
    op.create_index("ix_audit_log_object_type", "audit_log", ["object_type"])


def downgrade() -> None:
    op.drop_index("ix_audit_log_object_type", table_name="audit_log")
    op.drop_index("ix_audit_log_action", table_name="audit_log")
    op.drop_index("ix_audit_log_actor_user_id", table_name="audit_log")
    op.drop_index("ix_audit_log_created_at", table_name="audit_log")
    op.drop_table("audit_log")
