"""Inhaltsverzeichnis direkt am Knowledge

Revision ID: 0029
Revises: 0028
Create Date: 2026-10-13

knowledge.index_md / knowledge.index_updated_at: Das Inhaltsverzeichnis (frueher eine Datei index.md im
Markdown-Ordner) haengt jetzt am Knowledge-Eintrag, damit ein Chat immer genau das Verzeichnis des
gewaehlten Knowledge liest. Es wird beim ersten Zugriff bzw. nach jedem Einlesen neu erzeugt; alte Dateien
bleiben ungenutzt liegen und werden beim naechsten Erzeugen entfernt.

Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0029"
down_revision = "0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("knowledge")}
    with op.batch_alter_table("knowledge") as batch_op:
        if "index_md" not in cols:
            batch_op.add_column(sa.Column("index_md", sa.Text, nullable=True))
        if "index_updated_at" not in cols:
            batch_op.add_column(
                sa.Column("index_updated_at", sa.DateTime, nullable=True)
            )


def downgrade() -> None:
    with op.batch_alter_table("knowledge") as batch_op:
        batch_op.drop_column("index_updated_at")
        batch_op.drop_column("index_md")
