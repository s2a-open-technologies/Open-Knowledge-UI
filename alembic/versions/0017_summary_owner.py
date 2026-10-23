"""Eigentuemer von Analysen

Revision ID: 0017
Revises: 0016
Create Date: 2026-10-12

user_summary_files.user_id: Zuordnung der Analyse-Eintraege zum Ersteller (Analysen liegen
pro Nutzer in eigenen Ordnern). Altbestand bleibt ohne Eigentuemer (NULL).

Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {
        c["name"] for c in sa.inspect(op.get_bind()).get_columns("user_summary_files")
    }
    if "user_id" not in cols:
        with op.batch_alter_table("user_summary_files") as batch_op:
            batch_op.add_column(sa.Column("user_id", sa.Integer, nullable=True))
            batch_op.create_foreign_key(
                "fk_user_summary_files_user_id", "users", ["user_id"], ["id"]
            )
            batch_op.create_index("ix_user_summary_files_user_id", ["user_id"])


def downgrade() -> None:
    with op.batch_alter_table("user_summary_files") as batch_op:
        batch_op.drop_column("user_id")
