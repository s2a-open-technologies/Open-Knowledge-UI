"""Gespeichertes Markdown pro Dokument

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-12

user_documents.markdown_path / parser_version: Pfad des beim Einlesen erzeugten
Markdown (Basis fuer Neu-Indexieren und die Ansicht im UI). Bestehende Dokumente
haben keins (NULL); sie werden beim Neu-Indexieren aus dem archivierten Original
neu geparst.

Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("user_documents")}
    with op.batch_alter_table("user_documents") as batch_op:
        if "markdown_path" not in cols:
            batch_op.add_column(
                sa.Column("markdown_path", sa.String(500), nullable=True)
            )
        if "parser_version" not in cols:
            batch_op.add_column(sa.Column("parser_version", sa.Integer, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("user_documents") as batch_op:
        batch_op.drop_column("parser_version")
        batch_op.drop_column("markdown_path")
