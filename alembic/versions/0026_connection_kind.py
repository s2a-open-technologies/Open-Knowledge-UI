"""Connections: Typ (llm | tei | infinity)

Revision ID: 0026
Revises: 0025
Create Date: 2026-10-13

connections.kind: 'llm' (OpenAI-kompatibel, bisheriges Verhalten), 'tei' (Text Embeddings Inference)
oder 'infinity'. Bestehende Connections bleiben 'llm'.

Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("connections")}
    if "kind" not in cols:
        with op.batch_alter_table("connections") as batch_op:
            batch_op.add_column(
                sa.Column("kind", sa.String(20), nullable=False, server_default="llm")
            )


def downgrade() -> None:
    with op.batch_alter_table("connections") as batch_op:
        batch_op.drop_column("kind")
