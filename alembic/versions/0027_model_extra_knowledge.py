"""Modelle: zusaetzliches Knowledge der Nutzer erlauben

Revision ID: 0027
Revises: 0026
Create Date: 2026-10-13

model_configs.allow_extra_knowledge (Standard an): Das Knowledge eines Modells ist die Basis jeder Suche;
Nutzer duerfen zusaetzlich eigenes Knowledge waehlen. Aus = das Modell durchsucht nur sein eigenes.

Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0027"
down_revision = "0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("model_configs")}
    if "allow_extra_knowledge" not in cols:
        with op.batch_alter_table("model_configs") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "allow_extra_knowledge",
                    sa.Boolean,
                    nullable=False,
                    server_default=sa.true(),
                )
            )


def downgrade() -> None:
    with op.batch_alter_table("model_configs") as batch_op:
        batch_op.drop_column("allow_extra_knowledge")
