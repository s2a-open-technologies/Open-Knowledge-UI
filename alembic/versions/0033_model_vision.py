"""Bilder direkt an das Chat-Modell (model_configs.vision)

Revision ID: 0033
Revises: 0032

Ob das Modell Bilder versteht: dann gehen Bild-Anhaenge zusaetzlich zum gelesenen Text direkt an
das Modell. Standard aus. Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0033"
down_revision = "0032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("model_configs")}
    if "vision" not in cols:
        with op.batch_alter_table("model_configs") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "vision", sa.Boolean, nullable=False, server_default=sa.false()
                )
            )


def downgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("model_configs")}
    if "vision" in cols:
        with op.batch_alter_table("model_configs") as batch_op:
            batch_op.drop_column("vision")
