"""Websuche pro Modell

Revision ID: 0015
Revises: 0014
Create Date: 2026-10-12

model_configs.web_search: ob das Websuche-Werkzeug (SearXNG) an dem Modell haengt.
Standard aus; zuschaltbar nur mit eingerichtetem PII-Filter. Die globale Einstellung
SEARCH_SEARXNG_ENABLED entfaellt; bisher aktive Websuche muss pro Modell neu zugeschaltet
werden.

Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    cols = {c["name"] for c in sa.inspect(op.get_bind()).get_columns("model_configs")}
    if "web_search" not in cols:
        with op.batch_alter_table("model_configs") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "web_search", sa.Boolean, nullable=False, server_default=sa.false()
                )
            )


def downgrade() -> None:
    with op.batch_alter_table("model_configs") as batch_op:
        batch_op.drop_column("web_search")
