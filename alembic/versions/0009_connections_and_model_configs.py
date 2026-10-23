"""LLM-Connections und Modell-Konfigurationen

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-12

- connections: OpenAI-kompatible LLM-Endpunkte (Name, Base-URL, API-Key).
- model_configs: Prompt, Parameter und Knowledge pro Modell sowie Modell-Klone.

Die Standard-Connection wird zur Laufzeit aus CHAT_BASEURL/API_KEY angelegt
(die settings-Zeilen existieren bei frischen Installationen erst dann).
Idempotent, weil ``DatabaseService`` fehlende Tabellen per ``create_all`` anlegt.
"""

import sqlalchemy as sa

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    tables = set(sa.inspect(op.get_bind()).get_table_names())

    if "connections" not in tables:
        op.create_table(
            "connections",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("name", sa.String(120), nullable=False),
            sa.Column("base_url", sa.String(500), nullable=False),
            sa.Column("api_key", sa.Text),
            sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.true()),
            sa.Column("created_at", sa.DateTime),
        )

    if "model_configs" not in tables:
        op.create_table(
            "model_configs",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column(
                "connection_id",
                sa.Integer,
                sa.ForeignKey("connections.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column("base_model", sa.String(255), nullable=False),
            sa.Column("name", sa.String(120)),
            sa.Column(
                "is_clone", sa.Boolean, nullable=False, server_default=sa.false()
            ),
            sa.Column("visible", sa.Boolean, nullable=False, server_default=sa.true()),
            sa.Column("system_prompt", sa.Text),
            sa.Column(
                "replace_prompt", sa.Boolean, nullable=False, server_default=sa.false()
            ),
            sa.Column("params", sa.Text),
            sa.Column("knowledge_ids", sa.Text),
            sa.Column("created_at", sa.DateTime),
        )


def downgrade() -> None:
    op.drop_table("model_configs")
    op.drop_table("connections")
