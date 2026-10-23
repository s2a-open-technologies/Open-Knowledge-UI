"""Sprache der Stichwortsuche (BM25) je Knowledge

Revision ID: 0034
Revises: 0033

knowledge.bm25_language: womit der BM25-Index eines Knowledge entstand (leer = Altbestand =
Englisch, das war bis dahin fest eingestellt). Neue und neu indexierte Knowledge nutzen die
Einstellung BM25_LANGUAGE (Standard Deutsch). Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0034"
down_revision = "0033"
branch_labels = None
depends_on = None

SETTING = (
    "BM25_LANGUAGE",
    "german",
    "Sprache der Stichwortsuche (BM25: Stemmer und Stoppwoerter); gilt fuer neue und neu indexierte Knowledge",
)


def upgrade() -> None:
    bind = op.get_bind()
    cols = {c["name"] for c in sa.inspect(bind).get_columns("knowledge")}
    if "bm25_language" not in cols:
        with op.batch_alter_table("knowledge") as batch_op:
            batch_op.add_column(
                sa.Column("bm25_language", sa.String(20), nullable=True)
            )
    key, value, description = SETTING
    bind.execute(
        sa.text(
            "INSERT INTO settings (key, value, default_value, category, is_sensitive, description) "
            "VALUES (:key, :value, :value, 'ingest', :sensitive, :description) "
            "ON CONFLICT (key) DO NOTHING"
        ),
        {"key": key, "value": value, "sensitive": False, "description": description},
    )


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(sa.text("DELETE FROM settings WHERE key = :key"), {"key": SETTING[0]})
    cols = {c["name"] for c in sa.inspect(bind).get_columns("knowledge")}
    if "bm25_language" in cols:
        with op.batch_alter_table("knowledge") as batch_op:
            batch_op.drop_column("bm25_language")
