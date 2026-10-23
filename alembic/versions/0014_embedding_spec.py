"""Embedding-Modell pro Knowledge, globale Modell-Einstellungen

Revision ID: 0014
Revises: 0013
Create Date: 2026-10-12

knowledge.embedding_spec / index_signature / last_reindex_error: womit der Index entstanden
ist (Modell, Chunk-Einstellungen, Versionen) und der letzte Fehler beim Neu-Indexieren.

Bestehende Installationen haben ihre Vektoren mit fastembed erzeugt: Knowledge und die neuen
globalen Settings EMBEDDING_SPEC/RERANKER_SPEC werden darauf gesetzt, es muss nichts neu
indexiert werden. Reranker-Modelle in den Profilen werden zu "default" (global).

Idempotent (siehe Migration 0008).
"""

import json

import sqlalchemy as sa

from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None

LEGACY_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LEGACY_RERANKER = "BAAI/bge-reranker-base"


def upgrade() -> None:
    bind = op.get_bind()
    cols = {c["name"] for c in sa.inspect(bind).get_columns("knowledge")}
    with op.batch_alter_table("knowledge") as batch_op:
        if "embedding_spec" not in cols:
            batch_op.add_column(sa.Column("embedding_spec", sa.Text, nullable=True))
        if "index_signature" not in cols:
            batch_op.add_column(
                sa.Column("index_signature", sa.String(32), nullable=True)
            )
        if "last_reindex_error" not in cols:
            batch_op.add_column(sa.Column("last_reindex_error", sa.Text, nullable=True))

    settings = {
        r[0]: r[1] for r in bind.execute(sa.text("SELECT key, value FROM settings"))
    }
    legacy = "EMBEDDING_MODEL" in settings  # Datenbank einer bestehenden Installation
    model = settings.get("EMBEDDING_MODEL") or LEGACY_MODEL

    if legacy:
        for key, spec in (
            ("EMBEDDING_SPEC", {"source": "fastembed", "model": model}),
            ("RERANKER_SPEC", {"source": "fastembed", "model": LEGACY_RERANKER}),
        ):
            if key not in settings:
                bind.execute(
                    sa.text(
                        "INSERT INTO settings (key, value, default_value, category, is_sensitive, description) "
                        "VALUES (:k, :v, :v, 'embedding', 0, :d)"
                    ),
                    {
                        "k": key,
                        "v": json.dumps(spec),
                        "d": "Modell (JSON), Verwaltung unter Admin > Einlesen & Modelle",
                    },
                )
        bind.execute(
            sa.text(
                "UPDATE knowledge SET embedding_spec = :s, embedding_model = COALESCE(embedding_model, :m) "
                "WHERE embedding_spec IS NULL"
            ),
            {"s": json.dumps({"source": "fastembed", "model": model}), "m": model},
        )

    tables = sa.inspect(bind).get_table_names()
    if "retrieval_profiles" in tables:
        for pid, raw in bind.execute(
            sa.text("SELECT id, config FROM retrieval_profiles")
        ).fetchall():
            try:
                cfg = json.loads(raw)
                if cfg.get("rerank", {}).get("model") not in (None, "default"):
                    cfg["rerank"]["model"] = "default"
                    bind.execute(
                        sa.text(
                            "UPDATE retrieval_profiles SET config = :c WHERE id = :i"
                        ),
                        {"c": json.dumps(cfg), "i": pid},
                    )
            except (ValueError, AttributeError):
                continue


def downgrade() -> None:
    with op.batch_alter_table("knowledge") as batch_op:
        batch_op.drop_column("last_reindex_error")
        batch_op.drop_column("index_signature")
        batch_op.drop_column("embedding_spec")
