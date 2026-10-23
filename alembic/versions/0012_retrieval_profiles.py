"""Retrieval-Profile

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-12

- retrieval_profiles: benannte Abruf-Konfigurationen (JSON).
- knowledge.retrieval_profile_id: Zuordnung am Knowledge (NULL = Standardprofil).

Die Profile "Standard" und "Hohe Qualität" werden zur Laufzeit gesäet
(``app.rag.profiles.ensure_default_profiles``), weil der Standard aus den
RETRIEVER_*-Settings abgeleitet wird, die bei frischen DBs in Migrationen noch fehlen.
Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "retrieval_profiles" not in set(insp.get_table_names()):
        op.create_table(
            "retrieval_profiles",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("name", sa.String(80), nullable=False, unique=True),
            sa.Column("description", sa.Text),
            sa.Column("config", sa.Text, nullable=False),
            sa.Column(
                "is_default", sa.Boolean, nullable=False, server_default=sa.false()
            ),
            sa.Column("created_at", sa.DateTime),
        )

    if "retrieval_profile_id" not in {c["name"] for c in insp.get_columns("knowledge")}:
        with op.batch_alter_table("knowledge") as batch_op:
            batch_op.add_column(
                sa.Column("retrieval_profile_id", sa.Integer, nullable=True)
            )
            batch_op.create_foreign_key(
                "fk_knowledge_retrieval_profile_id",
                "retrieval_profiles",
                ["retrieval_profile_id"],
                ["id"],
            )


def downgrade() -> None:
    with op.batch_alter_table("knowledge") as batch_op:
        batch_op.drop_constraint(
            "fk_knowledge_retrieval_profile_id", type_="foreignkey"
        )
        batch_op.drop_column("retrieval_profile_id")
    op.drop_table("retrieval_profiles")
