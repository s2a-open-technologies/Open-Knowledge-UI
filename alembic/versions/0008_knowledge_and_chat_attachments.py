"""Knowledge-Bereiche und Chat-Anhaenge

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-12

- knowledge: ein Bereich pro Qdrant-Collection. Der Bestand (Collection
  ``open_knowledge_ui``) wird zum Standard-Bereich.
- user_documents.knowledge_id: Zuordnung der Dokumente zum Bereich.
- chat_attachments: Dateien, die im Chat hochgeladen wurden (pro Session).

Idempotent: ``DatabaseService`` legt fehlende Tabellen per ``create_all`` an
(AGENTS.md: bestehende app.db braucht sonst ``alembic stamp head``).
"""

import sqlalchemy as sa

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    tables = set(insp.get_table_names())

    if "knowledge" not in tables:
        op.create_table(
            "knowledge",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("name", sa.String(120), nullable=False),
            sa.Column("description", sa.Text),
            sa.Column("collection_name", sa.String(100), nullable=False, unique=True),
            sa.Column(
                "is_default", sa.Boolean, nullable=False, server_default=sa.false()
            ),
            sa.Column("embedding_model", sa.String(255)),
            sa.Column("embedding_dimension", sa.Integer),
            sa.Column("created_at", sa.DateTime),
        )

    default_id = conn.execute(
        sa.text("SELECT id FROM knowledge WHERE collection_name = 'open_knowledge_ui'")
    ).scalar()
    if default_id is None:
        conn.execute(
            sa.text(
                "INSERT INTO knowledge (name, description, collection_name, "
                "is_default, created_at) VALUES ('Standard', "
                "'Bisheriger Wissensbestand', 'open_knowledge_ui', :t, CURRENT_TIMESTAMP)"
            ),
            {"t": True},
        )
        default_id = conn.execute(
            sa.text(
                "SELECT id FROM knowledge WHERE collection_name = 'open_knowledge_ui'"
            )
        ).scalar()

    doc_cols = {c["name"] for c in insp.get_columns("user_documents")}
    if "knowledge_id" not in doc_cols:
        with op.batch_alter_table("user_documents") as batch_op:
            batch_op.add_column(sa.Column("knowledge_id", sa.Integer, nullable=True))
            batch_op.create_index("ix_user_documents_knowledge_id", ["knowledge_id"])
            batch_op.create_foreign_key(
                "fk_user_documents_knowledge_id",
                "knowledge",
                ["knowledge_id"],
                ["id"],
            )
    conn.execute(
        sa.text(
            "UPDATE user_documents SET knowledge_id = :kid WHERE knowledge_id IS NULL"
        ),
        {"kid": default_id},
    )

    if "chat_attachments" not in tables:
        op.create_table(
            "chat_attachments",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("session_id", sa.String(36), nullable=False, index=True),
            sa.Column("filename", sa.String(255), nullable=False),
            sa.Column("status", sa.String(20), nullable=False),
            sa.Column("error", sa.Text),
            sa.Column("created_at", sa.DateTime),
        )


def downgrade() -> None:
    op.drop_table("chat_attachments")
    with op.batch_alter_table("user_documents") as batch_op:
        batch_op.drop_constraint("fk_user_documents_knowledge_id", type_="foreignkey")
        batch_op.drop_index("ix_user_documents_knowledge_id")
        batch_op.drop_column("knowledge_id")
    op.drop_table("knowledge")
