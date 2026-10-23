"""Eigentuemer von Uploads

Revision ID: 0016
Revises: 0015
Create Date: 2026-10-12

user_documents.uploaded_by, chat_attachments.user_id / markdown_path: Grundlage fuer
"Meine Uploads" (eigene Dateien ansehen und loeschen). Altbestand hat keinen Eigentuemer
(NULL) und bleibt im Knowledge verwaltbar.

Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def _cols(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    docs = _cols("user_documents")
    if "uploaded_by" not in docs:
        with op.batch_alter_table("user_documents") as batch_op:
            batch_op.add_column(sa.Column("uploaded_by", sa.Integer, nullable=True))
            batch_op.create_foreign_key(
                "fk_user_documents_uploaded_by", "users", ["uploaded_by"], ["id"]
            )
            batch_op.create_index("ix_user_documents_uploaded_by", ["uploaded_by"])
    att = _cols("chat_attachments")
    with op.batch_alter_table("chat_attachments") as batch_op:
        if "user_id" not in att:
            batch_op.add_column(sa.Column("user_id", sa.Integer, nullable=True))
            batch_op.create_foreign_key(
                "fk_chat_attachments_user_id", "users", ["user_id"], ["id"]
            )
            batch_op.create_index("ix_chat_attachments_user_id", ["user_id"])
        if "markdown_path" not in att:
            batch_op.add_column(
                sa.Column("markdown_path", sa.String(500), nullable=True)
            )
    # Anhaenge bestehender Chats gehoeren dem Besitzer des Chats
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "UPDATE chat_attachments SET user_id = "
            "(SELECT s.user_id FROM sessions s WHERE s.id = chat_attachments.session_id) "
            "WHERE user_id IS NULL"
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("chat_attachments") as batch_op:
        batch_op.drop_column("markdown_path")
        batch_op.drop_column("user_id")
    with op.batch_alter_table("user_documents") as batch_op:
        batch_op.drop_column("uploaded_by")
