"""Embedding model name: add sentence-transformers/ prefix for fastembed

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-10

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    # fastembed (ONNX) erwartet den HF-Org-Prefix im Modellnamen,
    # z. B. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    # statt des bisherigen nackten paraphrase-multilingual-MiniLM-L12-v2.
    conn.execute(
        sa.text(
            "UPDATE settings SET value = :new "
            "WHERE key = 'EMBEDDING_MODEL' AND value = :old"
        ),
        {
            "new": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "old": "paraphrase-multilingual-MiniLM-L12-v2",
        },
    )


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text(
            "UPDATE settings SET value = :old "
            "WHERE key = 'EMBEDDING_MODEL' AND value = :new"
        ),
        {
            "old": "paraphrase-multilingual-MiniLM-L12-v2",
            "new": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
    )
