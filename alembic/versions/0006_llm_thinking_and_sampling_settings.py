"""LLM thinking and sampling settings seed

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-25

Fuegt erweiterte LLM-Settings hinzu:
- CHAT_THINKING: Reasoning-Effort fuer Thinking-Modelle (none/low/medium/high/max)
- CHAT_TOP_K: Top-K Sampling
- CHAT_MAX_TOKENS: Maximale Token-Anzahl pro Antwort
- CHAT_NUM_CTX: Kontextfenster-Groesse (Ollama-spezifisch)
"""

import sqlalchemy as sa

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    settings = [
        (
            "CHAT_THINKING",
            "none",
            "llm",
            False,
            "Thinking/Reasoning-Effort (none/low/medium/high/max) - nur fuer Reasoning-Modelle",
        ),
        (
            "CHAT_TOP_K",
            "40",
            "llm",
            False,
            "Top-K Sampling (Anzahl der hoechstwahrscheinlichen Token)",
        ),
        (
            "CHAT_MAX_TOKENS",
            "",
            "llm",
            False,
            "Maximale Token-Anzahl pro Antwort (leer = unbegrenzt)",
        ),
        (
            "CHAT_NUM_CTX",
            "4096",
            "llm",
            False,
            "Kontextfenster-Groesse (Ollama num_ctx)",
        ),
    ]

    for key, value, category, is_sensitive, description in settings:
        conn.execute(
            sa.text(
                "INSERT INTO settings "
                "(key, value, default_value, category, is_sensitive, description) "
                "VALUES (:key, :value, :value, :category, :is_sensitive, :description) "
                "ON CONFLICT (key) DO NOTHING"
            ),
            {
                "key": key,
                "value": value,
                "category": category,
                "is_sensitive": is_sensitive,
                "description": description,
            },
        )


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text(
            "DELETE FROM settings WHERE key IN "
            "('CHAT_THINKING', 'CHAT_TOP_K', 'CHAT_MAX_TOKENS', 'CHAT_NUM_CTX')"
        )
    )
