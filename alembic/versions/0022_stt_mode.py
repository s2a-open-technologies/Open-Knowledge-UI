"""Betriebsart der Spracheingabe

Revision ID: 0022
Revises: 0021
Create Date: 2026-10-12

STT_MODE (auto | stream | upload): Live-Streaming (Parakeet) oder Aufnehmen und Senden
(OpenAI-Standard /audio/transcriptions, z. B. Whisper). Standard auto waehlt selbst.

Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None

KEY = "STT_MODE"
DESCRIPTION = (
    "Betriebsart der Spracheingabe: auto, stream (live, Parakeet) oder upload "
    "(aufnehmen und senden, OpenAI-Standard)"
)


def upgrade() -> None:
    op.get_bind().execute(
        sa.text(
            "INSERT INTO settings (key, value, default_value, category, is_sensitive, description) "
            "VALUES (:key, 'auto', 'auto', 'stt', :sensitive, :description) "
            "ON CONFLICT (key) DO NOTHING"
        ),
        {"key": KEY, "sensitive": False, "description": DESCRIPTION},
    )


def downgrade() -> None:
    op.get_bind().execute(
        sa.text("DELETE FROM settings WHERE key = :key"), {"key": KEY}
    )
