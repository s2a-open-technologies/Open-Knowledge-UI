"""Websuche ohne PII-Filter ausdruecklich erlauben

Revision ID: 0023
Revises: 0022
Create Date: 2026-10-12

SEARCH_ALLOW_UNMASKED (Standard false): erlaubt die Websuche auch ohne aktiven PII-Filter. Dann
gehen Suchanfragen unmaskiert an SearXNG. Bestehende Installationen aendern ihr Verhalten nicht.

Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None

KEY = "SEARCH_ALLOW_UNMASKED"
DESCRIPTION = "Websuche auch ohne aktiven PII-Filter erlauben (Suchanfragen gehen dann unmaskiert raus)"


def upgrade() -> None:
    op.get_bind().execute(
        sa.text(
            "INSERT INTO settings (key, value, default_value, category, is_sensitive, description) "
            "VALUES (:key, 'false', 'false', 'search', :sensitive, :description) "
            "ON CONFLICT (key) DO NOTHING"
        ),
        {"key": KEY, "sensitive": False, "description": DESCRIPTION},
    )


def downgrade() -> None:
    op.get_bind().execute(
        sa.text("DELETE FROM settings WHERE key = :key"), {"key": KEY}
    )
