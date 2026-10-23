"""Zeitlimit und Hoechstlaenge der PII-Filter-Aufrufe

Revision ID: 0035
Revises: 0034

PII_FILTER_TIMEOUT_S (Standard 10 s, waechst mit der Textlaenge) und PII_FILTER_MAX_CHARS (Standard
12000 Zeichen je Aufruf, laengere Texte gehen in Stuecken). Gemessen an einem EntityGuard: 8000-12000
Zeichen warm 1,6-2,8 s, kalt oder unter Last 17-34 s, 50000 Zeichen brachten den Dienst zum Absturz.
Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0035"
down_revision = "0034"
branch_labels = None
depends_on = None

ROWS = [
    (
        "PII_FILTER_TIMEOUT_S",
        "10",
        "Zeitlimit eines PII-Filter-Aufrufs in Sekunden (waechst mit der Textlaenge)",
    ),
    (
        "PII_FILTER_MAX_CHARS",
        "12000",
        "Hoechste Textlaenge je PII-Filter-Aufruf; laengere Texte gehen in Stuecken",
    ),
]


def upgrade() -> None:
    bind = op.get_bind()
    for key, value, description in ROWS:
        bind.execute(
            sa.text(
                "INSERT INTO settings (key, value, default_value, category, is_sensitive, description) "
                "VALUES (:key, :value, :value, 'pii_filter', :sensitive, :description) "
                "ON CONFLICT (key) DO NOTHING"
            ),
            {
                "key": key,
                "value": value,
                "sensitive": False,
                "description": description,
            },
        )


def downgrade() -> None:
    bind = op.get_bind()
    for key, _value, _description in ROWS:
        bind.execute(sa.text("DELETE FROM settings WHERE key = :key"), {"key": key})
