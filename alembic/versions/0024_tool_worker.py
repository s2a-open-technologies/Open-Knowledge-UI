"""Tool-Worker: Adresse und Limits fuer Pandas-Auswertungen

Revision ID: 0024
Revises: 0023
Create Date: 2026-10-13

TOOL_WORKER_URL, PANDAS_TIMEOUT_S, PANDAS_MAX_MEMORY_MB, PANDAS_MAX_OUTPUT_CHARS. Das Token steht
nicht in der Datenbank (Umgebung TOOL_WORKER_TOKEN bzw. TOOL_WORKER_TOKEN_FILE).

Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None

SETTINGS = [
    (
        "TOOL_WORKER_URL",
        "http://tool-worker:8090",
        "Adresse des Tool-Workers (fuehrt Pandas-Code isoliert aus)",
    ),
    ("PANDAS_TIMEOUT_S", "30", "Zeitlimit einer Pandas-Auswertung in Sekunden"),
    ("PANDAS_MAX_MEMORY_MB", "1024", "Speicherlimit einer Pandas-Auswertung in MB"),
    (
        "PANDAS_MAX_OUTPUT_CHARS",
        "20000",
        "Maximale Laenge der Ausgabe einer Pandas-Auswertung in Zeichen",
    ),
]


def upgrade() -> None:
    bind = op.get_bind()
    for key, value, description in SETTINGS:
        bind.execute(
            sa.text(
                "INSERT INTO settings (key, value, default_value, category, is_sensitive, description) "
                "VALUES (:key, :value, :value, 'tools', :sensitive, :description) "
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
    for key, _value, _description in SETTINGS:
        bind.execute(sa.text("DELETE FROM settings WHERE key = :key"), {"key": key})
