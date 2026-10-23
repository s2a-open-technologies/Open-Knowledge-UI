"""SQL-Knowledge: Worker-Adresse, Limits und Hinweis fuer SQL-Quellen

Revision ID: 0025
Revises: 0024
Create Date: 2026-10-13

SQL_WORKER_URL, SQL_ROW_LIMIT, SQL_TIMEOUT_S, SQL_MAX_CELL_CHARS, SQL_PROMPT.

Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None

SQL_PROMPT = (
    "[Hinweis: Für diese Frage sind SQL-Datenbanken verbunden ({sources}). Rufe zuerst sql_schema auf, "
    "wenn du Tabellen und Spalten nicht kennst, und schreibe dann ein einzelnes SELECT mit sql_query. "
    "Nur lesende Abfragen sind erlaubt, Ergebnisse sind auf wenige Zeilen begrenzt: aggregiere mit "
    "COUNT, SUM, GROUP BY, statt viele Zeilen zu laden. Qualifiziere Tabellen bei Bedarf mit dem Schema, "
    "nenne Spalten einzeln und zeige menschenlesbare Namen (JOIN statt nackter IDs). Gib die Zahlen "
    "und Namen aus den Ergebnissen wieder und erfinde nichts.]"
)

# Der Wert von SQL_PROMPT muss zu app/agent/toolworker/config.py:SQL_PROMPT_DEFAULT passen (Test)
SETTINGS = [
    (
        "SQL_WORKER_URL",
        "http://tool-worker-sql:8090",
        "Adresse des SQL-Workers (fragt Datenbanken lesend ab)",
    ),
    ("SQL_ROW_LIMIT", "50", "Maximale Zeilen je SQL-Abfrage, die das Modell sieht"),
    ("SQL_TIMEOUT_S", "15", "Zeitlimit einer SQL-Abfrage in Sekunden"),
    (
        "SQL_MAX_CELL_CHARS",
        "500",
        "Maximale Laenge eines Zellwerts in SQL-Ergebnissen in Zeichen",
    ),
    (
        "SQL_PROMPT",
        SQL_PROMPT,
        "Hinweis an das Modell bei SQL-Quellen (Platzhalter {sources})",
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
