"""Automatische Compaction des Chat-Kontexts

Revision ID: 0021
Revises: 0020
Create Date: 2026-10-12

sessions.summary / sessions.summary_through_id: Zusammenfassung der aelteren Nachrichten eines
Chats (Compaction). Dazu die Einstellungen COMPACTION_* (Kategorie llm) inkl. Prompt.

Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None

PROMPT = "Du fasst den bisherigen Verlauf eines Gesprächs zusammen, damit es mit weniger Platz fortgesetzt werden kann. Die Zusammenfassung ersetzt die unten stehenden Nachrichten.\n\nBehalte unbedingt:\n- Fragen, Ziele und Wünsche des Nutzers\n- Wichtige Fakten, Zahlen, Namen, Fristen, Dateinamen und Quellenangaben (mit Seite oder Abschnitt)\n- Getroffene Entscheidungen und Ergebnisse\n- Offene Punkte und Rückfragen\n\nLass Floskeln, Wiederholungen und Zwischenschritte weg. Schreibe knapp, sachlich und auf Deutsch. Gib ausschließlich die Zusammenfassung aus, ohne Einleitung.\n\n{messages}"

SETTINGS = [
    (
        "COMPACTION_ENABLED",
        "true",
        "Chat-Verlauf automatisch zusammenfassen, bevor der Kontext ueberlaeuft",
    ),
    (
        "COMPACTION_HEADROOM_PERCENT",
        "5",
        "Luft in Prozent: Zusammenfassen ab (100 - Wert) % des Eingabe-Budgets",
    ),
    (
        "COMPACTION_KEEP_PERCENT",
        "30",
        "Anteil des Eingabe-Budgets in Prozent, der als juengster Verlauf wortwoertlich bleibt",
    ),
    (
        "COMPACTION_OUTPUT_RESERVE_PERCENT",
        "20",
        "Reserve fuer die Antwort in Prozent des Kontexts, wenn das Modell keine max. Tokens setzt",
    ),
    (
        "COMPACTION_PROMPT",
        PROMPT,
        "Anweisung fuer die Zusammenfassung des Verlaufs (Platzhalter {messages})",
    ),
]


def _cols(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    cols = _cols("sessions")
    with op.batch_alter_table("sessions") as batch_op:
        if "summary" not in cols:
            batch_op.add_column(sa.Column("summary", sa.Text, nullable=True))
        if "summary_through_id" not in cols:
            batch_op.add_column(
                sa.Column("summary_through_id", sa.Integer, nullable=True)
            )
    bind = op.get_bind()
    for key, value, description in SETTINGS:
        bind.execute(
            sa.text(
                "INSERT INTO settings (key, value, default_value, category, is_sensitive, description) "
                "VALUES (:key, :value, :value, 'llm', :sensitive, :description) "
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
    with op.batch_alter_table("sessions") as batch_op:
        batch_op.drop_column("summary_through_id")
        batch_op.drop_column("summary")
