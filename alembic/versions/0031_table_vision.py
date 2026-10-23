"""Tabellen aus PDFs per Vision-Modell pruefen (Einlese-Aufgabe "tables")

Revision ID: 0031
Revises: 0030
"""

import sqlalchemy as sa

from alembic import op

revision = "0031"
down_revision = "0030"
branch_labels = None
depends_on = None

# Der Prompt muss zu app/ingest/prompts.py:TABLES_PROMPT passen (Test)
ROWS = [
    (
        "INGEST_TABLE_VISION",
        "true",
        "Tabellen aus PDFs zusaetzlich per Vision-Modell pruefen und korrigieren (wirkt nur mit Bildbeschreibung)",
    ),
    (
        "TABLES_PROMPT",
        "Du liest eine Tabelle aus einem Dokument. Das Bild zeigt die Tabelle, darunter steht ein automatisch erzeugter Entwurf, der Fehler enthalten kann (verschobene Spalten, vertauschte oder fehlende Zellen).\n\nGib die Tabelle so aus, wie sie im Bild steht, als Markdown-Tabelle:\n- Eine Kopfzeile, dann die Trennzeile, dann alle Datenzeilen. Keine Zeile weglassen, keine erfinden.\n- Zellen wörtlich übernehmen, auch Zahlen, Einheiten, Währungszeichen und Schreibweisen (kein Umrechnen, kein Runden).\n- Verbundene Zellen in jede Zeile oder Spalte wiederholen, die sie überspannen. Mehrzeilige Zellen in einer Zelle zusammenfassen.\n- Leere Zellen bleiben leer. Hat die Tabelle mehrere Kopfzeilen, verbinde sie zu einer Kopfzeile (z. B. „Umsatz Q1“).\n- Ein senkrechter Strich in einer Zelle wird als \\| geschrieben.\n\nGib ausschließlich die Markdown-Tabelle aus, ohne Einleitung und ohne Erklärung.",
        "Anweisung fuer die Pruefung von Tabellen per Vision-Modell",
    ),
    (
        "TABLES_MODEL_REF",
        "",
        "Modell der Tabellen-Pruefung (leer = Modell der Bildbeschreibung)",
    ),
    ("TABLES_TEMPERATURE", "0", "Temperatur der Tabellen-Pruefung"),
    (
        "TABLES_MAX_TOKENS",
        "",
        "Maximale Antwortlaenge der Tabellen-Pruefung (leer = Wert des Modells)",
    ),
    (
        "TABLES_MAX_ROW_DEVIATION",
        "30",
        "Prozent, um die die Zeilenzahl der geprueften Tabelle vom Entwurf abweichen darf, bevor sie verworfen wird",
    ),
]


def upgrade() -> None:
    bind = op.get_bind()
    for key, value, description in ROWS:
        bind.execute(
            sa.text(
                "INSERT INTO settings (key, value, default_value, category, is_sensitive, description) "
                "VALUES (:key, :value, :value, 'ingest', :sensitive, :description) "
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
