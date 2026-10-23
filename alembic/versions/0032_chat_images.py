"""Bilder im Chat per Vision-Modell lesen (Einlese-Aufgabe "images")

Revision ID: 0032
Revises: 0031
"""

import sqlalchemy as sa

from alembic import op

revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None

# Der Prompt muss zu app/ingest/prompts.py:IMAGES_PROMPT passen (Test)
ROWS = [
    (
        "INGEST_CHAT_IMAGES",
        "true",
        "Bilder im Chat annehmen: das Bildmodell liest sie, der Text wird wie ein Anhang durchsuchbar",
    ),
    (
        "IMAGES_PROMPT",
        "Du erschließt ein Bild, das ein Nutzer in einen Chat hochgeladen hat, damit sein Inhalt per Textsuche gefunden und beantwortet werden kann.\n\nGib in dieser Reihenfolge aus:\n1. Eine Zeile, was das Bild ist (z. B. Foto, Screenshot, Scan eines Briefs, Diagramm, Tabelle).\n2. Alle lesbaren Texte wörtlich und vollständig, in Leserichtung (Zahlen, Einheiten und Schreibweisen unverändert). Unleserliches als [unleserlich] markieren, nichts ergänzen.\n3. Tabellen als Markdown-Tabelle, Diagramme und Abläufe als kurze Sätze (Wenn-Dann, Beteiligte, Werte).\n4. Bei Fotos ohne Text: eine sachliche Beschreibung dessen, was zu sehen ist.\n\nErfinde nichts, was nicht im Bild steht. Antworte ausschließlich auf Deutsch und gib nur das Ergebnis aus, ohne Einleitung.",
        "Anweisung fuer das Lesen von Bildern im Chat",
    ),
    (
        "IMAGES_MODEL_REF",
        "",
        "Modell fuer Bilder im Chat (leer = Modell der Bildbeschreibung)",
    ),
    ("IMAGES_TEMPERATURE", "0", "Temperatur fuer Bilder im Chat"),
    (
        "IMAGES_MAX_TOKENS",
        "",
        "Maximale Antwortlaenge fuer Bilder im Chat (leer = Wert des Modells)",
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
