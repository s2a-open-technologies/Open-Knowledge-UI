"""Prompts und Modelle fuer Bildbeschreibung und Kontext-Anreicherung

Revision ID: 0018
Revises: 0017
Create Date: 2026-10-12

Die Prompts, Modelle und Parameter beider Aufgaben liegen in der Datenbank (Tabelle
``settings``) und sind unter "Einlesen & Modelle" einstellbar. Die Migration legt die
Zeilen mit den bisherigen Prompts an (Wert und Standardwert) und uebernimmt das bisher
gemeinsame Modell (``VISION_MODEL_REF``) auch fuer die Kontext-Anreicherung.

Idempotent (siehe Migration 0008); Boolean als Parameter, damit sie auch auf Postgres laeuft.
"""

import sqlalchemy as sa

from alembic import op

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None

VISION_PROMPT = '## Rolle:\nDu bist ein spezialisierter Analyst für unstructurierte Dokumente. Deine Aufgabe ist es, Tabellen und Prozessdiagramme in präzise, suchoptimierte Markdown-Beschreibungen zu übersetzen.\n\n### Anweisungen:\n\n**Strukturelle Einordnung:** Beginne mit dem Typ (z. B. "Ablaufdiagramm Genehmigungsprozess" oder "Tabelle Überstundenvergütung").\n**Logik-Extraktion (Wenn-Dann):** Analysiere das Bild gezielt nach Bedingungen und Konsequenzen. Formuliere diese als klare Sätze (z. B. "Wenn [Bedingung], dann [Folge/Aktion]").\n**Daten-Highlights:** Extrahiere bei Tabellen nur die wichtigsten Schwellenwerte, Fristen oder Beträge.\nErstelle eine kompakte Markdown-Tabelle der Kernwerte (max. 8 Zeilen).\n\n## Zuständigkeiten:\nNenne explizit alle im Bild vorkommenden Rollen, Abteilungen oder Gremien (z. B. "Personalrat", "IT-Abteilung", "Vorgesetzter").\n\n## Zusammenfassung:\nSchreibe einen abschließenden Satz über den Hauptzweck des Bildes für die Indexierung.\n\n## Formatierungsvorgaben:\n\n* Gib ausschließlich das Markdown-Ergebnis aus.\n* Keine Einleitung ("Hier ist..."), keine Metadaten über das Bild-Preprocessing.\n* Nutze klare Hierarchien mit ### Überschriften.\n* Antworte ausschließlich aus Deutsch'

CONTEXT_PROMPT = "Du bist ein Dokumenten-Indexierer. Gib NUR 1-2 kurze Sätze zurück, die den Abschnitt im Gesamtdokument verorten.\n\n### Gesamtdokument (Auszug):\n{dokument}\n\n### Abschnitt:\n{abschnitt}\n\n### Kontext-Einordnung:"

ROWS = [
    (
        "VISION_PROMPT",
        VISION_PROMPT,
        "Anweisung fuer die Beschreibung von Bildern in Dokumenten",
    ),
    (
        "CONTEXT_PROMPT",
        CONTEXT_PROMPT,
        "Anweisung fuer die Kontext-Anreicherung (Platzhalter {dokument} und {abschnitt})",
    ),
    (
        "CONTEXT_EXCERPT_CHARS",
        "2000",
        "Zeichen des Dokuments, die der Kontext-Anreicherung als Auszug dienen",
    ),
    (
        "VISION_TEMPERATURE",
        "",
        "Temperatur der Bildbeschreibung (leer = Wert des Modells)",
    ),
    (
        "VISION_MAX_TOKENS",
        "",
        "Maximale Antwortlaenge der Bildbeschreibung (leer = Wert des Modells)",
    ),
    (
        "CONTEXT_MODEL_REF",
        "",
        "Modell der Kontext-Anreicherung (leer = Standardmodell)",
    ),
    (
        "CONTEXT_TEMPERATURE",
        "",
        "Temperatur der Kontext-Anreicherung (leer = Wert des Modells)",
    ),
    (
        "CONTEXT_MAX_TOKENS",
        "",
        "Maximale Antwortlaenge der Kontext-Anreicherung (leer = Wert des Modells)",
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
    # Bisher bediente ein Modell beide Aufgaben: die Kontext-Anreicherung behaelt es.
    current = bind.execute(
        sa.text("SELECT value FROM settings WHERE key = 'VISION_MODEL_REF'")
    ).scalar()
    if current:
        bind.execute(
            sa.text(
                "UPDATE settings SET value = :v WHERE key = 'CONTEXT_MODEL_REF' AND (value IS NULL OR value = '')"
            ),
            {"v": current},
        )


def downgrade() -> None:
    bind = op.get_bind()
    for key, _value, _description in ROWS:
        bind.execute(sa.text("DELETE FROM settings WHERE key = :key"), {"key": key})
