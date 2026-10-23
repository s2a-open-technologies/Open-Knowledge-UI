"""Standard-Anweisungen in der Datenbank; "Basisprompt ersetzen" entfaellt

Revision ID: 0030
Revises: 0029
"""

import sqlalchemy as sa

from alembic import op

revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None

# Der Wert muss zu app/agent/behavior.py:DEFAULT_BEHAVIOR passen (Test)
KEY = "AGENT_BEHAVIOR_PROMPT"
DESCRIPTION = "Standard-Anweisungen fuer alle Modelle (Rolle, Sprache, Stil, Verhalten bei Uebersichts- und Zusammenfassungsfragen)"
DEFAULT_BEHAVIOR = 'Du bist ein sorgfältiger Dokumentenanalyst. Du wertest Daten aus PDF-, Word-, Excel- und CSV-Dokumenten sachlich und genau aus und antwortest immer auf Deutsch.\n\n- **Zusammenfassungen**: Soll das Gespräch zusammengefasst werden ("unser Gespräch", "dieser Chat"), nutze NUR den Gesprächsverlauf, keine Dokumentsuche und keine neuen Quellen. Soll ein Dokument oder das Wissen zusammengefasst werden, starte direkt mit der Suche, ohne Vorbemerkungen. Ist unklar, was gemeint ist, frage in einem Satz nach.\n- **Gezielte Fragen** ("Ist X da?", "Gibt es etwas zu Y?"): Antworte knapp mit Ja oder Nein und nenne nur die passenden Fundstellen. Zähle nie den übrigen Inhalt auf.\n- **Übersichtsfragen**: Nur wenn der Nutzer ausdrücklich eine Übersicht oder Liste verlangt ("was gibt es alles", "Aufstellung", "Überblick"), nenne alle verfügbaren Quellen.\n- **Ablegen**: Bittet der Nutzer darum, etwas abzulegen oder zu speichern, lege genau den gewünschten Inhalt ab (Gespräch ODER Dokumentanalyse, nicht vermischen).\n- **Stil**: Formell, knapp und ohne Floskeln. Beschreibe nicht, was du tun wirst, sondern tue es.'


def upgrade() -> None:
    bind = op.get_bind()
    cols = {c["name"] for c in sa.inspect(bind).get_columns("model_configs")}
    if "replace_prompt" in cols:
        # Der Text eines Modells mit "Basisprompt ersetzen" bleibt als Zusatz erhalten; der Basis-Prompt gilt kuenftig immer.
        with op.batch_alter_table("model_configs") as batch_op:
            batch_op.drop_column("replace_prompt")
    bind.execute(
        sa.text(
            "INSERT INTO settings (key, value, default_value, category, is_sensitive, description) "
            "VALUES (:key, :value, :value, 'llm', :sensitive, :description) "
            "ON CONFLICT (key) DO NOTHING"
        ),
        {
            "key": KEY,
            "value": DEFAULT_BEHAVIOR,
            "sensitive": False,
            "description": DESCRIPTION,
        },
    )


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(sa.text("DELETE FROM settings WHERE key = :key"), {"key": KEY})
    cols = {c["name"] for c in sa.inspect(bind).get_columns("model_configs")}
    if "replace_prompt" not in cols:
        with op.batch_alter_table("model_configs") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "replace_prompt",
                    sa.Boolean,
                    nullable=False,
                    server_default=sa.false(),
                )
            )
