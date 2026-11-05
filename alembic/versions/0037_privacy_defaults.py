"""Datenschutz-Standards: Original nicht aufbewahren, Aufbewahrungsfrist fuer Chats

Revision ID: 0037
Revises: 0036

Datensparsamkeit: nach dem Einlesen genuegt das Markdown (Suche, Neu-Indexieren).
Der neue Standard fuer ``INGEST_KEEP_ORIGINAL`` ist ``false``. Bestehende Installationen
werden nur umgestellt, wenn der Wert noch dem alten Standard entspricht (nie im
Admin-Bereich geaendert); ein bewusst gesetzter Wert bleibt. Bereits archivierte
Originale bleiben liegen, bis das Dokument geloescht wird.

Neu ist ``CHAT_RETENTION_DAYS`` (0 = Chats unbegrenzt aufbewahren, siehe app/chat/retention.py).
"""

import sqlalchemy as sa

from alembic import op

revision = "0037"
down_revision = "0036"
branch_labels = None
depends_on = None

KEY = "INGEST_KEEP_ORIGINAL"
DESCRIPTION = "Originaldatei nach dem Einlesen archivieren (aus, Standard: nur das Markdown bleibt)"


RETENTION_DESCRIPTION = "Chats nach so vielen Tagen ohne Nutzung loeschen, samt Anhaengen (0 = nie)"


def upgrade() -> None:
    bind = op.get_bind()
    bind.execute(
        sa.text(
            "INSERT INTO settings (key, value, default_value, category, is_sensitive, description) "
            "VALUES ('CHAT_RETENTION_DAYS', '0', '0', 'privacy', :sensitive, :description) "
            "ON CONFLICT (key) DO NOTHING"
        ),
        {"sensitive": False, "description": RETENTION_DESCRIPTION},
    )
    bind.execute(
        sa.text(
            "UPDATE settings SET value = 'false' "
            "WHERE key = :key AND value = default_value AND lower(coalesce(value, '')) = 'true'"
        ),
        {"key": KEY},
    )
    bind.execute(
        sa.text("UPDATE settings SET default_value = 'false', description = :d WHERE key = :key"),
        {"key": KEY, "d": DESCRIPTION},
    )


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(sa.text("DELETE FROM settings WHERE key = 'CHAT_RETENTION_DAYS'"))
    bind.execute(
        sa.text(
            "UPDATE settings SET value = 'true' "
            "WHERE key = :key AND value = default_value AND value = 'false'"
        ),
        {"key": KEY},
    )
    bind.execute(
        sa.text("UPDATE settings SET default_value = 'true' WHERE key = :key"),
        {"key": KEY},
    )
