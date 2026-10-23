"""UI theme setting seed (light/dark/system)

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-10

Fuegt die Settings-Zeile UI_THEME hinzu, die das App-Farbschema steuert
(light | dark | system). Default 'system' (folgt prefers-color-scheme),
damit bestehende Nutzer beim Upgrade kein plötzliches Dunkel bekommen.
"""

import sqlalchemy as sa

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text(
            "INSERT INTO settings "
            "(key, value, default_value, category, is_sensitive, description) "
            "VALUES (:key, :value, :value, :category, :is_sensitive, :description) "
            "ON CONFLICT (key) DO NOTHING"
        ),
        {
            "key": "UI_THEME",
            "value": "system",
            "category": "ui",
            "is_sensitive": False,
            "description": "Oberflaeche: light, dark oder system (folgt OS)",
        },
    )


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("DELETE FROM settings WHERE key = 'UI_THEME'"))
