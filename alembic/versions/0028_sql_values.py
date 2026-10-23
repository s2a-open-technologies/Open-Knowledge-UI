"""SQL-Wertverzeichnis: Tabellen und Einstellungen

Revision ID: 0028
Revises: 0027
Create Date: 2026-10-13

sql_values (verschiedene Werte ausgewaehlter Textspalten einer SQL-Quelle mit Haeufigkeit, keine Zeilen),
sql_value_index (Stand je Quelle) und die Einstellungen SQL_VALUES_MAX_DISTINCT, SQL_VALUES_MAX_RESULTS,
SQL_VALUES_HINT.

Idempotent (siehe Migration 0008).
"""

import sqlalchemy as sa

from alembic import op

revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None

# Der Wert von SQL_VALUES_HINT muss zu app/agent/toolworker/config.py:SQL_VALUES_HINT_DEFAULT passen (Test)
SETTINGS = [
    (
        "SQL_VALUES_MAX_DISTINCT",
        "5000",
        "Hoechstens so viele verschiedene Werte je Spalte im Wertverzeichnis (darueber ist es kein Kategoriefeld)",
    ),
    (
        "SQL_VALUES_MAX_RESULTS",
        "20",
        "Hoechstens so viele Werte liefert die Wertsuche an das Modell",
    ),
    (
        "SQL_VALUES_HINT",
        "Für Textwerte (Namen, Kategorien, Orte) rufe vor dem SELECT sql_find_values mit dem Begriff aus der Frage auf: es liefert die exakten Schreibweisen aus der Datenbank (Ein-/Mehrzahl, Umlaute, zusammengesetzte Wörter). Setze diese Werte mit IN (...) ein, statt zu raten.",
        "Hinweis an das Modell, wenn ein Wertverzeichnis vorhanden ist",
    ),
]


def upgrade() -> None:
    insp = sa.inspect(op.get_bind())
    tables = insp.get_table_names()
    if "sql_values" not in tables:
        op.create_table(
            "sql_values",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column(
                "knowledge_id",
                sa.Integer,
                sa.ForeignKey("knowledge.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("table_name", sa.String(255), nullable=False),
            sa.Column("column_name", sa.String(255), nullable=False),
            sa.Column("value", sa.Text, nullable=False),
            sa.Column("count", sa.Integer, nullable=False, server_default="1"),
        )
        op.create_index("ix_sql_values_knowledge_id", "sql_values", ["knowledge_id"])
    if "sql_value_index" not in tables:
        op.create_table(
            "sql_value_index",
            sa.Column(
                "knowledge_id",
                sa.Integer,
                sa.ForeignKey("knowledge.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("status", sa.String(20), nullable=False, server_default="idle"),
            sa.Column("built_at", sa.DateTime),
            sa.Column("value_count", sa.Integer, nullable=False, server_default="0"),
            sa.Column("column_count", sa.Integer, nullable=False, server_default="0"),
            sa.Column("error", sa.Text),
        )
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
    op.drop_table("sql_value_index")
    op.drop_index("ix_sql_values_knowledge_id", table_name="sql_values")
    op.drop_table("sql_values")
