"""Anmeldung mit LDAP und OIDC

Revision ID: 0020
Revises: 0019
Create Date: 2026-10-12

users.auth_source ('local' | 'ldap' | 'oidc'), users.external_id (OIDC-sub bzw. LDAP-DN),
users.external_groups (JSON: beim letzten Login gemeldete Gruppennamen),
group_members.external (vom Login gesetzte Mitgliedschaft; nur diese raeumt der Abgleich ab).
Dazu die Einstellungen der Anmeldung (Kategorie auth); bestehende Konten bleiben lokal.

Idempotent (siehe Migration 0008); Boolean als Parameter (Postgres).
"""

import sqlalchemy as sa

from alembic import op

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None

SETTINGS = [
    (
        "AUTH_LDAP",
        "{}",
        True,
        "LDAP/AD-Anmeldung (JSON, Verwaltung unter Administration > Authentifizierung)",
    ),
    (
        "AUTH_OIDC",
        "{}",
        True,
        "OIDC-Anmeldung (JSON, Verwaltung unter Administration > Authentifizierung)",
    ),
    ("AUTH_DEFAULT_ROLE", "user", False, "Rolle neuer externer Konten"),
    (
        "AUTH_AUTO_CREATE",
        "true",
        False,
        "Externe Konten beim ersten Login automatisch anlegen",
    ),
    (
        "APP_BASE_URL",
        "",
        False,
        "Oeffentliche Adresse der App (fuer den OIDC-Rueckruf)",
    ),
]


def _cols(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    users = _cols("users")
    with op.batch_alter_table("users") as batch_op:
        if "auth_source" not in users:
            batch_op.add_column(
                sa.Column(
                    "auth_source", sa.String(20), nullable=False, server_default="local"
                )
            )
        if "external_id" not in users:
            batch_op.add_column(sa.Column("external_id", sa.String(255), nullable=True))
            batch_op.create_index("ix_users_external_id", ["external_id"])
        if "external_groups" not in users:
            batch_op.add_column(sa.Column("external_groups", sa.Text, nullable=True))
    if "external" not in _cols("group_members"):
        with op.batch_alter_table("group_members") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "external", sa.Boolean, nullable=False, server_default=sa.false()
                )
            )
    bind = op.get_bind()
    for key, value, sensitive, description in SETTINGS:
        bind.execute(
            sa.text(
                "INSERT INTO settings (key, value, default_value, category, is_sensitive, description) "
                "VALUES (:key, :value, :value, 'auth', :sensitive, :description) "
                "ON CONFLICT (key) DO NOTHING"
            ),
            {
                "key": key,
                "value": value,
                "sensitive": sensitive,
                "description": description,
            },
        )


def downgrade() -> None:
    bind = op.get_bind()
    for key, *_ in SETTINGS:
        bind.execute(sa.text("DELETE FROM settings WHERE key = :key"), {"key": key})
    with op.batch_alter_table("group_members") as batch_op:
        batch_op.drop_column("external")
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_index("ix_users_external_id")
        batch_op.drop_column("external_groups")
        batch_op.drop_column("external_id")
        batch_op.drop_column("auth_source")
