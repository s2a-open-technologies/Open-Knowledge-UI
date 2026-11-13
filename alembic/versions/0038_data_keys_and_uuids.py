"""Envelope-Encryption: UUID je Nutzer und Knowledge, Datenschluessel, Fingerprint des Master-Schluessels

Revision ID: 0038
Revises: 0037

- users.uuid, knowledge.uuid: feste, nie wiederverwendete Kennung (Ordnernamen, Schluessel-
  Zuordnung, AAD der Verschluesselung). Bestehende Zeilen bekommen eine zufaellige UUID.
- data_keys: Datenschluessel je Nutzer/Knowledge, mit dem Master-Schluessel verschluesselt
  (app/core/keys.py). Die Schluessel entstehen bei Bedarf in der App, nicht hier.
- encryption_key_state: Fingerprint des Master-Schluessels fuer die Rotation
  (app/core/key_rotation.py).

Idempotent und Postgres-sicher.
"""

import uuid

import sqlalchemy as sa

from alembic import op

revision = "0038"
down_revision = "0037"
branch_labels = None
depends_on = None


def _add_uuid(conn, insp, table: str) -> None:
    columns = {c["name"] for c in insp.get_columns(table)}
    if "uuid" not in columns:
        with op.batch_alter_table(table) as batch_op:
            batch_op.add_column(sa.Column("uuid", sa.String(36), nullable=True))
    for (row_id,) in conn.execute(
        sa.text(f"SELECT id FROM {table} WHERE uuid IS NULL")
    ).all():
        conn.execute(
            sa.text(f"UPDATE {table} SET uuid = :u WHERE id = :id"),
            {"u": str(uuid.uuid4()), "id": row_id},
        )
    indexes = {i["name"] for i in sa.inspect(conn).get_indexes(table)}
    uniques = {u["name"] for u in sa.inspect(conn).get_unique_constraints(table)}
    name = f"uq_{table}_uuid"
    with op.batch_alter_table(table) as batch_op:
        batch_op.alter_column("uuid", existing_type=sa.String(36), nullable=False)
        if name not in indexes and name not in uniques:
            batch_op.create_unique_constraint(name, ["uuid"])


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    tables = set(insp.get_table_names())

    _add_uuid(conn, insp, "users")
    _add_uuid(conn, insp, "knowledge")

    if "data_keys" not in tables:
        op.create_table(
            "data_keys",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("owner_kind", sa.String(20), nullable=False),
            sa.Column("owner_uuid", sa.String(36), nullable=False),
            sa.Column("version", sa.Integer, nullable=False, server_default="1"),
            sa.Column("wrapped_key", sa.Text, nullable=False),
            sa.Column("created_at", sa.DateTime, nullable=True),
            sa.UniqueConstraint(
                "owner_kind", "owner_uuid", "version", name="uq_data_keys_owner_version"
            ),
        )
        op.create_index("ix_data_keys_owner_uuid", "data_keys", ["owner_uuid"])
    if "encryption_key_state" not in tables:
        op.create_table(
            "encryption_key_state",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("key_fingerprint", sa.String(64), nullable=False),
            sa.Column("updated_at", sa.DateTime, nullable=True),
        )


def downgrade() -> None:
    conn = op.get_bind()
    tables = set(sa.inspect(conn).get_table_names())
    if "encryption_key_state" in tables:
        op.drop_table("encryption_key_state")
    if "data_keys" in tables:
        op.drop_index("ix_data_keys_owner_uuid", table_name="data_keys")
        op.drop_table("data_keys")
    for table in ("knowledge", "users"):
        with op.batch_alter_table(table) as batch_op:
            batch_op.drop_constraint(f"uq_{table}_uuid", type_="unique")
            batch_op.drop_column("uuid")
