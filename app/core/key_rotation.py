# Open Knowledge UI - A privacy-focused, self-hosted RAG system
# SPDX-FileCopyrightText: 2026 S2A Open Technologies gGmbH
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 S2A Open Technologies gGmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Rotation des Master-Schluessels (``ENCRYPTION_KEY``), nach dem Muster von openDox.

Ablauf fuer den Betrieb:
1. Neuen Schluessel erzeugen: ``python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"``
2. ``ENCRYPTION_KEY`` = neuer Schluessel, ``ENCRYPTION_KEY_PREVIOUS`` = alter Schluessel setzen.
3. Neu starten. Beim Start erkennt ``run_key_rotation_if_needed`` am Fingerprint
   (``encryption_key_state``) den Wechsel und verschluesselt alle Werte unter dem Master-Schluessel
   neu: Zugangsdaten, sensible Einstellungen, verschluesselte Spalten und die Datenschluessel
   (``data_keys``). Inhalte (Dokumente, Chats, Chunks) haengen an den Datenschluesseln und bleiben
   unberuehrt, deshalb dauert die Rotation auch bei grossen Bestaenden nur Sekunden.
4. Im Log auf "Schluesselrotation abgeschlossen" warten, dann ``ENCRYPTION_KEY_PREVIOUS`` entfernen.

Sicherheiten: idempotent (bereits neu verschluesselte Werte werden uebersprungen), ein Abbruch laeuft
beim naechsten Start weiter. Oeffnet ``ENCRYPTION_KEY_PREVIOUS`` keinen einzigen Wert, bleibt der
Fingerprint stehen (wahrscheinlich falscher alter Schluessel) und nichts wird geaendert. Auf
Postgres serialisiert eine Advisory-Lock mehrere gleichzeitig startende Prozesse. Der Schluessel
selbst wird nie gespeichert oder geloggt, nur sein SHA-256.
"""

import hashlib
from logging import getLogger

from sqlalchemy import text

from app.core import crypto

logger = getLogger(__name__)

_LOCK_ID = 726_004_117  # beliebige, feste Kennung fuer pg_advisory_xact_lock


def fingerprint(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()


def encrypted_columns() -> list[tuple[str, str, str]]:
    """Alle Spalten vom Typ ``EncryptedText`` (Tabelle, Spalte, Primaerschluessel) aus den Modellen.

    Aus den Metadaten statt einer Liste: eine neue verschluesselte Spalte kann nicht vergessen werden.
    """
    from app.core.models import Base

    result = []
    for table in Base.metadata.sorted_tables:
        pk = [c.name for c in table.primary_key.columns]
        for column in table.columns:
            if isinstance(column.type, crypto.EncryptedText):
                result.append((table.name, column.name, pk[0]))
    return result


def _rotate_rows(conn, select_sql: str, update_sql: str, params=None) -> tuple[int, int, int]:
    """(gesehen, neu verschluesselt, nicht zu oeffnen) fuer die Zeilen einer Abfrage."""
    seen = updated = failed = 0
    for row_id, value in conn.execute(text(select_sql), params or {}).all():
        if not crypto.is_encrypted(value):
            continue
        seen += 1
        try:
            new = crypto.rotate(value)
        except crypto.SecretKeyError:
            failed += 1
            continue
        if new is not None:
            conn.execute(text(update_sql), {"v": new, "id": row_id})
            updated += 1
    return seen, updated, failed


def _reencrypt_all(conn, sensitive_keys) -> tuple[int, int, int]:
    total = [0, 0, 0]
    for table, column, pk in encrypted_columns():
        counts = _rotate_rows(
            conn,
            f'SELECT "{pk}", "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL',
            f'UPDATE "{table}" SET "{column}" = :v WHERE "{pk}" = :id',
        )
        total = [a + b for a, b in zip(total, counts)]
    keys = sorted(sensitive_keys)
    for column in ("value", "default_value"):
        for key in keys:
            counts = _rotate_rows(
                conn,
                f"SELECT key, {column} FROM settings WHERE key = :k AND {column} IS NOT NULL",
                f"UPDATE settings SET {column} = :v WHERE key = :id",
                {"k": key},
            )
            total = [a + b for a, b in zip(total, counts)]
    return tuple(total)


def run_key_rotation_if_needed(engine, sensitive_keys) -> str:
    """Erkennt einen neuen Master-Schluessel und verschluesselt alles darunter neu.

    Returns:
        ``baseline`` (erster Start), ``unchanged``, ``rotated``, ``waiting`` (neuer Schluessel ohne
        ``ENCRYPTION_KEY_PREVIOUS``) oder ``failed`` (alter Schluessel passt nicht).
    """
    fp = fingerprint(crypto.primary_key())
    with engine.begin() as conn:
        if conn.dialect.name == "postgresql":
            conn.execute(text("SET LOCAL lock_timeout = '30s'"))
            conn.execute(text("SELECT pg_advisory_xact_lock(:id)"), {"id": _LOCK_ID})
        row = conn.execute(
            text("SELECT id, key_fingerprint FROM encryption_key_state ORDER BY id LIMIT 1")
        ).first()
        if row is None:
            conn.execute(
                text("INSERT INTO encryption_key_state (key_fingerprint) VALUES (:fp)"),
                {"fp": fp},
            )
            return "baseline"
        if row[1] == fp:
            return "unchanged"
        state_id = row[0]

        if not crypto._previous_keys():
            logger.warning(
                f"{crypto.ENV_KEY} hat sich geaendert, aber {crypto.ENV_PREVIOUS} ist nicht gesetzt: "
                "verschluesselte Daten bleiben unlesbar, bis der alte Schluessel fuer die einmalige "
                "Umschluesselung angegeben wird."
            )
            return "waiting"

        seen, updated, failed = _reencrypt_all(conn, sensitive_keys)
        if failed and not updated:
            logger.warning(
                f"Schluesselrotation: keiner von {seen} Werten liess sich oeffnen, "
                f"{crypto.ENV_PREVIOUS} ist vermutlich falsch. Fingerprint bleibt, neuer Versuch beim "
                "naechsten Start."
            )
            raise crypto.SecretKeyError(
                f"{crypto.ENV_PREVIOUS} passt nicht zu den verschluesselten Daten"
            )
        if failed:
            logger.error(
                f"Schluesselrotation: {failed} von {seen} Werten mit keinem Schluessel lesbar"
            )
            raise crypto.SecretKeyError(
                f"{failed} verschluesselte Werte lassen sich mit keinem Schluessel oeffnen"
            )
        conn.execute(
            text("UPDATE encryption_key_state SET key_fingerprint = :fp WHERE id = :id"),
            {"fp": fp, "id": state_id},
        )
    crypto.reset()
    from app.core import keys

    keys.reset()
    logger.info(f"Schluesselrotation abgeschlossen ({updated} Eintraege neu verschluesselt)")
    return "rotated"


def status() -> dict:
    """Fuer die Administration: Schluesselquelle, Fingerprint-Kuerzel, Rotation offen?"""
    from app.core.database_service import get_db_service
    from app.core.models import EncryptionKeyState

    fp = fingerprint(crypto.primary_key())
    session = get_db_service().get_session()
    try:
        row = session.query(EncryptionKeyState).order_by(EncryptionKeyState.id).first()
    finally:
        session.close()
    return {
        "source": crypto.key_source(),
        "fingerprint": fp[:12],
        "rotation_pending": bool(row and row.key_fingerprint != fp),
        "previous_set": bool(crypto._previous_keys()),
    }
