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

"""Inhalte verschluesseln: ein Datenschluessel je Nutzer bzw. Knowledge-Bereich (Envelope-Encryption).

- Jeder Nutzer und jeder Knowledge-Bereich hat einen zufaelligen 256-Bit-Schluessel in ``data_keys``,
  verpackt mit dem Master-Schluessel (``app/core/crypto.py``, ``ENCRYPTION_KEY``). Eine Rotation des
  Master-Schluessels verpackt nur diese Zeilen neu (``app/core/key_rotation.py``), die Inhalte bleiben.
- Verfahren: AES-256-GCM. Die UUID des Eigentuemers ist Associated Data: ein Chiffrat laesst sich nur
  unter dem Eigentuemer oeffnen, fuer den es entstand (kein Umkopieren zwischen Nutzern).
- Crypto-Shredding: ``shred`` loescht den Schluessel; alle Inhalte, auch in Backups, sind unlesbar.
- Altbestand: Werte ohne Kennung (``okc:v1:`` bzw. ``OKC1``) gelten als Klartext und kommen
  unveraendert zurueck, damit der Uebergang ohne Stillstand geht.

Schutzgrenze: Wer den laufenden Server und den Master-Schluessel hat, kann alles lesen. Geschuetzt
sind Backups, Volume- und Qdrant-Snapshots und ein Datenbank-Dump ohne den Master-Schluessel.
"""

import base64
import os
import struct
import threading
from logging import getLogger

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core import crypto
from app.core.crypto import SecretKeyError

logger = getLogger(__name__)

USER = "user"
KNOWLEDGE = "knowledge"
KINDS = (USER, KNOWLEDGE)

TEXT_PREFIX = "okc:v1:"  # Texte (DB-Spalten, Qdrant-Payload)
BLOB_MAGIC = b"OKC1"  # Dateien
_NONCE = 12

_lock = threading.Lock()
_keys: dict[tuple[str, str, int], bytes] = {}  # (kind, uuid, version) -> Schluessel
_current: dict[tuple[str, str], int] = {}  # (kind, uuid) -> aktuelle Version
_uuids: dict[tuple[str, int], str] = {}  # (kind, id) -> uuid


def reset() -> None:
    """Zwischenspeicher leeren (Tests, Rotation, nach dem Loeschen)."""
    with _lock:
        _keys.clear()
        _current.clear()
        _uuids.clear()


def forget(kind: str, owner_id: int) -> None:
    """Zwischengespeicherte UUID einer ID vergessen (nach dem Loeschen: IDs koennen neu vergeben werden)."""
    with _lock:
        _uuids.pop((kind, int(owner_id)), None)


# --- Datenbankzugriff (Core, wahlweise ueber eine vorhandene Verbindung) ------------------
#
# ORM-Ereignisse (app/core/content_crypto.py) laufen mitten in einem Flush. Eine zweite
# Verbindung wuerde dort auf SQLite auf die Schreibsperre der ersten warten, deshalb nehmen alle
# Funktionen optional die Verbindung des Aufrufers.


def _run(conn, fn):
    if conn is not None:
        return fn(conn)
    from app.core.database_service import get_db_service

    with get_db_service().engine.begin() as own:
        return fn(own)


def owner_uuid(kind: str, owner_id: int, conn=None) -> str:
    """UUID eines Nutzers bzw. Knowledge-Bereichs zur numerischen ID."""
    if kind not in KINDS:
        raise ValueError(f"Unbekannte Schluesselart: {kind}")
    cache_key = (kind, int(owner_id))
    with _lock:
        cached = _uuids.get(cache_key)
    if cached:
        return cached
    from sqlalchemy import select

    from app.core.models import Knowledge, User

    table = (User if kind == USER else Knowledge).__table__
    value = _run(
        conn,
        lambda c: c.execute(
            select(table.c.uuid).where(table.c.id == int(owner_id))
        ).scalar(),
    )
    if not value:
        raise SecretKeyError(f"{kind} {owner_id} existiert nicht: kein Schluessel")
    with _lock:
        _uuids[cache_key] = value
    return value


# --- Schluessel -----------------------------------------------------------------------------


def _table():
    from app.core.models import DataKey

    return DataKey.__table__


def _wrap(key: bytes) -> str:
    return crypto.encrypt(base64.b64encode(key).decode())


def _unwrap(value: str) -> bytes:
    return base64.b64decode(crypto.decrypt(value))


def _load(kind: str, uuid: str, version: int | None, create: bool, conn=None):
    from datetime import datetime

    from sqlalchemy import select
    from sqlalchemy.exc import IntegrityError

    t = _table()

    def find(c):
        q = select(t.c.version, t.c.wrapped_key).where(
            t.c.owner_kind == kind, t.c.owner_uuid == uuid
        )
        q = (
            q.where(t.c.version == version)
            if version is not None
            else q.order_by(t.c.version.desc()).limit(1)
        )
        return c.execute(q).first()

    row = _run(conn, find)
    if row is None:
        if not create or version is not None:
            raise SecretKeyError(
                f"Kein Schluessel fuer {kind} {uuid}"
                + (f" (Version {version})" if version is not None else "")
                + ": der Inhalt ist nicht mehr lesbar (geloescht)."
            )
        key = AESGCM.generate_key(bit_length=256)
        values = dict(
            owner_kind=kind,
            owner_uuid=uuid,
            version=1,
            wrapped_key=_wrap(key),
            created_at=datetime.utcnow(),
        )
        try:
            if conn is not None:
                with conn.begin_nested():
                    conn.execute(t.insert().values(**values))
            else:
                _run(None, lambda c: c.execute(t.insert().values(**values)))
            return 1, key
        except IntegrityError:
            row = _run(conn, find)  # gleichzeitig angelegt: den anderen nehmen
            if row is None:
                raise
    return row[0], _unwrap(row[1])


def _key(kind: str, uuid: str, version: int | None = None, create: bool = True, conn=None):
    with _lock:
        v = version if version is not None else _current.get((kind, uuid))
        if v is not None and (kind, uuid, v) in _keys:
            return v, _keys[(kind, uuid, v)]
    v, key = _load(kind, uuid, version, create, conn)
    with _lock:
        _keys[(kind, uuid, v)] = key
        if version is None:
            _current[(kind, uuid)] = v
    return v, key


def rotate_data_key(kind: str, uuid: str) -> int:
    """Neue Schluessel-Version anlegen (neue Inhalte nutzen sie, alte bleiben lesbar)."""
    from datetime import datetime

    current, _ = _key(kind, uuid)
    t = _table()
    _run(
        None,
        lambda c: c.execute(
            t.insert().values(
                owner_kind=kind,
                owner_uuid=uuid,
                version=current + 1,
                wrapped_key=_wrap(AESGCM.generate_key(bit_length=256)),
                created_at=datetime.utcnow(),
            )
        ),
    )
    with _lock:
        _current.pop((kind, uuid), None)
    return current + 1


def shred(kind: str, uuid: str, conn=None) -> int:
    """Crypto-Shredding: alle Schluessel des Eigentuemers loeschen. Liefert die Anzahl."""
    t = _table()
    count = _run(
        conn,
        lambda c: c.execute(
            t.delete().where(t.c.owner_kind == kind, t.c.owner_uuid == uuid)
        ).rowcount,
    )
    with _lock:
        for k in [k for k in _keys if k[:2] == (kind, uuid)]:
            del _keys[k]
        _current.pop((kind, uuid), None)
    return count


# --- Ver- und Entschluesseln ----------------------------------------------------------------


def _aad(kind: str, uuid: str) -> bytes:
    return f"{kind}:{uuid}".encode()


def seal(kind: str, uuid: str, data: bytes, conn=None) -> bytes:
    """Bytes verschluesseln: ``OKC1`` + Version (4 Byte) + Nonce + Chiffrat mit Tag."""
    version, key = _key(kind, uuid, conn=conn)
    nonce = os.urandom(_NONCE)
    ct = AESGCM(key).encrypt(nonce, data, _aad(kind, uuid))
    return BLOB_MAGIC + struct.pack(">I", version) + nonce + ct


def is_sealed(data: bytes) -> bool:
    return isinstance(data, (bytes, bytearray)) and bytes(data[:4]) == BLOB_MAGIC


def unseal(kind: str, uuid: str, data: bytes, conn=None) -> bytes:
    """Gegenstueck zu ``seal``; Bytes ohne Kennung (Altbestand) kommen unveraendert zurueck.

    Raises:
        SecretKeyError: Schluessel geloescht, falscher Eigentuemer oder Daten veraendert.
    """
    if not is_sealed(data):
        return bytes(data)
    (version,) = struct.unpack(">I", bytes(data[4:8]))
    nonce, ct = bytes(data[8 : 8 + _NONCE]), bytes(data[8 + _NONCE :])
    _, key = _key(kind, uuid, version=version, create=False, conn=conn)
    try:
        return AESGCM(key).decrypt(nonce, ct, _aad(kind, uuid))
    except InvalidTag as e:
        raise SecretKeyError(
            "Inhalt laesst sich nicht entschluesseln (falscher Eigentuemer oder veraendert)"
        ) from e


def is_sealed_text(value) -> bool:
    return isinstance(value, str) and value.startswith(TEXT_PREFIX)


def seal_text(kind: str, uuid: str, value: str | None, conn=None) -> str | None:
    """Text verschluesseln (``okc:v1:<base64>``); leere und schon verschluesselte Werte bleiben."""
    if value is None or value == "" or is_sealed_text(value):
        return value
    blob = seal(kind, uuid, value.encode("utf-8"), conn)
    return TEXT_PREFIX + base64.b64encode(blob).decode("ascii")


def unseal_text(kind: str, uuid: str, value, conn=None):
    """Gegenstueck zu ``seal_text``; Klartext (Altbestand) kommt unveraendert zurueck."""
    if not is_sealed_text(value):
        return value
    blob = base64.b64decode(value[len(TEXT_PREFIX) :])
    return unseal(kind, uuid, blob, conn).decode("utf-8")


# --- Bequeme Varianten mit numerischer ID ---------------------------------------------------


def seal_for(kind: str, owner_id: int | None, value: str | None, conn=None) -> str | None:
    """Wie ``seal_text`` fuer eine Nutzer- bzw. Knowledge-ID; ohne Eigentuemer bleibt der Text."""
    if owner_id is None or value is None or value == "" or is_sealed_text(value):
        return value
    return seal_text(kind, owner_uuid(kind, owner_id, conn), value, conn)


def unseal_for(kind: str, owner_id: int | None, value, conn=None):
    if owner_id is None or not is_sealed_text(value):
        return value
    return unseal_text(kind, owner_uuid(kind, owner_id, conn), value, conn)
