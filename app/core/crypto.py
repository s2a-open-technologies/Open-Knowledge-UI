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

"""
Geheimnisse in der Datenbank verschluesseln (Fernet).

Verschluesselt werden Zugangsdaten: API-Keys der Verbindungen, Zugangsdaten externer und
SQL-Quellen (``knowledge.source_config``) und sensible Einstellungen (``SENSITIVE_KEYS``).
Nicht verschluesselt werden Dokumente, Chat-Verlaeufe und Vektoren (die Suche braucht sie im
Klartext); Passwoerter von Nutzern sind ohnehin nur als Hash gespeichert.

Schluessel: Umgebungsvariable ``ENCRYPTION_KEY`` (Fernet-Schluessel; mehrere, durch Komma getrennt,
fuer die Rotation: der erste verschluesselt, alle entschluesseln). Fehlt sie, erzeugt die App beim
ersten Start eine Schluesseldatei ``<DATA_DIR>/.encryption_key`` (Rechte 600) und warnt: das schuetzt
vor einem Datenbank-Dump allein, nicht vor Zugriff auf das ganze Datenverzeichnis. Fuer echten Schutz
den Schluessel getrennt von der Datenbank halten (Env/Secret).

Format: ``enc:v1:<Fernet-Token>``. Werte ohne Praefix gelten als Klartext (Altbestand) und werden
beim Start einmalig verschluesselt (``encrypt_existing``). Ist ein Wert verschluesselt und der
Schluessel fehlt oder passt nicht, wird NIE still ein leerer Wert geliefert: ``SecretKeyError``.
"""

import os
import threading
from logging import getLogger
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator

logger = getLogger(__name__)

PREFIX = "enc:v1:"
ENV_KEY = "ENCRYPTION_KEY"
KEY_FILE = ".encryption_key"

_lock = threading.Lock()
_fernet: MultiFernet | None = None


class SecretKeyError(RuntimeError):
    """Ein verschluesselter Wert laesst sich nicht entschluesseln (Schluessel fehlt oder passt nicht)."""


def key_file_path() -> Path:
    return Path(os.getenv("DATA_DIR", "./data")) / KEY_FILE


def _load_keys() -> list[bytes]:
    raw = (os.getenv(ENV_KEY) or "").strip()
    if raw:
        keys = [k.strip().encode() for k in raw.split(",") if k.strip()]
        for key in keys:
            Fernet(key)  # ungueltigen Schluessel sofort melden (ValueError)
        return keys
    path = key_file_path()
    if path.exists():
        return [path.read_bytes().strip()]
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(key)
    logger.warning(
        f"{ENV_KEY} ist nicht gesetzt: Schluessel in {path} erzeugt. Fuer echten Schutz den Schluessel "
        f"getrennt von der Datenbank halten (Umgebungsvariable {ENV_KEY} oder Secret)."
    )
    return [key]


def _cipher() -> MultiFernet:
    global _fernet
    with _lock:
        if _fernet is None:
            _fernet = MultiFernet([Fernet(k) for k in _load_keys()])
        return _fernet


def reset() -> None:
    """Schluessel neu laden (Tests, Rotation)."""
    global _fernet
    with _lock:
        _fernet = None


def key_source() -> str:
    """``env`` oder ``file``: woher der Schluessel kommt (fuer die Anzeige in der Administration)."""
    return "env" if (os.getenv(ENV_KEY) or "").strip() else "file"


def is_encrypted(value) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


def encrypt(value):
    """Verschluesselt ``value``; leere und bereits verschluesselte Werte bleiben unveraendert."""
    if value is None or value == "" or is_encrypted(value):
        return value
    return PREFIX + _cipher().encrypt(str(value).encode()).decode()


def decrypt(value):
    """Klartext zu ``value``; Werte ohne Praefix (Altbestand) kommen unveraendert zurueck."""
    if not is_encrypted(value):
        return value
    try:
        return _cipher().decrypt(value[len(PREFIX) :].encode()).decode()
    except InvalidToken as e:
        raise SecretKeyError(
            f"Ein verschluesselter Wert laesst sich nicht entschluesseln: {ENV_KEY} fehlt oder ist nicht der "
            f"Schluessel, mit dem die Datenbank verschluesselt wurde (Datei: {key_file_path()})."
        ) from e


class EncryptedText(TypeDecorator):
    """Text-Spalte, die beim Schreiben verschluesselt und beim Lesen entschluesselt."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return encrypt(value)

    def process_result_value(self, value, dialect):
        return decrypt(value)


# Spalten mit Zugangsdaten (Tabelle, Spalte), die ``encrypt_existing`` im Klartext vorfindet
ENCRYPTED_COLUMNS = (
    ("connections", "api_key", "id"),
    ("knowledge", "source_config", "id"),
)


def encrypt_existing(engine, sensitive_keys) -> int:
    """Verschluesselt vorhandene Klartext-Zugangsdaten (idempotent) und prueft den Schluessel.

    Raises:
        SecretKeyError: es gibt verschluesselte Werte, die der Schluessel nicht oeffnet.
    """
    from sqlalchemy import text

    count = 0
    with engine.begin() as conn:
        targets = [(table, column, pk, "") for table, column, pk in ENCRYPTED_COLUMNS]
        for table, column, pk, where in targets:
            rows = conn.execute(
                text(
                    f"SELECT {pk}, {column} FROM {table} WHERE {column} IS NOT NULL AND {column} <> ''"
                )
            ).all()
            for row_id, value in rows:
                if is_encrypted(value):
                    decrypt(
                        value
                    )  # falscher Schluessel -> SecretKeyError, bevor irgendetwas geaendert wird
                    continue
                conn.execute(
                    text(f"UPDATE {table} SET {column} = :v WHERE {pk} = :id"),
                    {"v": encrypt(value), "id": row_id},
                )
                count += 1
        keys = sorted(sensitive_keys)
        if keys:
            rows = conn.execute(
                text(
                    "SELECT key, value FROM settings WHERE value IS NOT NULL AND value <> '' AND key IN ({})".format(
                        ",".join(f":k{i}" for i in range(len(keys)))
                    )
                ),
                {f"k{i}": k for i, k in enumerate(keys)},
            ).all()
            for key, value in rows:
                if is_encrypted(value):
                    decrypt(value)
                    continue
                conn.execute(
                    text("UPDATE settings SET value = :v WHERE key = :k"),
                    {"v": encrypt(value), "k": key},
                )
                count += 1
            rows = conn.execute(
                text(
                    "SELECT key, default_value FROM settings WHERE default_value IS NOT NULL AND default_value <> '' AND key IN ({})".format(
                        ",".join(f":k{i}" for i in range(len(keys)))
                    )
                ),
                {f"k{i}": k for i, k in enumerate(keys)},
            ).all()
            for key, value in rows:
                if not is_encrypted(value):
                    conn.execute(
                        text("UPDATE settings SET default_value = :v WHERE key = :k"),
                        {"v": encrypt(value), "k": key},
                    )
                    count += 1
    if count:
        logger.info(f"{count} Zugangsdaten in der Datenbank verschluesselt")
    return count
