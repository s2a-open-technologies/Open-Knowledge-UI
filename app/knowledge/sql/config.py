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

"""Konfiguration einer SQL-Quelle (Knowledge ``kind='sql'``): Parsen, Pruefen, Weitergabe an den Worker."""

import ipaddress
import json
import re
from dataclasses import dataclass, field

DIALECTS = ("postgresql", "mysql", "mssql", "sqlite")
DEFAULT_PORTS = {"postgresql": 5432, "mysql": 3306, "mssql": 1433, "sqlite": 0}
HOST_RE = re.compile(r"^[A-Za-z0-9._-]+$")
NAME_RE = re.compile(r"^[A-Za-z0-9_$.\- ]+$")
SCHEMA_RE = re.compile(r"^[A-Za-z0-9_$.\-]*$")


class SqlConfigError(ValueError):
    """Ungueltige Angabe (Text ist fuer den Admin gedacht)."""


@dataclass(frozen=True)
class SqlConfig:
    dialect: str = "postgresql"
    host: str = ""
    port: int = 0
    database: str = ""
    user: str = ""
    password: str = field(default="", repr=False)
    schema: str = ""
    instance: str = ""  # nur MSSQL: benannte Instanz (host\\instanz)
    tables: tuple[str, ...] = ()
    blocked_columns: dict = field(default_factory=dict)
    # Wertverzeichnis: Textspalten, deren verschiedene Werte durchsuchbar sind ({tabelle: (spalte, ...)})
    value_columns: dict = field(default_factory=dict)

    def to_dict(self, with_password: bool = True) -> dict:
        d = {
            "dialect": self.dialect,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "schema": self.schema,
            "instance": self.instance,
            "tables": list(self.tables),
            "blocked_columns": {t: list(c) for t, c in self.blocked_columns.items()},
            "value_columns": {t: list(c) for t, c in self.value_columns.items()},
        }
        if with_password:
            d["password"] = self.password
        return d

    def public(self) -> dict:
        """Fuer Antworten an die Oberflaeche: nie das Passwort."""
        return {
            **self.to_dict(with_password=False),
            "has_password": bool(self.password),
        }

    def connection(self) -> dict:
        return {
            "dialect": self.dialect,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "schema": self.schema,
            "instance": self.instance,
        }

    def allowed_tables(self) -> set[str] | None:
        return set(self.tables) if self.tables else None

    def blocked(self) -> dict[str, set[str]]:
        return {t: set(c) for t, c in self.blocked_columns.items() if c}


def parse_config(raw: dict | str | None) -> SqlConfig:
    if isinstance(raw, str):
        raw = json.loads(raw or "{}")
    raw = raw or {}
    dialect = str(raw.get("dialect") or "postgresql").strip().lower()
    try:
        port = int(raw.get("port") or DEFAULT_PORTS.get(dialect, 0))
    except (TypeError, ValueError):
        port = -1
    blocked = {
        str(t).strip(): tuple(str(c).strip() for c in cols if str(c).strip())
        for t, cols in (raw.get("blocked_columns") or {}).items()
        if isinstance(cols, (list, tuple)) and str(t).strip()
    }
    value_columns = {
        str(t).strip(): tuple(str(c).strip() for c in cols if str(c).strip())
        for t, cols in (raw.get("value_columns") or {}).items()
        if isinstance(cols, (list, tuple)) and str(t).strip()
    }
    return SqlConfig(
        value_columns={t: c for t, c in value_columns.items() if c},
        dialect=dialect,
        host=str(raw.get("host") or "").strip(),
        port=port,
        database=str(raw.get("database") or "").strip(),
        user=str(raw.get("user") or "").strip(),
        password=str(raw.get("password") or ""),
        schema=str(raw.get("schema") or "").strip(),
        instance=str(raw.get("instance") or "").strip(),
        tables=tuple(
            str(t).strip() for t in (raw.get("tables") or []) if str(t).strip()
        ),
        blocked_columns=blocked,
    )


def validate(cfg: SqlConfig) -> None:
    if cfg.dialect not in DIALECTS:
        raise SqlConfigError("Unbekannter Datenbanktyp.")
    if not cfg.database:
        raise SqlConfigError("Die Datenbank (bei SQLite die Datei) fehlt.")
    if cfg.dialect == "sqlite":
        parts = cfg.database.replace("\\", "/").split("/")
        if cfg.database.startswith("/") or ".." in parts or "\x00" in cfg.database:
            raise SqlConfigError(
                "SQLite: nur ein relativer Dateiname im freigegebenen Ordner ist erlaubt."
            )
    else:
        if not cfg.host or not HOST_RE.match(cfg.host):
            raise SqlConfigError("Der Host fehlt oder enthält unerlaubte Zeichen.")
        try:
            link_local = ipaddress.ip_address(cfg.host).is_link_local
        except ValueError:
            link_local = False  # Hostname statt IP
        if link_local:
            raise SqlConfigError("Link-lokale Adressen sind nicht erlaubt.")
        if not 1 <= cfg.port <= 65535:
            raise SqlConfigError("Der Port muss zwischen 1 und 65535 liegen.")
        if not cfg.user:
            raise SqlConfigError("Der Benutzer fehlt.")
    if not SCHEMA_RE.match(cfg.schema):
        raise SqlConfigError("Das Schema enthält unerlaubte Zeichen.")
    if not SCHEMA_RE.match(cfg.instance):
        raise SqlConfigError("Die Instanz enthält unerlaubte Zeichen.")
    for name in (
        list(cfg.tables)
        + list(cfg.blocked_columns)
        + [c for cols in cfg.blocked_columns.values() for c in cols]
        + list(cfg.value_columns)
        + [c for cols in cfg.value_columns.values() for c in cols]
    ):
        if not NAME_RE.match(name):
            raise SqlConfigError(f"Unerlaubter Tabellen- oder Spaltenname: {name[:40]}")


def same_target(a: SqlConfig, b: SqlConfig) -> bool:
    key = lambda c: (c.dialect, c.host, c.instance, c.port, c.database, c.user)
    return key(a) == key(b)


def merge_password(new: SqlConfig, old: SqlConfig | None) -> SqlConfig:
    """Leeres Passwort behaelt das gespeicherte, aber nur fuer dasselbe Ziel."""
    if new.password or old is None or not same_target(new, old):
        return new
    return SqlConfig(**{**new.__dict__, "password": old.password})
