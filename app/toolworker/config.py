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

"""Einstellungen des Tool-Workers (Datenbank), Token aus Umgebung/_FILE (nie in der Datenbank)."""

import ipaddress
import os
import urllib.parse
from dataclasses import dataclass

from app.core.database_service import get_db_service
from app.core.settings_service import SettingsService

# (Schluessel, Standardwert, Beschreibung). Die Untergrenze fuer den Speicher wurde in
# tests/test_tool_worker_sandbox.py gemessen (pandas reserviert virtuellen Speicher).
PANDAS_SETTINGS = [
    (
        "TOOL_WORKER_URL",
        "http://tool-worker:8090",
        "Adresse des Tool-Workers (fuehrt Pandas-Code isoliert aus)",
    ),
    ("PANDAS_TIMEOUT_S", "30", "Zeitlimit einer Pandas-Auswertung in Sekunden"),
    ("PANDAS_MAX_MEMORY_MB", "1024", "Speicherlimit einer Pandas-Auswertung in MB"),
    (
        "PANDAS_MAX_OUTPUT_CHARS",
        "20000",
        "Maximale Laenge der Ausgabe einer Pandas-Auswertung in Zeichen",
    ),
]

# Wert muss zu alembic/versions/0025_sql_knowledge.py passen (Test: test_migrations_seed_the_same_keys)
SQL_PROMPT_DEFAULT = (
    "[Hinweis: Für diese Frage sind SQL-Datenbanken verbunden ({sources}). Rufe zuerst sql_schema auf, "
    "wenn du Tabellen und Spalten nicht kennst, und schreibe dann ein einzelnes SELECT mit sql_query. "
    "Nur lesende Abfragen sind erlaubt, Ergebnisse sind auf wenige Zeilen begrenzt: aggregiere mit "
    "COUNT, SUM, GROUP BY, statt viele Zeilen zu laden. Qualifiziere Tabellen bei Bedarf mit dem Schema, "
    "nenne Spalten einzeln und zeige menschenlesbare Namen (JOIN statt nackter IDs). Gib die Zahlen "
    "und Namen aus den Ergebnissen wieder und erfinde nichts.]"
)

SQL_SETTINGS = [
    (
        "SQL_WORKER_URL",
        "http://tool-worker-sql:8090",
        "Adresse des SQL-Workers (fragt Datenbanken lesend ab)",
    ),
    ("SQL_ROW_LIMIT", "50", "Maximale Zeilen je SQL-Abfrage, die das Modell sieht"),
    ("SQL_TIMEOUT_S", "15", "Zeitlimit einer SQL-Abfrage in Sekunden"),
    (
        "SQL_MAX_CELL_CHARS",
        "500",
        "Maximale Laenge eines Zellwerts in SQL-Ergebnissen in Zeichen",
    ),
    (
        "SQL_PROMPT",
        SQL_PROMPT_DEFAULT,
        "Hinweis an das Modell bei SQL-Quellen (Platzhalter {sources})",
    ),
]
SQL_VALUES_HINT_DEFAULT = "Für Textwerte (Namen, Kategorien, Orte) rufe vor dem SELECT sql_find_values mit dem Begriff aus der Frage auf: es liefert die exakten Schreibweisen aus der Datenbank (Ein-/Mehrzahl, Umlaute, zusammengesetzte Wörter). Setze diese Werte mit IN (...) ein, statt zu raten."

VALUES_SETTINGS = [
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
        SQL_VALUES_HINT_DEFAULT,
        "Hinweis an das Modell, wenn ein Wertverzeichnis vorhanden ist",
    ),
]
SETTINGS = PANDAS_SETTINGS + SQL_SETTINGS + VALUES_SETTINGS
RANGES = {
    "timeout_s": (1, 300),
    "memory_mb": (128, 8192),
    "max_output_chars": (1000, 200_000),
    "sql_row_limit": (1, 1000),
    "sql_timeout_s": (1, 120),
    "sql_max_cell_chars": (50, 5000),
}
KEYS = {
    "timeout_s": "PANDAS_TIMEOUT_S",
    "memory_mb": "PANDAS_MAX_MEMORY_MB",
    "max_output_chars": "PANDAS_MAX_OUTPUT_CHARS",
    "sql_row_limit": "SQL_ROW_LIMIT",
    "sql_timeout_s": "SQL_TIMEOUT_S",
    "sql_max_cell_chars": "SQL_MAX_CELL_CHARS",
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class WorkerSettings:
    url: str
    timeout_s: int
    memory_mb: int
    max_output_chars: int


def token() -> str:
    path = os.getenv("TOOL_WORKER_TOKEN_FILE")
    if path:
        try:
            with open(path, encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError:
            return ""
    return os.getenv("TOOL_WORKER_TOKEN", "").strip()


def load() -> WorkerSettings:
    session = get_db_service().get_session()
    try:
        s = SettingsService(session)
        return WorkerSettings(
            url=(s.get("TOOL_WORKER_URL", "http://tool-worker:8090") or "")
            .strip()
            .rstrip("/"),
            timeout_s=s.get_int("PANDAS_TIMEOUT_S", 30),
            memory_mb=s.get_int("PANDAS_MAX_MEMORY_MB", 1024),
            max_output_chars=s.get_int("PANDAS_MAX_OUTPUT_CHARS", 20000),
        )
    finally:
        session.close()


@dataclass(frozen=True)
class SqlSettings:
    url: str
    row_limit: int
    timeout_s: int
    max_cell_chars: int


def load_sql() -> SqlSettings:
    session = get_db_service().get_session()
    try:
        s = SettingsService(session)
        return SqlSettings(
            url=(s.get("SQL_WORKER_URL", "http://tool-worker-sql:8090") or "")
            .strip()
            .rstrip("/"),
            row_limit=s.get_int("SQL_ROW_LIMIT", 50),
            timeout_s=s.get_int("SQL_TIMEOUT_S", 15),
            max_cell_chars=s.get_int("SQL_MAX_CELL_CHARS", 500),
        )
    finally:
        session.close()


def values_settings() -> tuple[int, int]:
    """(hoechstens so viele verschiedene Werte je Spalte, hoechstens so viele Treffer der Wertsuche)."""
    session = get_db_service().get_session()
    try:
        s = SettingsService(session)
        return max(1, s.get_int("SQL_VALUES_MAX_DISTINCT", 5000)), max(
            1, s.get_int("SQL_VALUES_MAX_RESULTS", 20)
        )
    finally:
        session.close()


def values_hint() -> str:
    session = get_db_service().get_session()
    try:
        return (
            SettingsService(session).get("SQL_VALUES_HINT", SQL_VALUES_HINT_DEFAULT)
            or SQL_VALUES_HINT_DEFAULT
        )
    finally:
        session.close()


def sql_prompt() -> str:
    session = get_db_service().get_session()
    try:
        return (
            SettingsService(session).get("SQL_PROMPT", SQL_PROMPT_DEFAULT)
            or SQL_PROMPT_DEFAULT
        )
    finally:
        session.close()


def view() -> dict:
    w = load()
    sql = load_sql()
    return {
        "sql_url": sql.url,
        "sql_row_limit": sql.row_limit,
        "sql_timeout_s": sql.timeout_s,
        "sql_max_cell_chars": sql.max_cell_chars,
        "url": w.url,
        "timeout_s": w.timeout_s,
        "memory_mb": w.memory_mb,
        "max_output_chars": w.max_output_chars,
        "token_configured": bool(token()),
    }


def _check_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ConfigError("Die Adresse muss mit http:// oder https:// beginnen.")
    try:
        link_local = ipaddress.ip_address(parsed.hostname).is_link_local
    except ValueError:
        link_local = False  # Hostname statt IP
    if link_local:
        raise ConfigError("Link-lokale Adressen sind nicht erlaubt.")
    return url.strip().rstrip("/")


def save(data: dict) -> None:
    values: dict[str, str] = {}
    if "url" in data:
        values["TOOL_WORKER_URL"] = _check_url(str(data["url"]))
    if "sql_url" in data:
        values["SQL_WORKER_URL"] = _check_url(str(data["sql_url"]))
    for field, key in KEYS.items():
        if field in data:
            lo, hi = RANGES[field]
            try:
                n = int(data[field])
            except (TypeError, ValueError):
                raise ConfigError(f"{field}: ganze Zahl erwartet.") from None
            if not lo <= n <= hi:
                raise ConfigError(f"{field}: erlaubt sind {lo} bis {hi}.")
            values[key] = str(n)
    session = get_db_service().get_session()
    try:
        s = SettingsService(session)
        for key, value in values.items():
            s.set(key, value)
        session.commit()
    finally:
        session.close()
