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
"""Text fuer das Modell: Schema und Abfrageergebnisse."""


def _cell(value) -> str:
    if value is None:
        return "NULL"
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def result_text(
    source: str, columns: list, rows: list, truncated: bool, row_limit: int
) -> str:
    if not columns:
        return f"Quelle „{source}“: Die Abfrage lieferte keine Spalten."
    n = len(rows)
    lines = [f"Quelle „{source}“ · {n} Zeile{'' if n == 1 else 'n'}", ""]
    lines.append("| " + " | ".join(_cell(c) for c in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    if truncated:
        lines += [
            "",
            f"(Ergebnis auf {row_limit} Zeilen gekürzt – fasse mit COUNT/SUM/GROUP BY zusammen oder filtere genauer.)",
        ]
    return "\n".join(lines)


DIALECTS = {
    "postgresql": "PostgreSQL",
    "mysql": "MySQL/MariaDB",
    "mssql": "SQL Server",
    "sqlite": "SQLite",
}


def schema_text(source: str, tables: list, dialect: str = "") -> str:
    lines = [f"Quelle „{source}“: {len(tables)} Tabellen"]
    if (
        dialect in DIALECTS
    ):  # sonst nimmt das Modell Funktionen eines anderen Dialekts (z. B. strftime auf PostgreSQL)
        lines.append(
            f"SQL-Dialekt: {DIALECTS[dialect]} (Datums- und Textfunktionen dieses Dialekts verwenden)"
        )
    for t in tables:
        lines += [
            "",
            f"### {t['name']}" + (" (View)" if t.get("kind") == "view" else ""),
        ]
        if t.get("comment"):
            lines.append(t["comment"])
        for c in t["columns"]:
            flags = ", ".join(
                x
                for x in (
                    c["type"],
                    "PK" if c["pk"] else "",
                    "" if c["nullable"] else "NOT NULL",
                )
                if x
            )
            lines.append(
                f"- {c['name']} ({flags})"
                + (f": {c['comment']}" if c.get("comment") else "")
            )
        for fk in t["foreign_keys"]:
            lines.append(
                f"- FK: {', '.join(fk['columns'])} → {fk['ref_table']}({', '.join(fk['ref_columns'])})"
            )
    return "\n".join(lines)
