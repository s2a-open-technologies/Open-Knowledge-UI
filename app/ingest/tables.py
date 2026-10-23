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
Tabellen aus Dokumenten als CSV.

Tabellen aus PDF/DOCX sind als Text in einer Vektordatenbank schlecht auszuwerten
(Summen, Filter, Vergleiche). Deshalb legt das Einlesen jede erkannte Tabelle als CSV im
Datenordner des Knowledge ab. Dort sehen sie die pandas-Tools. Einen eigenen Eintrag in der
Vektordatenbank haben sie nicht: sie gehoeren zum Dokument, dessen Chunks den Verweis tragen
(``render_reference``). Im Markdown bleibt eine kleine Tabelle
lesbar stehen, eine grosse wird durch einen Verweis mit Dateiname, Spalten und den ersten
Zeilen ersetzt.
"""

import re
from collections.abc import Callable
from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)

# Tabellen mit weniger Zeilen/Spalten sind meist Layout, keine Daten.
MIN_ROWS = 2
MIN_COLS = 2
# Bis hierhin bleibt die Tabelle im Markdown stehen (zusaetzlich zur CSV).
INLINE_MAX_ROWS = 10
INLINE_MAX_CHARS = 700
PREVIEW_ROWS = 3

TableSink = Callable[["object", int], "dict | None"]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "dokument"


def short_id(file_group_id: str) -> str:
    return file_group_id.replace("-", "")[:8]


def table_filename(stem: str, file_group_id: str, number: int) -> str:
    """``vertrag--ab12cd34--tabelle-3.csv`` (eindeutig pro Dokument und Tabelle)."""
    return f"{_slug(stem)[:60]}--{short_id(file_group_id)}--tabelle-{number}.csv"


_TABLE_NAME = re.compile(r"^.+--[0-9a-f]{8}--tabelle-\d+\.csv$")


def is_extracted_table(filename: str) -> bool:
    """True fuer CSVs, die beim Einlesen aus einem Dokument entstanden sind."""
    return bool(_TABLE_NAME.match(Path(filename).name))


def column_hints_text(df, max_columns: int = 12) -> str:
    """Kurze Inhaltshinweise je Spalte (Wertebereich bzw. Beispielwerte), damit die Tabelle
    ueber ihren Inhalt gefunden wird."""
    parts = []
    for col in list(df.columns)[:max_columns]:
        series = df[col].dropna()
        if series.empty:
            continue
        if str(series.dtype).startswith(("int", "float")):
            parts.append(f"`{col}`: {series.min():g} bis {series.max():g}")
        else:
            examples = ", ".join(str(v)[:30] for v in series.astype(str).unique()[:3])
            parts.append(f"`{col}`: z. B. {examples}")
    return ("Inhalt: " + "; ".join(parts) + "\n") if parts else ""


def table_pattern(file_group_id: str) -> str:
    return f"*--{short_id(file_group_id)}--tabelle-*.csv"


def make_sink(paths, stem: str, file_group_id: str) -> TableSink:
    """Liefert die Funktion, die eine Tabelle als CSV speichert.

    ``sink(dataframe, nummer)`` gibt ``{"name", "rows", "columns"}`` zurueck (``name`` so,
    wie ihn die pandas-Tools erwarten, also mit Knowledge-Praefix) oder ``None``, wenn die
    Tabelle zu klein ist.
    """

    def sink(df, number: int) -> dict | None:
        if df.shape[0] < MIN_ROWS or df.shape[1] < MIN_COLS:
            return None
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
        name = table_filename(stem, file_group_id, number)
        folder = Path(paths.data_dir)
        folder.mkdir(parents=True, exist_ok=True)
        df.to_csv(folder / name, index=False, encoding="utf-8")
        return {
            "name": f"{paths.name_prefix}{name}",
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "frame": df,
        }

    return sink


def render_reference(
    info: dict, number: int, table_markdown: str, pages: tuple[int, int] | None = None
) -> str:
    """Ersatztext fuer eine extrahierte Tabelle im Markdown."""
    name, rows, df = info["name"], info["rows"], info["frame"]
    label = (
        f"TABELLE {number} (Seiten {pages[0]}–{pages[1]})"
        if pages
        else f"TABELLE {number}"
    )
    # Die Tabelle gehoert zum Dokument und hat keinen eigenen Eintrag in der Vektordatenbank:
    # ueber diesen Verweis wird sie mit dem Dokument gefunden, deshalb stehen die Inhaltshinweise hier.
    try:
        hints = column_hints_text(df).strip()
    except Exception:
        hints = ""
    hints = f"\n{hints}" if hints else ""
    if rows <= INLINE_MAX_ROWS and len(table_markdown) <= INLINE_MAX_CHARS:
        return (
            f"{table_markdown}\n\n[{label}: als Datei `{name}` ({rows} Zeilen)]{hints}"
        )
    columns = ", ".join(info["columns"])
    try:
        preview = df.head(PREVIEW_ROWS).to_markdown(index=False)
    except Exception:  # to_markdown braucht tabulate; ohne Vorschau weiter
        preview = ""
    text = (
        f"[{label}: Datei `{name}`, {rows} Zeilen, Spalten: {columns}. "
        "Für Auswertungen (Summen, Filter, Vergleiche) das Tool run_pandas mit diesem "
        "Dateinamen nutzen.]"
    )
    text += hints
    return f"{text}\n\nErste Zeilen:\n\n{preview}" if preview else text


def delete_tables(paths, collection_name: str, file_group_id: str) -> int:
    """Entfernt die CSVs eines Dokuments samt Verweis-Chunks. Liefert die Anzahl."""
    folder = Path(paths.data_dir)
    removed = 0
    if not folder.is_dir():
        return 0
    for csv in folder.glob(table_pattern(file_group_id)):
        csv.unlink(missing_ok=True)
        removed += 1
        try:
            from app.vector.ingestion import delete_data_reference

            delete_data_reference(collection_name, f"{paths.name_prefix}{csv.name}")
        except Exception as e:
            logger.warning(f"Verweis-Chunk fuer {csv.name} nicht loeschbar: {e}")
    return removed
