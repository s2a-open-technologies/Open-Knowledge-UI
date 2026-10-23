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
Tabellen, die ueber Seitengrenzen laufen, wieder zu einer Tabelle verbinden.

Docling liefert pro Seite eine eigene Tabelle. Zwei aufeinanderfolgende Tabellen gelten als Fortsetzung,
wenn alle Bedingungen zutreffen: aufeinanderfolgende Seiten, die erste endet unten, die zweite beginnt oben
auf ihrer Seite, im Dokument steht nichts dazwischen, Spaltenzahl und waagerechte Lage passen, und die
Kopfzeile der zweiten ist entweder die wiederholte Kopfzeile der ersten oder (wenn Docling die erste
Datenzeile als Kopf gelesen hat) sieht aus wie eine Datenzeile der ersten. Die Schwellen sind Anteile der
Seitenhoehe/-breite.
"""

import re
from dataclasses import dataclass

# Anteil der Seitenhoehe, innerhalb dessen eine Tabelle "unten" bzw. "oben" auf der Seite endet bzw. beginnt
EDGE_ZONE = 0.15
# erlaubte Abweichung der waagerechten Lage (Anteil der Seitenbreite)
X_TOLERANCE = 0.05


@dataclass
class Geometry:
    page: int
    top: float
    bottom: float
    left: float
    right: float
    page_height: float
    page_width: float


def table_geometry(item, doc) -> Geometry | None:
    prov = (getattr(item, "prov", None) or [None])[0]
    if prov is None:
        return None
    page = doc.pages.get(prov.page_no) if hasattr(doc, "pages") else None
    if page is None:
        return None
    box = prov.bbox.to_top_left_origin(page_height=page.size.height)
    return Geometry(
        prov.page_no, box.t, box.b, box.l, box.r, page.size.height, page.size.width
    )


def _norm(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _is_numeric(value) -> bool:
    return bool(re.fullmatch(r"[-+]?[\d.,\s]*\d[\d.,\s%€$]*", str(value).strip()))


def _kind(value) -> str:
    text = str(value).strip()
    if not text:
        return "leer"
    return "zahl" if _is_numeric(text) else "text"


def _unnamed(frame) -> bool:
    """Docling hat keine Kopfzeile erkannt: die Spalten heissen 0..n."""
    return all(isinstance(c, int) for c in frame.columns)


def header_of(frame) -> list[str]:
    """Kopfzeile (normalisiert); bei unbenannten Spalten die erste Zeile."""
    if _unnamed(frame) and frame.shape[0] > 0:
        return [_norm(c) for c in frame.iloc[0]]
    return [_norm(c) for c in frame.columns]


def _looks_like_row_of(columns, frame) -> bool:
    """Passen die Spaltennamen der Fortsetzung zu den Zeilen der ersten Tabelle (gleiche Zell-Art je Spalte)?"""
    if frame.shape[0] < 1:
        return False
    last = list(frame.iloc[-1])
    pairs = [
        (a, b)
        for a, b in zip(columns, last, strict=False)
        if _kind(a) != "leer" and _kind(b) != "leer"
    ]
    if not pairs:
        return False
    same = sum(1 for a, b in pairs if _kind(a) == _kind(b))
    return same / len(pairs) >= 0.75


def is_continuation(
    prev_frame, prev_geo: Geometry | None, next_frame, next_geo: Geometry | None
) -> bool:
    if prev_geo is None or next_geo is None or next_geo.page != prev_geo.page + 1:
        return False
    if prev_frame.shape[1] != next_frame.shape[1]:
        return False
    if (
        prev_geo.bottom < prev_geo.page_height * (1 - EDGE_ZONE)
        or next_geo.top > next_geo.page_height * EDGE_ZONE
    ):
        return False
    tol = prev_geo.page_width * X_TOLERANCE
    if (
        abs(prev_geo.left - next_geo.left) > tol
        or abs(prev_geo.right - next_geo.right) > tol
    ):
        return False
    if header_of(next_frame) == header_of(prev_frame):
        return True
    return not _unnamed(next_frame) and _looks_like_row_of(
        list(next_frame.columns), prev_frame
    )


def merge_frames(frames: list):
    """Haengt die Fortsetzungen an die erste Tabelle: wiederholte Kopfzeilen fallen weg, als Kopf gelesene Datenzeilen werden Zeilen."""
    import pandas as pd

    first = frames[0]
    header = header_of(first)
    parts = [first]
    for frame in frames[1:]:
        rows = list(frame.itertuples(index=False, name=None))
        if header_of(frame) == header:
            if _unnamed(frame) and rows:
                rows = rows[1:]  # die erste Zeile ist die wiederholte Kopfzeile
        else:
            rows = [
                tuple(frame.columns)
            ] + rows  # Docling hat die erste Datenzeile als Kopf gelesen
        parts.append(pd.DataFrame(rows, columns=first.columns))
    return pd.concat(parts, ignore_index=True)
