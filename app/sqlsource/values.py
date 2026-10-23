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

"""Wertverzeichnis einer SQL-Quelle: finde zu einem Suchbegriff die passenden Werte der Quelle.

Das Verzeichnis haelt nur die *verschiedenen Werte* ausgewaehlter Textspalten (mit Haeufigkeit), keine
Zeilen. Das Modell sucht damit, welche exakten Schreibweisen es in ``WHERE spalte IN (...)`` einsetzen muss
("Stuehle" -> Stuhl, Buerostuhl, Gartenstuehle).

Die Methode wurde gemessen (57 Werte, 19 Anfragen; Praezision/Recall): exakt normalisiert 1.00/0.30,
Stemmer 1.00/0.52, Trigramme 1.00/0.57, Stamm als Teilstring 0.98/0.98, Kombination Stemmer + Teilstring +
Trigramme 0.98/1.00, dichte Einbettung (MiniLM) 0.12-0.53/0.62-0.75, Kombination mit Einbettung 0.37-0.53.
Einbettungen bringen hier Rauschen; darum rein lexikalisch.
"""

import re
import threading
import unicodedata

import snowballstemmer

_stemmer = snowballstemmer.stemmer("german")
_stem_lock = threading.Lock()
MIN_STEM_FOR_SUBSTRING = 4
MIN_DICE = 0.6


def key(text: str) -> str:
    """Vergleichsform: kleingeschrieben, ß=ss, ä/ae gleich, Akzente weg, nur Buchstaben und Ziffern."""
    s = (
        text.casefold()
        .replace("ß", "ss")
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
    )
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s.replace("ae", "a").replace("oe", "o").replace("ue", "u")


def stem(text: str) -> str:
    k = key(text)
    with _stem_lock:
        return " ".join(_stemmer.stemWord(w) for w in k.split())


def grams(text_key: str, n: int = 3) -> set[str]:
    s = f"  {text_key} "
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def dice(a: set[str], b: set[str]) -> float:
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


def entry(table: str, column: str, value: str, count: int) -> dict:
    return {
        "table": table,
        "column": column,
        "value": value,
        "count": count,
        "key": key(value),
        "stem": stem(value),
    }


def rank(
    term: str,
    entries: list[dict],
    limit: int = 20,
    table: str | None = None,
    column: str | None = None,
):
    """``[(eintrag, score)]``, beste zuerst. Score: exakt 1.0, gleicher Stamm 0.9, Stamm steckt im Wert
    (Kompositum) 0.7, aehnliche Schreibweise (Trigramme) bis 0.5."""
    tk = key(term or "")
    if len(tk) < 2:
        return []
    ts = stem(term)
    tg = None
    out = []
    for e in entries:
        if table and e["table"] != table:
            continue
        if column and e["column"] != column:
            continue
        if e["key"] == tk:
            score = 1.0
        elif e["stem"] == ts:
            score = 0.9
        elif len(ts) >= MIN_STEM_FOR_SUBSTRING and ts in e["key"]:
            score = 0.7
        else:
            if tg is None:
                tg = grams(tk)
            d = dice(tg, grams(e["key"]))
            if d < MIN_DICE:
                continue
            score = 0.5 * d
        out.append((e, score))
    out.sort(key=lambda p: (-p[1], -p[0]["count"], p[0]["value"]))
    return out[:limit]
