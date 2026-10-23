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
PII-Filter ueber das ganze Gespraech (Chat).

* ``mask_conversation``: Zusammenfassung, Verlauf und neue Frage gehen als **ein** Text mit
  Trennmarken in **einem** Aufruf an den Sanitizer. Dadurch vergibt der Dienst im ganzen Kontext
  eindeutige Platzhalter (``[NAME_1]``, ``[NAME_2]`` ...) und liefert ein gemeinsames Mapping.
  Kommen die Marken nicht unversehrt zurueck, wird abgebrochen (fail-closed).
* ``StreamDemasker``: setzt Platzhalter in der gestreamten Antwort schon waehrend des Streamens
  zurueck, auch wenn sie ueber Token-Grenzen gesplittet ankommen.
"""

import re
import threading
from contextvars import ContextVar

from app.chat.pii_client import (
    PIIFilterClient,
    PIIFilterUnavailable,
    mapping_pairs,
    mask_or_fail,
)

# Bewusst ohne Buchstaben: spaCy-NER hat "=====MSG-1=====" teilweise als Entitaet erkannt (gemessen)
MARKER = "=====#{}====="
_MARKER_RE = re.compile(r"=====#(\d+)=====")
BASE_TIMEOUT_S = 5.0
CHARS_PER_EXTRA_SECOND = 2000
MAX_TIMEOUT_S = 60.0


def timeout_for(chars: int, base: float = BASE_TIMEOUT_S) -> float:
    """Zeitlimit waechst mit der Textlaenge: ``base`` + 1 s je 2000 Zeichen, hoechstens 60 s (nie unter ``base``)."""
    return max(base, min(MAX_TIMEOUT_S, base + chars / CHARS_PER_EXTRA_SECOND))


def join_texts(texts: list[str]) -> str:
    return "".join(
        f"\n\n{MARKER.format(i)}\n\n{text}" for i, text in enumerate(texts)
    ).lstrip("\n")


def split_texts(joined: str, count: int) -> list[str]:
    """Teilt den maskierten Text an den Marken; wirft, wenn sie nicht genau einmal und in Reihenfolge da sind."""
    found = [int(m.group(1)) for m in _MARKER_RE.finditer(joined)]
    if found != list(range(count)):
        raise PIIFilterUnavailable(
            "der Filter hat die Trennmarken des Verlaufs verändert"
        )
    parts = _MARKER_RE.split(
        joined
    )  # ['', '0', '\n\ntext0\n\n', '1', '\n\ntext1', ...]
    texts = []
    for i in range(count):
        part = parts[2 + 2 * i]
        part = part[2:] if part.startswith("\n\n") else part.lstrip("\n")
        if i < count - 1 and part.endswith("\n\n"):
            part = part[:-2]
        texts.append(part)
    return texts


_YEAR = re.compile(r"(19|20)\d{2}")


def keep_years(texts: list[str], mapping) -> tuple[list[str], object]:
    """Macht Platzhalter fuer reine Jahreszahlen ("2026") wieder rueckgaengig.

    Die Entitaetenerkennung hielt Jahre mehrfach fuer Organisationen oder Orte; die Frage hiess dann
    "in [ORGANISATION_1]", und das Modell suchte statt eines Zeitraums nach einem Namen. Eine Jahreszahl
    allein ist keine personenbezogene Angabe (ein vollstaendiges Datum, Geburtsdaten und alles andere
    bleiben unter Kontrolle des Filters).
    """
    pairs = mapping_pairs(mapping)
    years = [(p, o) for p, o in pairs if _YEAR.fullmatch(o.strip())]
    if not years:
        return texts, mapping
    for placeholder, original in years:
        texts = [t.replace(placeholder, original.strip()) for t in texts]
    return texts, {p: o for p, o in pairs if (p, o) not in years}


DEFAULT_MAX_CHARS = 12000


def max_chars(client) -> int:
    """Hoechste Textlaenge je Filteraufruf (Einstellung ``PII_FILTER_MAX_CHARS``): Gemessen brachte ein Text von
    50 000 Zeichen den Filter zum Absturz, 20 000 Zeichen brauchten ueber 30 s, 12 000 warm unter 3 s."""
    return int(getattr(client, "max_chars", None) or DEFAULT_MAX_CHARS)


def split_text(text: str, limit: int) -> list[str]:
    """Teilt ``text`` in Stuecke bis ``limit`` Zeichen, bevorzugt an Absaetzen, Zeilen und Satzenden.

    Die Stuecke ergeben aneinandergehaengt genau den Text zurueck."""
    pieces, rest = [], text
    while len(rest) > limit:
        window = rest[:limit]
        cut = 0
        for marker, offset in (("\n\n", 2), ("\n", 1), (". ", 2), (" ", 1)):
            idx = window.rfind(marker)
            if idx > limit // 2:
                cut = idx + offset
                break
        cut = cut or limit
        pieces.append(rest[:cut])
        rest = rest[cut:]
    pieces.append(rest)
    return pieces


def mask_conversation(
    client: PIIFilterClient, texts: list[str]
) -> tuple[list[str], object]:
    """Maskiert alle ``texts``; was in ein Stueck passt, in EINEM Aufruf, sonst in Stuecken.

    Bei Stuecken vergibt der Dienst je Aufruf neu zu zaehlen an: ``merge_masked`` ordnet die Platzhalter
    der Stuecke in ein gemeinsames Mapping ein (gleicher Wert = gleicher Platzhalter).
    """
    limit = max_chars(client)
    if not texts or len(join_texts(texts)) <= limit:
        return _mask_batch(client, texts)
    units = [
        (i, piece)
        for i, text in enumerate(texts)
        for piece in split_text(text, limit - 200)
    ]
    batches, current, size = [], [], 0
    for unit in units:
        if current and size + len(unit[1]) + 40 > limit:
            batches.append(current)
            current, size = [], 0
        current.append(unit)
        size += len(unit[1]) + 40
    batches.append(current)
    known: dict[str, str] = {}
    pieces: list[str] = []
    for batch in batches:
        masked, mapping = _mask_batch(client, [piece for _, piece in batch])
        batch_pairs = mapping_pairs(mapping)
        if batch_pairs:
            masked = split_texts(
                merge_masked(known, join_texts(masked), batch_pairs), len(masked)
            )
        pieces.extend(masked)
    out = [""] * len(texts)
    for (i, _), piece in zip(units, pieces):
        out[i] += piece
    return keep_years(out, known)


def _mask_batch(client: PIIFilterClient, texts: list[str]) -> tuple[list[str], object]:
    """Maskiert alle ``texts`` in einem Aufruf.

    Returns:
        ``(maskierte Texte in gleicher Reihenfolge, Mapping Platzhalter -> Original)``.
    Raises:
        PIIFilterUnavailable: Dienst nicht erreichbar, Fehlerantwort oder zerstoerte Marken.
    """
    if not texts:
        return [], {}
    joined = join_texts(texts)
    try:
        # reuse_placeholders: optionaler Hinweis, gleicher Wert = gleicher Platzhalter (EntityGuard >= 1.2);
        # Dienste, die das Feld nicht kennen, ignorieren es.
        result = client.filter_text(
            joined,
            timeout=timeout_for(
                len(joined), getattr(client, "timeout", None) or BASE_TIMEOUT_S
            ),
            extra={"reuse_placeholders": True},
        )
    except Exception as e:  # filter_text faengt selbst; Sicherheitsnetz
        raise PIIFilterUnavailable(type(e).__name__) from None
    if result.error:
        raise PIIFilterUnavailable(result.error)
    return keep_years(split_texts(result.filtered, len(texts)), result.mapping)


def unavailable_message(error) -> str:
    """Satz fuer Nutzer, warum der PII-Filter nicht benutzt werden konnte (ohne Adressen oder Ausnahmetexte).

    Unterscheidet, was wirklich los war: ein langsamer Filter ist nicht dasselbe wie ein abgelehnter Key
    oder ein nicht erreichbarer Dienst."""
    text = str(error)
    if "Timeout" in text or "timed out" in text:
        limit = re.search(r"\(([^)]*)\)", text)
        return (
            "Der PII-Filter hat nicht rechtzeitig geantwortet (Zeitüberschreitung"
            + (f" nach {limit.group(1)}" if limit else "")
            + ")"
        )
    if text.startswith(("HTTP 401", "HTTP 403")):
        return f"Der PII-Filter lehnt die Anmeldung ab ({text.split(':')[0][:12]}, API-Key prüfen)"
    if text.startswith("HTTP "):
        return f"Der PII-Filter hat mit einem Fehler geantwortet ({text.split(':')[0][:12]})"
    if "Trennmarken" in text:
        return (
            "Der PII-Filter hat den Text verändert (Trennmarken des Verlaufs verändert)"
        )
    return "Der PII-Filter ist nicht erreichbar"


def short_reason(error: str) -> str:
    """Kurzer, fuer Nutzer geeigneter Grund (ohne interne Adressen oder Ausnahmetexte)."""
    text = str(error)
    if text.startswith("HTTP "):
        return text.split(":")[0][:12]
    if "Timeout" in text or "timed out" in text:
        return "Zeitüberschreitung"
    if "Trennmarken" in text:
        return "Trennmarken des Verlaufs verändert"
    return "nicht erreichbar"


def demask_text(client: PIIFilterClient, text: str, mapping) -> str:
    return client.demask(text, mapping) if mapping else text


def merge_masked(known: dict, text: str, new_pairs: list[tuple[str, str]]) -> str:
    """Ordnet die Platzhalter eines spaeteren Filteraufrufs in die bekannten ein (siehe Plan Task 6)."""
    reverse = {original: placeholder for placeholder, original in known.items()}
    taken = {placeholder for placeholder, _ in new_pairs}
    rename: dict[str, str] = {}
    for placeholder, original in sorted(
        new_pairs, key=lambda p: len(p[0]), reverse=True
    ):
        if original in reverse:
            target = reverse[original]
        elif known.get(placeholder, original) == original:
            target = placeholder
        else:
            n = 2
            while True:
                candidate = (
                    f"{placeholder[:-1]}_{n}{placeholder[-1]}"
                    if placeholder[-1:] in "]>)}"
                    else f"{placeholder}_{n}"
                )
                if candidate not in known and candidate not in taken:
                    break
                n += 1
            target = candidate
            taken.add(target)
        if target != placeholder:
            rename[placeholder] = target
        known[target] = original
        reverse[original] = target
    if rename:
        pattern = re.compile(
            "|".join(re.escape(k) for k in sorted(rename, key=len, reverse=True))
        )
        text = pattern.sub(lambda m: rename[m.group(0)], text)
    return text


class PIISession:
    """Alle Platzhalter eines Chat-Requests: Kontext, Antwort und Werkzeug-Ergebnisse teilen ein Mapping."""

    def __init__(self, client, mapping=None):
        self.client = client
        self._known: dict[str, str] = {}
        self._lock = threading.Lock()
        for placeholder, original in mapping_pairs(mapping):
            self._known[placeholder] = original

    def pairs(self) -> list[tuple[str, str]]:
        with self._lock:
            return sorted(self._known.items(), key=lambda p: len(p[0]), reverse=True)

    def mapping(self) -> dict:
        with self._lock:
            return dict(self._known)

    def unmask(self, text: str) -> str:
        for placeholder, original in self.pairs():
            text = text.replace(placeholder, original)
        return text

    def mask_tool_text(self, text: str) -> str:
        """Maskiert Werkzeug-Text (nur mit eingerichtetem Filter). Fehler -> ``PIIFilterUnavailable``."""
        if self.client is None or not text:
            return text
        out = []
        for piece in split_text(
            text, max_chars(self.client)
        ):  # grosse Ergebnisse stueckweise (siehe max_chars)
            masked, mapping = mask_or_fail(self.client, piece)
            (masked,), mapping = keep_years([masked], mapping)
            pairs = mapping_pairs(mapping)
            if pairs:
                with self._lock:
                    masked = merge_masked(self._known, masked, pairs)
            out.append(masked)
        return "".join(out)

    def mask_values(self, values: list) -> list:
        """Maskiert einzelne Datenwerte (Zellen) in EINEM Filteraufruf; Platzhalter sind sitzungsweit eindeutig.

        Nur echte Daten gehen durch den Filter, nicht der Text drumherum (Spaltenueberschriften, Hinweise,
        Schluesselwoerter): die Erkennung hielt dort Woerter wie "IN" oder "Verzeichnis" fuer Namen und Orte
        und zerstoerte die Anweisungen an das Modell. Schlichte Zahlen (bis 9 Stellen, z. B. Jahr, Anzahl,
        ID) bleiben unveraendert. Fehler -> ``PIIFilterUnavailable``.
        """
        out = list(values)
        if self.client is None:
            return out
        positions = [
            i
            for i, v in enumerate(out)
            if isinstance(v, str) and v.strip() and not _PLAIN_NUMBER.match(v.strip())
        ]
        if not positions:
            return out
        masked, mapping = mask_conversation(self.client, [out[i] for i in positions])
        pairs = mapping_pairs(mapping)
        if pairs:
            with self._lock:
                masked = split_texts(
                    merge_masked(self._known, join_texts(masked), pairs), len(masked)
                )
        for i, text in zip(positions, masked):
            out[i] = text
        return out


_PLAIN_NUMBER = re.compile(r"^-?\d{1,9}([.,]\d+)?$")

_session: ContextVar["PIISession | None"] = ContextVar("pii_session", default=None)


def set_session(session: "PIISession | None"):
    return _session.set(session)


def get_session() -> "PIISession | None":
    return _session.get()


class StreamDemasker:
    """Setzt Platzhalter in einem Stream zurueck und haelt moegliche Platzhalter-Anfaenge zurueck."""

    def __init__(self, mapping):
        self._session = mapping if isinstance(mapping, PIISession) else None
        self._static = (
            sorted(mapping_pairs(mapping), key=lambda p: len(p[0]), reverse=True)
            if self._session is None
            else []
        )
        self._buffer = ""

    @property
    def _pairs(self):
        return self._session.pairs() if self._session is not None else self._static

    def _hold(self, text: str) -> int:
        """Laenge des Endes von ``text``, das der Anfang eines Platzhalters sein koennte."""
        longest = 0
        for placeholder, _original in self._pairs:
            for k in range(min(len(placeholder) - 1, len(text)), longest, -1):
                if text.endswith(placeholder[:k]):
                    longest = k
                    break
        return longest

    def _replace(self, text: str) -> str:
        for placeholder, original in self._pairs:
            text = text.replace(placeholder, original)
        return text

    def feed(self, chunk: str) -> str:
        if not self._pairs:
            return chunk
        self._buffer += chunk
        text = self._replace(self._buffer)
        hold = self._hold(text)
        self._buffer = text[len(text) - hold :] if hold else ""
        return text[: len(text) - hold] if hold else text

    def flush(self) -> str:
        rest, self._buffer = self._buffer, ""
        return self._replace(rest) if self._pairs else rest
