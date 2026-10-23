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
Markdown -> Chunks.

Zuerst teilt der ``MarkdownHeaderTextSplitter`` an den Ueberschriften (``#`` bis
``####``). Zu grosse Abschnitte werden weiter zerlegt (Absaetze rekursiv, Tabellen
zeilenweise mit wiederholtem Tabellenkopf), sehr kleine Abschnitte mit dem
nachfolgenden zusammengefasst. Jeder Chunk kennt seinen Ueberschriften-Pfad.

``SPLITTER_VERSION`` fliesst in die Index-Signatur ein: Aendert sich das Verhalten,
gelten bestehende Indizes als veraltet.
"""

import re
from dataclasses import dataclass, field

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

SPLITTER_VERSION = 1

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
_HEADER_KEYS = [key for _, key in HEADERS]
_TABLE_SEPARATOR = re.compile(r"^\|[\s:|-]+\|\s*$")
_HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s+")
# Seitenmarker aus dem PDF-Parser (``<!--p:3-->`` in einer eigenen Zeile); sie bleiben beim
# Zerlegen im Text und werden danach in ``Chunk.page_start/page_end`` ueberfuehrt.
PAGE_MARK = "<!--p:{}-->"
_PAGE_MARK = re.compile(r"^[ \t]*<!--p:(\d+)-->[ \t]*$", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    headers: list[str] = field(default_factory=list)
    title: str = ""
    # Folgestueck eines zerlegten Abschnitts (beginnt nicht mit der Ueberschrift)
    continuation: bool = False
    # Seiten der Fundstelle (nur wenn der Parser Seitenmarker lieferte, z. B. PDF)
    page_start: int | None = None
    page_end: int | None = None


def _split_table(lines: list[str], chunk_size: int) -> list[str]:
    """Teilt eine Markdown-Tabelle zeilenweise; der Kopf steht in jedem Teil."""
    head: list[str] = []
    if len(lines) >= 2 and _TABLE_SEPARATOR.match(lines[1]):
        head, lines = lines[:2], lines[2:]
    base = "\n".join(head)
    parts: list[str] = []
    current: list[str] = []
    size = len(base)
    for line in lines:
        if current and size + len(line) + 1 > chunk_size:
            parts.append("\n".join([*head, *current]))
            current, size = [], len(base)
        current.append(line)
        size += len(line) + 1
    if current or not parts:
        parts.append("\n".join([*head, *current]))
    return parts


def _segments(text: str) -> list[tuple[str, str]]:
    """Zerlegt Text in ('text'|'table', Inhalt)-Segmente."""
    segments: list[tuple[str, str]] = []
    buf: list[str] = []
    kind = "text"

    def flush():
        if buf and "".join(buf).strip():
            segments.append((kind, "\n".join(buf).strip("\n")))
        buf.clear()

    for line in text.split("\n"):
        line_kind = "table" if line.lstrip().startswith("|") else "text"
        if line_kind != kind:
            flush()
            kind = line_kind
        buf.append(line)
    flush()
    return segments


def _split_section(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Zerlegt einen zu grossen Abschnitt in Stuecke <= chunk_size (wo moeglich).

    Fuehrende Ueberschriftszeilen bleiben immer beim ersten Textstueck: Sonst trennt
    der rekursive Splitter sie vom Text ab, und sie landen allein in einem Chunk.
    """
    lines = text.split("\n")
    n_heading = 0
    while n_heading < len(lines) and (
        not lines[n_heading].strip() or _HEADING_LINE.match(lines[n_heading])
    ):
        n_heading += 1
    prefix = "\n".join(lines[:n_heading]).strip()
    body = "\n".join(lines[n_heading:]).strip()
    if not body:
        return [text]

    recursive = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",  # der Satzpunkt bleibt am Ende des Satzes, nicht vor dem naechsten
    )
    pieces: list[str] = []
    for kind, content in _segments(body):
        if len(content) <= chunk_size:
            pieces.append(content)
        elif kind == "table":
            pieces.extend(_split_table(content.split("\n"), chunk_size))
        else:
            pieces.extend(recursive.split_text(content))

    # Kleine Stuecke (z. B. Absatz + Tabelle) zu einem Chunk zusammenpacken.
    packed: list[str] = []
    for piece in pieces:
        if packed and len(packed[-1]) + len(piece) + 2 <= chunk_size:
            packed[-1] = f"{packed[-1]}\n\n{piece}"
        else:
            packed.append(piece)
    if prefix and packed:
        packed[0] = f"{prefix}\n\n{packed[0]}"  # darf chunk_size leicht ueberschreiten
    return packed


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    prefix = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        prefix.append(x)
    return prefix


def _has_body(text: str) -> bool:
    """Mehr als nur Ueberschriftszeilen?"""
    return any(
        line.strip() and not _HEADING_LINE.match(line) for line in text.split("\n")
    )


_KEY_TOKENS = 6


def _strip_page_marks(markdown: str) -> tuple[str, list[str], list[int | None]]:
    """Entfernt die Seitenmarker und liefert Woerter samt Seite je Wort (fuer ``_assign_pages``)."""
    page: int | None = None
    lines: list[str] = []
    tokens: list[str] = []
    pages: list[int | None] = []
    for line in markdown.split("\n"):
        m = _PAGE_MARK.match(line)
        if m:
            page = int(m.group(1))
            continue
        lines.append(line)
        for word in line.split():
            tokens.append(word)
            pages.append(page)
    return "\n".join(lines), tokens, pages


def _find(tokens: list[str], needle: list[str], start: int) -> int:
    """Erste Position ab ``start``, an der ``needle`` als Wortfolge steht, sonst -1."""
    if not needle:
        return -1
    first = needle[0]
    for i in range(start, len(tokens) - len(needle) + 1):
        if tokens[i] == first and tokens[i : i + len(needle)] == needle:
            return i
    return -1


def _assign_pages(
    chunks: list[Chunk], tokens: list[str], pages: list[int | None]
) -> list[Chunk]:
    """Setzt ``page_start/page_end`` je Chunk, indem seine Woerter im Dokument gesucht werden.

    Wiederholte Tabellenkoepfe oder nicht auffindbare Stellen erben den Stand des Vorgaengers.
    """
    if not any(p is not None for p in pages):
        return chunks
    cursor = 0
    last_end: int | None = None
    for chunk in chunks:
        words = chunk.text.split()
        head = words[:_KEY_TOKENS]
        pos = _find(tokens, head, cursor)
        if pos < 0:
            chunk.page_start = chunk.page_end = last_end
            continue
        tail = words[-_KEY_TOKENS:]
        end = _find(tokens, tail, pos)
        end_idx = (
            (end + len(tail) - 1)
            if end >= 0
            else min(pos + len(words) - 1, len(tokens) - 1)
        )
        chunk.page_start = pages[pos]
        chunk.page_end = pages[end_idx] if pages[end_idx] is not None else pages[pos]
        last_end = chunk.page_end
        cursor = pos + 1  # Ueberlappung: der naechste Chunk beginnt nach diesem Anfang
    return chunks


def split_markdown(
    markdown: str,
    *,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    min_size: int | None = None,
) -> list[Chunk]:
    """Teilt Markdown in Chunks mit Ueberschriften-Pfad.

    Args:
        chunk_size: Zielgroesse in Zeichen (Tabellenzeilen und lange Woerter koennen sie ueberschreiten).
        chunk_overlap: Ueberlappung beim Zerlegen langer Absaetze.
        min_size: Abschnitte darunter werden mit dem naechsten zusammengefasst, solange das
            Ergebnis in ``chunk_size`` passt (Default: ein Drittel von ``chunk_size``).
    """
    if not markdown or not markdown.strip():
        return []
    markdown, tokens, token_pages = _strip_page_marks(markdown)
    chunk_overlap = min(chunk_overlap, max(chunk_size // 2, 0))
    min_size = chunk_size // 3 if min_size is None else min_size

    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS, strip_headers=False
    ).split_text(markdown)

    chunks: list[Chunk] = []
    for section in sections:
        headers = [section.metadata[k] for k in _HEADER_KEYS if k in section.metadata]
        # Der LangChain-Splitter haengt an Zeilenenden zwei Leerzeichen an (Markdown-Umbruch).
        text = re.sub(r" +\n", "\n", section.page_content).strip()
        if not text or not _has_body(text):
            continue  # reine Ueberschrift: steht im Pfad der Unterabschnitte
        pieces = (
            [text]
            if len(text) <= chunk_size
            else _split_section(text, chunk_size, chunk_overlap)
        )
        for i, piece in enumerate(pieces):
            if piece.strip():
                chunks.append(
                    Chunk(
                        piece.strip(),
                        headers,
                        headers[-1] if headers else filename,
                        continuation=i > 0,
                    )
                )

    # Kleine Abschnitte mit dem nachfolgenden zusammenfassen.
    merged: list[Chunk] = []
    for chunk in chunks:
        prev = merged[-1] if merged else None
        # Folgestuecke werden nicht mit dem naechsten Abschnitt vermischt.
        if (
            prev
            and not prev.continuation
            and len(prev.text) < min_size
            and len(prev.text) + len(chunk.text) + 2 <= chunk_size
        ):
            prev.text = f"{prev.text}\n\n{chunk.text}"
            prev.headers = _common_prefix(prev.headers, chunk.headers)
            prev.title = prev.headers[-1] if prev.headers else filename
        else:
            merged.append(
                Chunk(chunk.text, list(chunk.headers), chunk.title, chunk.continuation)
            )
    return _assign_pages(merged, tokens, token_pages)


def embedding_text(chunk: Chunk) -> str:
    """Text fuer Embedding/BM25: Ueberschriften-Pfad voran (Kontext fuer kurze Abschnitte).

    Ueberschriften, die der Text schon am Anfang enthaelt, stehen nicht doppelt im Pfad.
    """
    leading: set[str] = set()
    for line in chunk.text.split("\n"):
        if not line.strip():
            continue
        if not _HEADING_LINE.match(line):
            break
        leading.add(_HEADING_LINE.sub("", line).strip())
    path = [h for h in chunk.headers if h not in leading]
    if not path:
        return chunk.text
    return f"{' > '.join(path)}\n\n{chunk.text}"
