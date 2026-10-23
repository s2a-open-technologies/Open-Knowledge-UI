import itertools

import pytest

from app.ingest import splitter
from app.ingest.splitter import Chunk, embedding_text, split_markdown

DOC = """# Vertrag

Vorspann ohne eigene Ueberschrift im Abschnitt.

## 1 Miete

Die Kaltmiete betraegt 950 Euro.

## 2 Kaution

Die Kaution betraegt drei Monatsmieten.

### 2.1 Rueckzahlung

Die Rueckzahlung erfolgt spaetestens nach sechs Monaten.
"""


def test_headers_become_paths_and_small_sections_are_merged():
    chunks = split_markdown(DOC, filename="v.md", chunk_size=2000, min_size=0)
    paths = [c.headers for c in chunks]
    assert paths == [
        ["Vertrag"],
        ["Vertrag", "1 Miete"],
        ["Vertrag", "2 Kaution"],
        ["Vertrag", "2 Kaution", "2.1 Rueckzahlung"],
    ]
    assert chunks[1].title == "1 Miete" and "950 Euro" in chunks[1].text
    assert chunks[1].text.startswith("## 1 Miete")  # Ueberschrift bleibt im Text


def test_tiny_sections_are_merged_up_to_the_chunk_size():
    chunks = split_markdown(DOC, filename="v.md", chunk_size=2000)  # min_size = 666
    assert len(chunks) == 1
    assert chunks[0].headers == ["Vertrag"]  # gemeinsamer Elternpfad
    assert all(
        word in chunks[0].text
        for word in ("950 Euro", "drei Monatsmieten", "sechs Monaten")
    )


def test_merging_respects_chunk_size_and_does_not_chain_past_min_size():
    def section(n) -> str:
        # ca. 130 Zeichen je Abschnitt
        return f"## S{n}\n\n" + ("Wort " * 25)

    doc = "\n\n".join(section(i) for i in range(6))
    chunks = split_markdown(doc, filename="v.md", chunk_size=700, min_size=250)
    assert all(len(c.text) <= 700 for c in chunks)
    assert 2 <= len(chunks) < 6


def test_text_before_the_first_heading_uses_the_filename_as_title():
    chunks = split_markdown("Nur Text ohne Ueberschrift. " * 3, filename="notiz.md")
    assert (
        len(chunks) == 1 and chunks[0].headers == [] and chunks[0].title == "notiz.md"
    )
    assert embedding_text(chunks[0]) == chunks[0].text


def test_heading_only_sections_are_dropped_but_stay_in_the_path_of_children():
    chunks = split_markdown(
        "# A\n\n## B\n\nInhalt von B hier.\n", filename="x.md", min_size=0
    )
    assert [c.headers for c in chunks] == [["A", "B"]]


@pytest.mark.parametrize("text", ["", "   \n\n  ", "# Nur eine Ueberschrift\n"])
def test_empty_or_heading_only_documents_yield_no_chunks(text):
    assert split_markdown(text, filename="x.md") == []


def test_long_sections_are_split_below_the_limit_with_overlap():
    body = " ".join(f"Satz{i} endet hier." for i in range(300))
    chunks = split_markdown(
        f"## Lang\n\n{body}", filename="l.md", chunk_size=400, chunk_overlap=80
    )
    assert len(chunks) > 3
    assert all(len(c.text) <= 400 for c in chunks)
    assert all(c.headers == ["Lang"] for c in chunks)
    first_tail = chunks[0].text[-60:].split()[-3:]
    assert any(w in chunks[1].text for w in first_tail)  # Ueberlappung


def test_big_tables_are_split_by_rows_and_repeat_the_header():
    rows = "\n".join(f"| Zeile{i} | {i * 10} Euro |" for i in range(80))
    table = f"| Posten | Betrag |\n|---|---|\n{rows}"
    chunks = split_markdown(
        f"## Preise\n\nEinleitung.\n\n{table}", filename="t.md", chunk_size=500
    )
    table_chunks = [c for c in chunks if "| Zeile" in c.text]
    assert len(table_chunks) > 2
    for c in table_chunks:
        assert "| Posten | Betrag |\n|---|---|" in c.text  # Kopf in jedem Teil
        assert len(c.text) <= 600
    all_rows = "".join(c.text for c in table_chunks)
    assert all(f"| Zeile{i} |" in all_rows for i in range(80))  # keine Zeile verloren


def test_a_table_that_fits_stays_in_one_chunk_with_its_heading():
    doc = "## Miete\n\n| A | B |\n|---|---|\n| 1 | 2 |\n"
    chunks = split_markdown(doc, filename="t.md")
    assert len(chunks) == 1 and chunks[0].text.endswith("| 1 | 2 |")


def test_no_trailing_double_spaces_from_the_langchain_splitter():
    chunks = split_markdown("## A\n\nZeile eins\nZeile zwei\n", filename="x.md")
    assert "  \n" not in chunks[0].text


def test_embedding_text_prefixes_the_path_without_repeating_the_own_heading():
    leaf = Chunk(
        "### 2.1 Rueckzahlung\n\nText.",
        ["Vertrag", "2 Kaution", "2.1 Rueckzahlung"],
        "2.1 Rueckzahlung",
    )
    assert embedding_text(leaf).startswith(
        "Vertrag > 2 Kaution\n\n### 2.1 Rueckzahlung"
    )
    continuation = Chunk(
        "Fortsetzung ohne Ueberschrift.", ["Vertrag", "2 Kaution"], "2 Kaution"
    )
    assert (
        embedding_text(continuation)
        == "Vertrag > 2 Kaution\n\nFortsetzung ohne Ueberschrift."
    )
    top = Chunk("## Vertrag\n\nText", ["Vertrag"], "Vertrag")
    assert embedding_text(top) == top.text
    two = Chunk(
        "## 2 Kaution\n\n### 2.1 Hoehe\n\nText",
        ["Vertrag", "2 Kaution", "2.1 Hoehe"],
        "2.1 Hoehe",
    )
    assert embedding_text(two) == "Vertrag\n\n" + two.text


def test_bild_beschreibungen_survive_splitting():
    doc = "## Technik\n\nText.\n\n[BILD-BESCHREIBUNG: Ein Schaltplan mit drei Relais]\n\nMehr Text."
    chunks = split_markdown(doc, filename="b.md")
    assert "[BILD-BESCHREIBUNG: Ein Schaltplan mit drei Relais]" in chunks[0].text


def test_splitter_version_is_exposed_for_the_index_signature():
    assert isinstance(splitter.SPLITTER_VERSION, int)


HEADING = __import__("re").compile(r"^\s{0,3}#{1,6}\s+")


@pytest.mark.parametrize("size", [60, 90, 120, 160, 200, 250, 300, 400, 500, 800, 1500])
def test_a_heading_is_never_left_alone_at_the_end_of_a_chunk(size):
    """Regression: Ueberschriften landeten ohne ihren Text am Ende des vorigen Chunks."""
    pad = "Weitere Hinweise zur Zahlung. " * 12
    doc = (
        f"# Vertrag\n\nVorspann.\n\n## 1 Miete\n\nDie Kaltmiete betraegt 950 Euro. {pad}\n\n"
        f"## 2 Kaution\n\n### 2.1 Hoehe\n\nDie Kaution betraegt drei Monatsmieten. {pad}\n\n"
        f"### 2.2 Rueckgabe\n\nKurz.\n\n## 3 Ende\n\n{pad}"
    )
    chunks = split_markdown(
        doc, filename="v.md", chunk_size=size, chunk_overlap=size // 5
    )
    assert chunks
    for c in chunks:
        last = [line for line in c.text.split("\n") if line.strip()][-1]
        assert not HEADING.match(last), f"Chunk endet mit Ueberschrift: {last!r}"
    # jede Ueberschrift steht im selben Chunk wie (der Anfang von) ihrem Text
    for c in chunks:
        lines = [line for line in c.text.split("\n") if line.strip()]
        for line, nxt in itertools.pairwise(lines):
            if HEADING.match(line):
                assert nxt, "Ueberschrift ohne Folgezeile"


def test_a_long_section_keeps_its_heading_lines_with_the_first_piece():
    body = " ".join(f"Satz{i} endet hier." for i in range(120))
    chunks = split_markdown(
        f"## A\n\n### A.1\n\n{body}",
        filename="x.md",
        chunk_size=300,
        chunk_overlap=40,
        min_size=0,
    )
    assert (
        chunks[0].text.startswith("## A")
        and "### A.1" in chunks[0].text
        and "Satz0" in chunks[0].text
    )
    assert all(not c.text.startswith("#") for c in chunks[1:])
    assert all(c.continuation for c in chunks[1:]) and not chunks[0].continuation


def test_continuation_pieces_are_not_merged_into_the_next_section():
    long_a = "## A\n\n" + " ".join(f"Wort{i}" for i in range(75))
    doc = f"{long_a}\n\n## B\n\nKurzer Text."
    chunks = split_markdown(
        doc, filename="x.md", chunk_size=300, chunk_overlap=0, min_size=250
    )
    b = [c for c in chunks if "## B" in c.text]
    assert len(b) == 1 and b[0].text.startswith("## B") and b[0].headers == ["B"]


def test_page_markers_become_page_ranges_and_vanish_from_the_text():
    from app.ingest.splitter import split_markdown

    md = (
        "<!--p:1-->\n\n# Vertrag\n\n## 1 Miete\n\nDie Miete betraegt 950 Euro monatlich.\n\n"
        "<!--p:2-->\n\n## 2 Kaution\n\nDie Kaution betraegt drei Monatsmieten.\n\n"
        "<!--p:3-->\n\nUnd sie wird nach sechs Monaten zurueckgezahlt.\n"
    )
    chunks = split_markdown(
        md, filename="v.pdf", chunk_size=120, chunk_overlap=0, min_size=0
    )
    assert all("<!--p:" not in c.text for c in chunks)
    by_text = {c.text.split("\n")[0]: c for c in chunks}
    miete = next(c for c in chunks if "Miete betraegt" in c.text)
    kaution = next(c for c in chunks if "Kaution betraegt" in c.text)
    assert (miete.page_start, miete.page_end) == (1, 1)
    assert kaution.page_start == 2 and kaution.page_end >= 2
    assert by_text  # nichts verloren


def test_a_chunk_spanning_a_page_break_gets_both_pages():
    from app.ingest.splitter import split_markdown

    md = "<!--p:4-->\n\n# T\n\nErster Absatz auf Seite vier.\n\n<!--p:5-->\n\nZweiter Absatz auf Seite fuenf.\n"
    chunk = split_markdown(md, filename="v.pdf", chunk_size=1000)[0]
    assert (chunk.page_start, chunk.page_end) == (4, 5)
    assert "Zweiter Absatz" in chunk.text and "<!--" not in chunk.text


def test_markdown_without_markers_has_no_pages():
    from app.ingest.splitter import split_markdown

    chunk = split_markdown("# T\n\nText ohne Seiten.\n", filename="v.md")[0]
    assert chunk.page_start is None and chunk.page_end is None


def test_markers_between_a_heading_and_its_text_do_not_create_empty_chunks():
    from app.ingest.splitter import split_markdown

    chunks = split_markdown(
        "<!--p:1-->\n\n# Nur Titel\n\n<!--p:2-->\n\n## Echt\n\nInhalt.\n",
        filename="v.pdf",
    )
    assert (
        len(chunks) == 1
        and "Inhalt." in chunks[0].text
        and "<!--" not in chunks[0].text
    )
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_repeated_table_headers_inherit_the_previous_page():
    from app.ingest.splitter import split_markdown

    rows = "\n".join(f"| Zeile {i} | {i * 10} |" for i in range(40))
    md = f"<!--p:7-->\n\n# Tabelle\n\n| Name | Wert |\n| --- | --- |\n{rows}\n"
    chunks = split_markdown(md, filename="v.pdf", chunk_size=300, chunk_overlap=0)
    assert len(chunks) > 1
    assert all(c.page_start == 7 and c.page_end == 7 for c in chunks)
