"""Textvergleich des Wertverzeichnisses. Die Beispielwerte stammen aus der Messung (Stuhl/Stühle usw.)."""

import pytest

from app.sqlsource import values as v

VALUES = [
    "Stuhl",
    "Bürostuhl",
    "Drehstuhl",
    "Esszimmerstuhl",
    "Gartenstühle",
    "Stühle",
    "Klappstuhl",
    "Tisch",
    "Schreibtisch",
    "Esstisch",
    "Tische",
    "Couchtisch",
    "Lampe",
    "Stehlampe",
    "Schreibtischlampe",
    "Schrank",
    "Kleiderschrank",
    "Schränke",
    "Regal",
    "Bücherregal",
    "Regale",
    "Sofa",
    "Couch",
    "Sessel",
    "Bett",
    "Doppelbett",
    "Straße",
    "Hauptstraße",
    "Bahnhofstraße",
    "Strasse",
    "München",
    "Köln",
    "Nürnberg",
    "Düsseldorf",
    "Hamburg",
    "Berlin",
    "Arzt",
    "Ärzte",
    "Ärztin",
    "Anwalt",
    "Anwältin",
    "Hund",
    "Katze",
    "Maus",
    "Mäuse",
    "Haus",
    "Häuser",
    "Lagerhalle",
    "Halle",
    "Stahlträger",
    "Träger",
    "Schraube",
    "Schrauben",
    "Mutter",
    "Muttern",
    "Dichtung",
    "Dichtungen",
]
ENTRIES = [v.entry("items", "name", value, 1) for value in VALUES]
STUHL = {
    "Stuhl",
    "Bürostuhl",
    "Drehstuhl",
    "Esszimmerstuhl",
    "Gartenstühle",
    "Stühle",
    "Klappstuhl",
}
QUERIES = {
    "Stühle": STUHL,
    "Stuhl": STUHL,
    "stuehle": STUHL,
    "Tische": {"Tisch", "Schreibtisch", "Esstisch", "Tische", "Couchtisch"},
    "Strasse": {"Straße", "Hauptstraße", "Bahnhofstraße", "Strasse"},
    "Muenchen": {"München"},
    "Koeln": {"Köln"},
    "MÜNCHEN": {"München"},
    "Schränke": {"Schrank", "Kleiderschrank", "Schränke"},
    "Buecherregal": {"Bücherregal"},
    "Bürostul": {"Bürostuhl"},
    "Ärzte": {"Arzt", "Ärzte", "Ärztin"},
    "Anwälte": {"Anwalt", "Anwältin"},
    "Mäuse": {"Maus", "Mäuse"},
    "Häuser": {"Haus", "Häuser"},
    "Schrauben": {"Schraube", "Schrauben"},
    "Dichtung": {"Dichtung", "Dichtungen"},
    "Lampen": {"Lampe", "Stehlampe", "Schreibtischlampe"},
    "Halle": {"Halle", "Lagerhalle"},
}


def found(term):
    return {e["value"] for e, _ in v.rank(term, ENTRIES, limit=100)}


@pytest.mark.parametrize("term,expected", list(QUERIES.items()))
def test_each_query_finds_what_it_should(term, expected):
    got = found(term)
    assert expected <= got, f"{term}: verpasst {expected - got}"


def test_overall_precision_and_recall_hold_the_measured_level():
    tp = fp = fn = 0
    for term, expected in QUERIES.items():
        got = found(term)
        tp, fp, fn = (
            tp + len(got & expected),
            fp + len(got - expected),
            fn + len(expected - got),
        )
    assert tp / (tp + fn) >= 0.98  # Recall (gemessen 1.00)
    assert tp / (tp + fp) >= 0.95  # Praezision (gemessen 0.98)


def test_exact_matches_rank_before_stems_before_compounds():
    ranked = [e["value"] for e, _ in v.rank("Stuhl", ENTRIES, limit=10)]
    assert ranked[0] == "Stuhl"
    assert ranked.index("Stühle") < ranked.index(
        "Bürostuhl"
    )  # gleicher Stamm vor Kompositum


def test_unrelated_words_do_not_match_and_short_stems_do_not_explode():
    assert found("Gurke") == set()
    assert found("Bett") == {"Bett", "Doppelbett"}
    assert found("a") == set() and found("") == set()


def test_ties_prefer_the_more_frequent_value():
    entries = [v.entry("t", "c", "Stuhl", 3), v.entry("t", "c", "Stühle", 40)]
    assert [e["value"] for e, _ in v.rank("Stuhl", entries)][:2] == [
        "Stuhl",
        "Stühle",
    ]  # exakt zuerst
    entries = [v.entry("t", "c", "Stuhl A", 3), v.entry("t", "c", "Stuhl B", 40)]
    assert next(e["value"] for e, _ in v.rank("Stuhl", entries)) == "Stuhl B"


def test_multi_word_terms_and_umlaut_spellings():
    e = [v.entry("t", "c", "Max Mustermann", 5), v.entry("t", "c", "Müller GmbH", 2)]
    assert [x["value"] for x, _ in v.rank("max mustermann", e)] == ["Max Mustermann"]
    assert [x["value"] for x, _ in v.rank("Mueller GmbH", e)] == ["Müller GmbH"]


def test_limit_and_column_filter():
    e = [v.entry("a", "x", "Stuhl", 1), v.entry("b", "y", "Stuhl", 1)]
    assert len(v.rank("Stuhl", e, limit=1)) == 1
    assert [x["table"] for x, _ in v.rank("Stuhl", e, table="b")] == ["b"]
    assert [x["column"] for x, _ in v.rank("Stuhl", e, column="x")] == ["x"]
