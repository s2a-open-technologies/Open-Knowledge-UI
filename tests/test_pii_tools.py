import pytest

from app.chat.pii_client import PIIFilterResult, PIIFilterUnavailable
from app.chat.pii_session import (
    PIISession,
    StreamDemasker,
    get_session,
    merge_masked,
    set_session,
)


class FakeFilter:
    """Wie ein Sanitizer: nummeriert die Namen je Aufruf neu ([NAME_1] ...)."""

    NAMES = ("Anna", "Bernd", "Clara")

    def __init__(self):
        self.calls = []
        self.fail = False

    def filter_text(self, text, timeout=None, extra=None):
        self.calls.append(text)
        if self.fail:
            return PIIFilterResult(original=text, filtered=text, error="HTTP 500")
        mapping, out = {}, text
        for name in self.NAMES:
            if name in out:
                ph = f"[NAME_{len(mapping) + 1}]"
                mapping[ph] = name
                out = out.replace(name, ph)
        return PIIFilterResult(original=text, filtered=out, mapping=mapping)


def test_a_reused_placeholder_for_another_person_is_renamed_and_both_unmask_correctly():
    known = {"[NAME_1]": "Anna"}
    text = merge_masked(known, "[NAME_1] kauft ein", [("[NAME_1]", "Bernd")])
    assert text == "[NAME_1_2] kauft ein"
    assert known == {"[NAME_1]": "Anna", "[NAME_1_2]": "Bernd"}


def test_the_same_person_keeps_the_existing_placeholder():
    known = {"[NAME_1]": "Anna"}
    assert (
        merge_masked(
            known,
            "[NAME_2] und [NAME_1]",
            [("[NAME_2]", "Anna"), ("[NAME_1]", "Bernd")],
        )
        == "[NAME_1] und [NAME_1_2]"
    )
    assert known == {"[NAME_1]": "Anna", "[NAME_1_2]": "Bernd"}


def test_renaming_never_collides_with_other_new_placeholders():
    known = {"[NAME_1]": "Anna", "[NAME_1_2]": "Zora"}
    merge_masked(known, "[NAME_1]", [("[NAME_1]", "Bernd")])
    assert set(known.values()) == {"Anna", "Zora", "Bernd"} and len(known) == 3


def test_unmask_replaces_longest_placeholders_first():
    s = PIISession(None, {"[NAME_1]": "Anna", "[NAME_10]": "Zoe"})
    assert s.unmask("[NAME_10] trifft [NAME_1]") == "Zoe trifft Anna"


def test_session_accepts_the_legacy_list_mapping():
    s = PIISession(None, [{"placeholder": "[NAME_1]", "original": "Anna"}])
    assert s.mapping() == {"[NAME_1]": "Anna"}


def test_tool_results_get_unique_placeholders_over_several_calls():
    f = FakeFilter()
    s = PIISession(f, {"[NAME_1]": "Anna"})  # aus dem Chat-Kontext
    out = s.mask_tool_text("Bernd kauft")
    assert out == "[NAME_1_2] kauft"
    assert s.mapping() == {"[NAME_1]": "Anna", "[NAME_1_2]": "Bernd"}
    assert s.unmask("[NAME_1] und [NAME_1_2]") == "Anna und Bernd"


def test_without_a_filter_nothing_is_changed_or_called():
    s = PIISession(None)
    assert s.mask_tool_text("Anna kauft") == "Anna kauft" and s.mapping() == {}


def test_filter_errors_are_fail_closed():
    f = FakeFilter()
    f.fail = True
    with pytest.raises(PIIFilterUnavailable):
        PIISession(f).mask_tool_text("Anna kauft")


def test_text_without_entities_passes_unchanged():
    assert PIISession(FakeFilter()).mask_tool_text("3 Bestellungen") == "3 Bestellungen"


def test_live_demasker_sees_placeholders_that_appear_during_the_run():
    f = FakeFilter()
    s = PIISession(f, {"[NAME_1]": "Anna"})
    d = StreamDemasker(s)
    assert d.feed("Hallo [NAME_1]. ") == "Hallo Anna. "
    s.mask_tool_text("Bernd kauft")  # Werkzeugergebnis erzeugt [NAME_1_2]
    out = d.feed("Dann [NAME_1_") + d.feed("2] fertig") + d.flush()
    assert out == "Dann Bernd fertig"


def test_the_static_mapping_form_still_works():
    d = StreamDemasker({"[NAME_1]": "Anna"})
    assert d.feed("[NAM") + d.feed("E_1]!") + d.flush() == "Anna!"


def test_session_context_var_round_trip():
    s = PIISession(None)
    set_session(s)
    assert get_session() is s
    set_session(None)
    assert get_session() is None
