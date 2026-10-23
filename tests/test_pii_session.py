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


import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
from flask import Flask

from app.api import register_blueprints
from app.chat import compaction
from app.chat.pii_client import PIIFilterClient, PIIFilterUnavailable
from app.chat.pii_session import (
    StreamDemasker,
    join_texts,
    mask_conversation,
    split_texts,
    timeout_for,
)
from app.core.settings_service import SettingsService
from app.llm import connections as store
from app.llm import registry
from tests.test_access import add_user, client_for
from tests.test_sanitizer import FakeAgent

NAMES = ["Max Mustermann", "Anna Schmidt", "Karl Otto"]
EMAIL = re.compile(r"[\w.]+@[\w.]+\.\w+")


class EntityGuardLike(BaseHTTPRequestHandler):
    """Nummeriert je Aufruf eindeutig durch (NAME_1, NAME_2 ...) und antwortet wie EntityGuard mit sanitized_text."""

    mode = "good"
    calls: list = []
    flags: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text, reuse = body["text"], bool(body.get("reuse_placeholders"))
        EntityGuardLike.calls.append(text)
        EntityGuardLike.flags.append(reuse)
        if EntityGuardLike.mode == "500":
            return self._send(500, {})
        mapping, out, counters = {}, text, {"NAME": 0, "EMAIL": 0}
        for name in NAMES:
            while name in out:
                if reuse and name in mapping.values():
                    token = next(t for t, v in mapping.items() if v == name)
                else:
                    counters["NAME"] += 1
                    token = f"[NAME_{counters['NAME']}]"
                    mapping[token] = name
                out = out.replace(name, token, 1)
        for match in list(EMAIL.finditer(out)):
            counters["EMAIL"] += 1
            token = f"[EMAIL_{counters['EMAIL']}]"
            mapping[token] = match.group(0)
            out = out.replace(match.group(0), token, 1)
        if EntityGuardLike.mode == "drop_marker":
            out = out.replace("=====#1=====", "")
        if EntityGuardLike.mode == "mangle_marker":
            out = out.replace("=====#1=====", "[NAME_9]")
        self._send(200, {"sanitized_text": out, "mapping": mapping})

    def _send(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def guard():
    EntityGuardLike.mode, EntityGuardLike.calls, EntityGuardLike.flags = "good", [], []
    srv = HTTPServer(("127.0.0.1", 0), EntityGuardLike)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/api/v1/sanitize"
    srv.shutdown()


# --- mask_conversation ----------------------------------------------------------------------


def test_the_whole_context_is_masked_in_one_call_with_unique_placeholders(guard):
    texts = [
        "Max Mustermann fragt",
        "Antwort an Max Mustermann",
        "Anna Schmidt (anna@firma.de) schreibt",
    ]
    masked, mapping = mask_conversation(PIIFilterClient(guard), texts)
    assert len(EntityGuardLike.calls) == 1  # ein Aufruf fuer alles
    assert masked == [
        "[NAME_1] fragt",
        "Antwort an [NAME_1]",
        "[NAME_2] ([EMAIL_1]) schreibt",
    ]  # gleicher Wert = gleicher Platzhalter
    assert EntityGuardLike.flags == [
        True
    ]  # der Chat schickt den Hinweis reuse_placeholders mit
    client = PIIFilterClient(guard)
    assert [
        client.demask(t, mapping) for t in masked
    ] == texts  # Round-Trip ueber alle Nachrichten
    assert not any(n in " ".join(masked) for n in NAMES)


def test_texts_without_personal_data_pass_unchanged_and_whitespace_survives(guard):
    texts = ["Hallo", "  eingerückt\n\nzwei Absätze\n", "", "Ende"]
    masked, mapping = mask_conversation(PIIFilterClient(guard), texts)
    assert masked == texts and mapping == {}


def test_destroyed_or_missing_markers_fail_closed(guard):
    for mode in ("drop_marker", "mangle_marker"):
        EntityGuardLike.mode = mode
        with pytest.raises(PIIFilterUnavailable, match="Trennmarken"):
            mask_conversation(PIIFilterClient(guard), ["a", "b", "c"])


def test_a_user_text_containing_a_marker_fails_closed_instead_of_confusing_the_split(
    guard,
):
    with pytest.raises(PIIFilterUnavailable):
        mask_conversation(PIIFilterClient(guard), ["a", "böse =====#1===== Text", "c"])


def test_errors_and_unreachable_services_fail_closed(guard):
    EntityGuardLike.mode = "500"
    with pytest.raises(PIIFilterUnavailable, match="HTTP 500"):
        mask_conversation(PIIFilterClient(guard), ["a"])
    with pytest.raises(PIIFilterUnavailable):
        mask_conversation(PIIFilterClient("http://127.0.0.1:9/x"), ["a"])


def test_join_and_split_are_inverse_and_the_timeout_grows_with_the_text():
    texts = ["eins", "zwei\nzwei", "drei"]
    assert split_texts(join_texts(texts), 3) == texts
    assert (
        timeout_for(100) == pytest.approx(5.05)
        and timeout_for(20000) == pytest.approx(15)
        and timeout_for(10**9) == 60
    )


# --- StreamDemasker -------------------------------------------------------------------------


def stream(mapping, chunks):
    d = StreamDemasker(mapping)
    out = "".join(d.feed(c) for c in chunks) + d.flush()
    return out


@pytest.mark.parametrize(
    "chunks",
    [
        ["Hallo [NAME_1]!"],
        ["Hallo [NA", "ME_1]!"],
        ["Hallo [", "NAME", "_1", "]!"],
        ["Hallo ", "[NAME_1", "] und [NAME_2]", "."],
    ],
)
def test_placeholders_split_across_chunks_are_restored(chunks):
    mapping = {"[NAME_1]": "Max Mustermann", "[NAME_2]": "Anna Schmidt"}
    expected = (
        "".join(chunks)
        .replace("[NAME_1]", "Max Mustermann")
        .replace("[NAME_2]", "Anna Schmidt")
    )
    assert stream(mapping, chunks) == expected


def test_the_stream_never_emits_a_half_placeholder_early():
    d = StreamDemasker({"[NAME_1]": "Max"})
    assert d.feed("Hallo [NA") == "Hallo "  # Rest wird zurueckgehalten
    assert d.feed("ME_1]") == "Max"
    assert d.flush() == ""


def test_text_that_only_looks_like_a_placeholder_is_released_at_the_end():
    assert (
        stream({"[NAME_1]": "Max"}, ["Liste [1", "] fertig [NA"])
        == "Liste [1] fertig [NA"
    )


def test_longest_placeholder_wins_and_no_mapping_passes_through():
    mapping = {"[NAME_1]": "Max", "[NAME_10]": "Zoe"}
    assert stream(mapping, ["[NAME_10] und [NAME_1]"]) == "Zoe und Max"
    assert stream({}, ["a [NAME_1] b"]) == "a [NAME_1] b"
    assert (
        stream([{"placeholder": "[A]", "original": "Anna"}], ["x [A", "] y"])
        == "x Anna y"
    )


# --- Compaction mit Maskierung --------------------------------------------------------------


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content="Zusammenfassung: [NAME_1] plant eine Reise.")


def test_the_summarizer_sees_only_masked_text_and_the_stored_summary_is_plain(
    db, guard
):
    client = PIIFilterClient(guard)
    sid = db.create_session("t").id
    for i in range(10):
        db.save_message(sid, "user", f"Frage {i}: Max Mustermann ({'x' * 1500})")
        db.save_message(
            sid, "assistant", f"Antwort {i} an Max Mustermann ({'y' * 1500})"
        )
    summary, rows = db.get_history(sid)
    cfg = compaction.load_settings()
    budget = compaction.compute_budget({"num_ctx": 8000, "max_tokens": 2000}, cfg)
    llm = FakeLLM()
    _msgs, note = compaction.fit_history(
        rows,
        summary,
        "neu",
        100,
        budget,
        cfg,
        lambda: llm,
        lambda t, i: db.set_session_summary(sid, t, i),
        mask_many=lambda texts: mask_conversation(client, texts),
        unmask=lambda text, mapping: client.demask(text, mapping),
    )
    assert (
        note
        and llm.prompts
        and "Max Mustermann" not in llm.prompts[0]
        and "[NAME_" in llm.prompts[0]
    )
    stored, _ = db.get_history(sid)
    assert (
        stored == "Zusammenfassung: Max Mustermann plant eine Reise."
    )  # Klartext in der DB


# --- Route: ganzer Ablauf mit Fake-Agent ----------------------------------------------------


@pytest.fixture
def app(db):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    app.extensions["agent_tools"] = []
    register_blueprints(app)
    return app


def setup_chat(app, db, monkeypatch, pii_url, chunks):
    monkeypatch.setattr(registry, "fetch_remote_models", lambda c, force=False: ["m1"])
    monkeypatch.setattr("app.api.chat.resources_ready", lambda: True)
    conn = store.create_connection("C", "http://127.0.0.1:9/v1", "k")
    store.set_default_model(store.make_base_ref(conn["id"], "m1"))
    s = db.get_session()
    SettingsService(s).set_batch(
        {
            "PII_FILTER_URL": pii_url,
            "PII_FILTER_ENABLED": "true",
            "INGEST_INDEX_ENABLED": "false",
        }
    )
    s.close()
    agent = FakeAgent(chunks)
    monkeypatch.setattr("app.api.chat.get_agent", lambda *a, **k: agent)
    return agent


def sse_events(response):
    return [
        json.loads(line[6:])
        for line in response.get_data(as_text=True).split("\n\n")
        if line.startswith("data: ")
    ]


def test_history_and_question_reach_the_model_masked_and_the_stream_comes_back_in_plain_text(
    app, db, guard, monkeypatch
):
    agent = setup_chat(
        app,
        db,
        monkeypatch,
        guard,
        ["Gern, [NA", "ME_1] ", "und [NAME", "_2] melden sich."],
    )
    user = add_user(db, "ed", "editor")
    c = client_for(app, user)
    first = sse_events(
        c.post("/chat", json={"message": "Max Mustermann (max@firma.de) braucht Hilfe"})
    )
    sid = next(e["session_id"] for e in first if e.get("done"))
    assert (
        agent.inputs[0]["messages"][-1]["content"]
        == "[NAME_1] ([EMAIL_1]) braucht Hilfe"
    )

    EntityGuardLike.calls.clear()
    second = c.post("/chat", json={"message": "Und Anna Schmidt?", "session_id": sid})
    events = sse_events(second)
    sent = json.dumps(agent.inputs[1], ensure_ascii=False)
    assert (
        not any(n in sent for n in NAMES) and "max@firma.de" not in sent
    )  # Verlauf UND Frage maskiert
    assert len(EntityGuardLike.calls) == 1  # ein Aufruf fuer den ganzen Kontext
    roles = [m["role"] for m in agent.inputs[1]["messages"]]
    assert roles == ["user", "assistant", "user"]

    # Live-Antwort: Chunks "[NA"+"ME_1] " ... duerfen keine Platzhalter zeigen
    streamed = "".join(e.get("token", "") for e in events)
    assert "[NAME" not in streamed and "melden sich." in streamed
    # gespeichert ist Klartext (Nutzerfrage original, Antwort demaskiert)
    messages = [m.content for m in db.get_messages(sid)]
    assert messages[2] == "Und Anna Schmidt?" and "[NAME" not in messages[3]


def test_unreachable_filter_or_destroyed_markers_block_the_whole_chat(
    app, db, guard, monkeypatch
):
    agent = setup_chat(app, db, monkeypatch, guard, ["x"])
    c = client_for(app, add_user(db, "ed", "editor"))
    sid = next(
        e["session_id"]
        for e in sse_events(c.post("/chat", json={"message": "Hallo Max Mustermann"}))
        if e.get("done")
    )
    agent.inputs.clear()
    EntityGuardLike.mode = "drop_marker"
    body = c.post(
        "/chat", json={"message": "Noch eine Frage", "session_id": sid}
    ).get_data(as_text=True)
    assert (
        "Trennmarken" in body and "nicht gesendet" in body
    )  # die Ursache wird genannt, nicht pauschal "nicht erreichbar"
    assert agent.inputs == []  # bei Folgefragen ebenso: nichts geht ans Modell


def test_the_chat_error_names_a_short_reason_and_never_an_internal_address(
    app, db, monkeypatch
):
    import requests

    def refuse(*a, **k):
        raise requests.exceptions.ConnectionError(
            "HTTPConnectionPool(host='10.0.0.5', port=9500): Max retries exceeded"
        )

    monkeypatch.setattr("app.chat.pii_client.requests.post", refuse)
    agent = setup_chat(app, db, monkeypatch, "http://10.0.0.5:9500/sanitize", ["x"])
    body = (
        client_for(app, add_user(db, "ed", "editor"))
        .post("/chat", json={"message": "Hallo"})
        .get_data(as_text=True)
    )
    assert (
        "PII-Filter ist nicht erreichbar" in body
        and "10.0.0.5" not in body
        and "HTTPConnectionPool" not in body
    )
    assert agent.inputs == []


def test_a_plain_year_is_not_masked_but_everything_else_stays():
    from app.chat.pii_session import keep_years, mask_conversation
    from tests.test_pii_tools import FakeFilter

    class YearFilter(FakeFilter):
        NAMES = ("Anna", "2026", "1984")

    texts, mapping = mask_conversation(
        YearFilter(), ["Anna hatte 2026 Patienten", "geboren 1984"]
    )
    assert texts == ["[NAME_1] hatte 2026 Patienten", "geboren 1984"] and dict(
        mapping
    ) == {"[NAME_1]": "Anna"}
    # ein Jahr als Teil eines Datums ist keine reine Jahreszahl: das entscheidet weiter der Filter
    t, m = keep_years(["am [DATUM_1]"], {"[DATUM_1]": "2026-03-20"})
    assert t == ["am [DATUM_1]"] and dict(m) == {"[DATUM_1]": "2026-03-20"}
    t, m = keep_years(["nichts"], [])
    assert t == ["nichts"]


# --- Zeitlimit, Wiederholung, Stuecke, Meldungen --------------------------------------------------


def test_the_timeout_grows_with_the_text_and_never_falls_below_the_base():
    from app.chat.pii_session import timeout_for

    assert timeout_for(100, 10) == pytest.approx(10.05) and timeout_for(
        12000, 10
    ) == pytest.approx(16)
    assert (
        timeout_for(10**9, 10) == 60 and timeout_for(100, 90) == 90
    )  # die Basis ist die Untergrenze


def test_tool_texts_get_the_scaled_timeout_too(monkeypatch):
    from app.chat.pii_client import PIIFilterClient, PIIFilterResult, mask_or_fail

    seen = []
    client = PIIFilterClient("http://x/s", timeout=10)
    monkeypatch.setattr(
        client,
        "filter_text",
        lambda text, timeout=None, extra=None: (
            seen.append(timeout) or PIIFilterResult(original=text, filtered=text)
        ),
    )
    mask_or_fail(client, "k" * 100)
    mask_or_fail(client, "g" * 12000)
    assert seen == [
        pytest.approx(10.05),
        pytest.approx(16),
    ]  # vorher immer die festen 5 s


class _Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body, self.text = (
            status,
            body or {"filtered": "x", "mapping": {}},
            "",
        )

    def json(self):
        return self._body


def _client_with(monkeypatch, outcomes):
    import requests

    from app.chat.pii_client import PIIFilterClient

    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append(timeout)
        outcome = outcomes[min(len(calls), len(outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(requests, "post", post)
    return PIIFilterClient("http://x/s", timeout=10), calls


def test_one_transient_failure_is_retried_with_double_the_limit(monkeypatch):
    import requests

    for first in (
        requests.exceptions.Timeout(),
        requests.exceptions.ConnectionError("x"),
        _Resp(503),
    ):
        client, calls = _client_with(monkeypatch, [first, _Resp(200)])
        result = client.filter_text("Hallo")
        assert not result.error and calls == [10, 20]


def test_a_rejected_key_or_a_second_failure_is_final(monkeypatch):
    import requests

    client, calls = _client_with(monkeypatch, [_Resp(401)])
    assert (
        client.filter_text("Hallo").error == "HTTP 401" and len(calls) == 1
    )  # 4xx: kein Wiederholen
    client, calls = _client_with(monkeypatch, [requests.exceptions.Timeout()])
    result = client.filter_text("Hallo")
    assert result.error == "Timeout (20s)" and calls == [10, 20]
    client, calls = _client_with(monkeypatch, [_Resp(500)])
    assert client.filter_text("Hallo").error == "HTTP 500" and len(calls) == 2


def test_the_messages_name_the_real_cause():
    from app.chat.pii_session import unavailable_message

    assert "nicht rechtzeitig" in unavailable_message(
        "Timeout (20s)"
    ) and "nach 20s" in unavailable_message("Timeout (20s)")
    assert "lehnt die Anmeldung ab" in unavailable_message(
        "HTTP 401"
    ) and "API-Key" in unavailable_message("HTTP 403")
    assert "Fehler geantwortet" in unavailable_message("HTTP 500: boom")
    assert "Trennmarken" in unavailable_message(
        "der Filter hat die Trennmarken des Verlaufs verändert"
    )
    assert (
        unavailable_message("HTTPConnectionPool(host='10.0.0.5')")
        == "Der PII-Filter ist nicht erreichbar"
    )  # nie Adressen


def test_split_text_keeps_every_character_and_prefers_paragraphs():
    from app.chat.pii_session import split_text

    text = ("Absatz eins ist lang genug. " * 20 + "\n\n") * 5
    pieces = split_text(text, 700)
    assert "".join(pieces) == text and all(len(p) <= 700 for p in pieces)
    assert all(p.endswith("\n\n") for p in pieces[:-1] if "\n\n" in p)
    assert split_text("kurz", 700) == ["kurz"]
    assert (
        "".join(split_text("x" * 2500, 1000)) == "x" * 2500
    )  # ohne Trennstelle hart geschnitten


class _Cap:
    """Filter mit Hoechstlaenge: Aufrufe darueber scheitern (der echte Dienst stuerzte bei 50 000 Zeichen ab)."""

    NAMES = ("Anna", "Bernd")

    def __init__(self, limit):
        self.max_chars, self.sizes = limit, []

    def filter_text(self, text, timeout=None, extra=None):
        from app.chat.pii_client import PIIFilterResult

        self.sizes.append(len(text))
        if len(text) > self.max_chars + 500:
            return PIIFilterResult(original=text, filtered=text, error="HTTP 500")
        mapping, out = {}, text
        for name in self.NAMES:
            if name in out:
                ph = f"[NAME_{len(mapping) + 1}]"
                mapping[ph] = name
                out = out.replace(name, ph)
        return PIIFilterResult(original=text, filtered=out, mapping=mapping)


def test_long_conversations_go_to_the_filter_in_pieces_with_consistent_placeholders():
    from app.chat.pii_session import PIISession, mask_conversation

    filler = "Das ist ein Satz ohne Namen. " * 40
    texts = [
        f"Frage {i}: {'Anna' if i % 2 else 'Bernd'} schrieb. {filler}"
        for i in range(12)
    ]  # ~15 000 Zeichen
    client = _Cap(4000)
    masked, mapping = mask_conversation(client, texts)
    assert (
        len(client.sizes) > 3 and max(client.sizes) <= 4500
    )  # mehrere Aufrufe, keiner ueber der Hoechstlaenge
    assert (
        len(masked) == 12
        and "Anna" not in "".join(masked)
        and "Bernd" not in "".join(masked)
    )
    session = PIISession(None, mapping)
    assert [session.unmask(m) for m in masked] == texts  # verlustfrei zurueck
    # gleicher Name = gleicher Platzhalter ueber alle Stuecke, zwei Namen = zwei Platzhalter
    assert sorted(dict(mapping).values()) == ["Anna", "Bernd"]


def test_one_huge_tool_result_is_masked_piece_by_piece(db):
    from app.chat.pii_session import PIISession

    client = _Cap(3000)
    session = PIISession(client)
    text = "".join(
        f"Zeile {i}: Anna und Bernd, {'x' * 80}\n" for i in range(120)
    )  # ~12 000 Zeichen
    out = session.mask_tool_text(text)
    assert max(client.sizes) <= 3500 and len(client.sizes) >= 4
    assert "Anna" not in out and "Bernd" not in out and session.unmask(out) == text


def test_admin_sets_timeout_and_max_length_and_the_client_uses_them(app, db):
    from app.agent.tools import web_access
    from tests.test_web_and_vision import ADMIN

    admin = client_for(app, ADMIN)
    assert (
        admin.get("/api/admin/tools").json["pii"]["timeout_s"] == 10
        and admin.get("/api/admin/tools").json["pii"]["max_chars"] == 12000
    )
    res = admin.put(
        "/api/admin/tools",
        json={
            "pii": {
                "url": "http://example.org:9500/s",
                "timeout_s": 25,
                "max_chars": 8000,
            }
        },
    )
    assert (
        res.status_code == 200
        and res.json["pii"]["timeout_s"] == 25
        and res.json["pii"]["max_chars"] == 8000
    )
    client = web_access.pii_client()
    assert client.timeout == 25 and client.max_chars == 8000
    for bad in (
        {"timeout_s": 1},
        {"timeout_s": "x"},
        {"max_chars": 500},
        {"max_chars": 10**7},
    ):
        assert (
            admin.put(
                "/api/admin/tools",
                json={"pii": {"url": "http://example.org:9500/s", **bad}},
            ).status_code
            == 400
        )


def test_migration_0035_seeds_the_limits_and_runs_up_down_up(tmp_path):
    import importlib.util
    import os
    import subprocess
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "m0035", Path("alembic/versions/0035_pii_filter_limits.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert {k: v for k, v, _d in mod.ROWS} == {
        "PII_FILTER_TIMEOUT_S": "10",
        "PII_FILTER_MAX_CHARS": "12000",
    }
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{tmp_path / 'm.db'}"}
    run = lambda *a: subprocess.run(
        [sys.executable, "-m", "alembic", *a], env=env, check=True, capture_output=True
    )
    run("upgrade", "head")
    run("downgrade", "0034")
    run("upgrade", "head")
