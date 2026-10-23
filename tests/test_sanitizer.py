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
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from flask import Flask

from app.agent.tools import web_access
from app.api.admin_routes import admin_bp
from app.api.api_routes import api_bp
from app.api.routes import bp as main_bp
from app.chat.pii_client import PIIFilterClient, mapping_pairs, parse_response
from app.core.settings_service import SettingsService
from app.llm import connections as store
from app.llm import registry
from tests.test_access import add_user, client_for

ADMIN = {"id": 1, "role": "admin", "username": "admin"}


# --- Vertrag: Antwort lesen -----------------------------------------------------------------


def test_parse_response_reads_the_documented_format():
    data = {"filtered": "Hallo [NAME_1]", "mapping": {"[NAME_1]": "Max"}}
    assert parse_response(data, "Hallo Max") == (
        "Hallo Max",
        "Hallo [NAME_1]",
        {"[NAME_1]": "Max"},
    )


def test_parse_response_accepts_the_compatible_alternatives():
    data = {
        "sanitized_text": "Hi [P1]",
        "entities": [{"placeholder": "[P1]", "original": "Anna"}],
    }
    _o, filtered, mapping = parse_response(data, "Hi Anna")
    assert filtered == "Hi [P1]" and mapping_pairs(mapping) == [("[P1]", "Anna")]


def test_an_empty_filtered_text_is_a_valid_answer_and_never_falls_back_to_the_original():
    _o, filtered, _m = parse_response({"filtered": "", "mapping": {}}, "Max Mustermann")
    assert filtered == ""  # reiner PII-Text: nichts darf zurueckfallen


def test_parse_response_without_mapping_and_with_junk():
    assert parse_response({"filtered": "x"}, "y")[2] == {}
    assert parse_response("nicht json-objekt", "y") == ("y", "y", {})
    assert (
        mapping_pairs(None) == []
        and mapping_pairs([{"placeholder": "", "original": "a"}]) == []
    )


def test_demask_uses_both_mapping_shapes():
    c = PIIFilterClient("http://x")
    assert c.demask("Hi [A]", {"[A]": "Anna"}) == "Hi Anna"
    assert c.demask("Hi [A]", [{"placeholder": "[A]", "original": "Anna"}]) == "Hi Anna"


# --- Fake-Sanitizer ------------------------------------------------------------------------


class Sanitizer(BaseHTTPRequestHandler):
    behavior = "good"
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = body["text"]
        Sanitizer.seen.append({"text": text, "auth": self.headers.get("Authorization")})
        mode = Sanitizer.behavior
        if mode == "401":
            return self._send(401, {"error": "nope"})
        if mode == "500":
            return self._send(500, {"error": "kaputt"})
        if mode == "html":
            raw = b"<html>"
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if mode == "unchanged":
            return self._send(200, {"filtered": text, "mapping": {}})
        masked = text.replace("Max Mustermann", "[NAME_1]").replace(
            "max.mustermann@example.com", "[EMAIL_1]"
        )
        if mode == "no_mapping":
            return self._send(200, {"filtered": masked})
        if mode == "bad_mapping":
            return self._send(
                200, {"filtered": masked, "mapping": {"[NAME_1]": "Falsch"}}
            )
        self._send(
            200,
            {
                "filtered": masked,
                "mapping": {
                    "[NAME_1]": "Max Mustermann",
                    "[EMAIL_1]": "max.mustermann@example.com",
                },
            },
        )

    def _send(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def sanitizer():
    Sanitizer.behavior, Sanitizer.seen = "good", []
    srv = HTTPServer(("127.0.0.1", 0), Sanitizer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/sanitize"
    srv.shutdown()


class SearX(BaseHTTPRequestHandler):
    queries: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        SearX.queries.append(parse_qs(urlparse(self.path).query).get("q", [""])[0])
        raw = json.dumps(
            {"results": [{"title": "T", "url": "http://e.test", "content": "C"}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def searx():
    SearX.queries = []
    srv = HTTPServer(("127.0.0.1", 0), SearX)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture
def app(db):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    for bp in (main_bp, api_bp, admin_bp):
        app.register_blueprint(bp)
    return app


def set_settings(db, **values):
    s = db.get_session()
    SettingsService(s).set_batch(values)
    s.close()


# --- Test-Knopf ----------------------------------------------------------------------------


def run_test(app, url, key=""):
    return client_for(app, ADMIN).post(
        "/api/admin/tools/test",
        json={"kind": "pii", "values": {"url": url, "api_key": key}},
    )


def test_the_test_button_verifies_masking_mapping_and_demasking(app, db, sanitizer):
    r = run_test(app, sanitizer, "k1")
    assert r.status_code == 200 and "vertragsgemäß" in r.json["message"]
    assert (
        "Text wird maskiert" in r.json["details"]
        and "Demaskierung stimmt" in r.json["details"]
    )
    assert (
        Sanitizer.seen[0]["auth"] == "Bearer k1"
        and "Max Mustermann" in Sanitizer.seen[0]["text"]
    )


@pytest.mark.parametrize(
    "behavior,status,needle",
    [
        ("unchanged", 502, "nichts erkannt"),
        ("401", 502, "API-Key"),
        ("500", 502, "HTTP 500"),
        ("html", 502, "kein JSON"),
    ],
)
def test_the_test_button_names_the_exact_problem(
    app, db, sanitizer, behavior, status, needle
):
    Sanitizer.behavior = behavior
    r = run_test(app, sanitizer)
    assert r.status_code == status and needle in (r.json.get("error") or "")


def test_the_test_button_warns_about_missing_mapping_and_wrong_demasking(
    app, db, sanitizer
):
    Sanitizer.behavior = "no_mapping"
    r = run_test(app, sanitizer)
    assert (
        r.status_code == 200
        and "mit Hinweisen" in r.json["message"]
        and "Kein Mapping" in r.json["details"]
    )
    Sanitizer.behavior = "bad_mapping"
    r = run_test(app, sanitizer)
    assert (
        r.status_code == 200
        and "Demaskierung ergibt nicht den Originaltext" in r.json["details"]
    )


def test_the_test_button_refuses_link_local_targets(app, db):
    assert run_test(app, "http://169.254.169.254/x").status_code == 400


# --- Websuche mit und ohne Sanitizer -------------------------------------------------------


def search(searx_url, query, pii_url=None, behavior=None):
    tool = web_access.build_web_search_tool()
    return tool.invoke({"query": query})


def test_web_search_availability_depends_on_masking_or_the_explicit_release(
    db, searx, sanitizer
):
    assert web_access.web_search_available() is False  # nichts eingerichtet
    set_settings(db, SEARCH_SEARXNG_URL=searx)
    assert (
        web_access.web_search_available() is False
    )  # ohne Filter und ohne Freigabe: wie bisher gesperrt
    assert web_access.build_web_search_tool() is None
    set_settings(db, PII_FILTER_URL=sanitizer, PII_FILTER_ENABLED="true")
    assert web_access.web_search_available() and web_access.web_search_masked()
    set_settings(db, PII_FILTER_ENABLED="false")
    assert web_access.web_search_available() is False
    set_settings(db, SEARCH_ALLOW_UNMASKED="true")
    assert (
        web_access.web_search_available() is True
        and web_access.web_search_masked() is False
    )


def test_web_search_without_sanitizer_sends_the_query_unchanged(db, searx):
    set_settings(db, SEARCH_SEARXNG_URL=searx, SEARCH_ALLOW_UNMASKED="true")
    out = search(searx, "Wetter in Lissabon")
    assert SearX.queries == ["Wetter in Lissabon"] and "T" in out


def test_web_search_with_sanitizer_removes_personal_data(db, searx, sanitizer):
    set_settings(db, SEARCH_SEARXNG_URL=searx, PII_FILTER_URL=sanitizer)
    search(searx, "Kontakt Max Mustermann Hotel Lissabon")
    assert SearX.queries == ["Kontakt Hotel Lissabon"]


def test_web_search_uses_the_filtered_text_when_the_service_has_no_mapping(
    db, searx, sanitizer
):
    Sanitizer.behavior = "no_mapping"
    set_settings(db, SEARCH_SEARXNG_URL=searx, PII_FILTER_URL=sanitizer)
    search(searx, "Hotel für Max Mustermann")
    assert SearX.queries == ["Hotel für [NAME_1]"]  # nie das Original


@pytest.mark.parametrize("behavior", ["500", "401", "html"])
def test_web_search_fails_closed_when_the_sanitizer_is_configured_but_fails(
    db, searx, sanitizer, behavior
):
    Sanitizer.behavior = behavior
    set_settings(db, SEARCH_SEARXNG_URL=searx, PII_FILTER_URL=sanitizer)
    out = search(searx, "Hotel für Max Mustermann")
    assert "abgebrochen" in out and SearX.queries == []  # nichts ging an SearXNG


def test_web_search_sends_nothing_when_the_query_was_only_personal_data(
    db, searx, sanitizer
):
    set_settings(db, SEARCH_SEARXNG_URL=searx, PII_FILTER_URL=sanitizer)
    out = search(searx, "Max Mustermann")
    assert "nur aus personenbezogenen" in out and SearX.queries == []


# --- Modelle und Oberflaeche ---------------------------------------------------------------


def test_model_config_allows_web_search_only_when_available(
    app, db, searx, monkeypatch
):
    monkeypatch.setattr(registry, "fetch_remote_models", lambda c, force=False: ["m1"])
    conn = store.create_connection("C", "http://127.0.0.1:9/v1", "k")
    ref = store.make_base_ref(conn["id"], "m1")
    admin = client_for(app, ADMIN)
    set_settings(db, SEARCH_SEARXNG_URL=searx)
    blocked = admin.put("/api/models/config", json={"ref": ref, "web_search": True})
    assert (
        blocked.status_code == 400
        and "Auch ohne PII-Filter erlauben" in blocked.json["error"]
    )
    models = admin.get("/api/models").json
    assert (
        models["web_search_available"] is False and models["web_search_masked"] is False
    )
    admin.put(
        "/api/admin/tools", json={"search": {"url": searx, "allow_unmasked": True}}
    )
    ok = admin.put("/api/models/config", json={"ref": ref, "web_search": True})
    assert ok.status_code == 200 and ok.json["web_search"] is True
    models = admin.get("/api/models").json
    assert (
        models["web_search_available"] is True and models["web_search_masked"] is False
    )


def test_tools_api_exposes_and_saves_the_release(app, db, searx):
    admin = client_for(app, ADMIN)
    assert admin.get("/api/admin/tools").json["search"]["allow_unmasked"] is False
    got = admin.put(
        "/api/admin/tools", json={"search": {"url": searx, "allow_unmasked": True}}
    ).json
    assert (
        got["search"]["allow_unmasked"] is True
        and got["web_search_available"] is True
        and got["web_search_masked"] is False
    )
    ed = client_for(app, add_user(db, "ed", "editor"))
    assert (
        ed.put(
            "/api/admin/tools", json={"search": {"allow_unmasked": True}}
        ).status_code
        == 403
    )


# --- Chat: fail-closed --------------------------------------------------------------------


def test_mask_or_fail_masks_and_raises_on_errors(sanitizer):
    from app.chat.pii_client import PIIFilterUnavailable, mask_or_fail

    client = PIIFilterClient(sanitizer)
    text, mapping = mask_or_fail(client, "Hallo Max Mustermann")
    assert text == "Hallo [NAME_1]" and mapping == {
        "[NAME_1]": "Max Mustermann",
        "[EMAIL_1]": "max.mustermann@example.com",
    }
    assert mask_or_fail(client, "Hallo Welt") == (
        "Hallo Welt",
        [],
    )  # nichts erkannt: unveraendert
    for behavior in ("500", "401"):
        Sanitizer.behavior = behavior
        with pytest.raises(PIIFilterUnavailable):
            mask_or_fail(client, "Hallo Max Mustermann")
    with pytest.raises(PIIFilterUnavailable):
        mask_or_fail(
            PIIFilterClient("http://127.0.0.1:9/x"), "Hallo"
        )  # nicht erreichbar


class FakeAgent:
    """Statt des echten Agenten: merkt sich, was das Modell bekaeme, und streamt feste Chunks."""

    def __init__(self, chunks=("Antwort",)):
        self.chunks = list(chunks)
        self.inputs = []

    async def astream_events(self, input_data, version="v2"):
        self.inputs.append(input_data)
        for text in self.chunks:
            yield {
                "event": "on_chat_model_stream",
                "data": {
                    "chunk": type("C", (), {"content": text, "additional_kwargs": {}})()
                },
            }


def chat_setup(app, db, monkeypatch, pii_url, chunks=("Antwort",)):
    monkeypatch.setattr(registry, "fetch_remote_models", lambda c, force=False: ["m1"])
    monkeypatch.setattr("app.api.routes.resources_ready", lambda: True)
    conn = store.create_connection("C", "http://127.0.0.1:9/v1", "k")
    store.set_default_model(store.make_base_ref(conn["id"], "m1"))
    set_settings(
        db,
        PII_FILTER_URL=pii_url,
        PII_FILTER_ENABLED="true",
        INGEST_INDEX_ENABLED="false",
    )
    app.extensions["agent_tools"] = []
    agent = FakeAgent(chunks)
    monkeypatch.setattr("app.api.routes.get_agent", lambda *a, **k: agent)
    return agent


def test_chat_shows_an_error_and_sends_nothing_when_the_filter_is_unreachable(
    app, db, monkeypatch
):
    agent = chat_setup(app, db, monkeypatch, "http://127.0.0.1:9/sanitize")
    r = client_for(app, add_user(db, "ed", "editor")).post(
        "/chat", json={"message": "Schreibe an Max Mustermann"}
    )
    body = r.get_data(as_text=True)
    assert (
        r.mimetype == "text/event-stream"
        and "PII-Filter ist nicht erreichbar" in body
        and "nicht gesendet" in body
    )
    assert agent.inputs == []  # nichts ging an das Modell
    s = db.get_session()
    from app.core.models import Message

    assert s.query(Message).count() == 0  # die Nachricht wurde nicht gespeichert
    s.close()


def test_chat_also_fails_closed_when_the_filter_answers_with_an_error(
    app, db, monkeypatch, sanitizer
):
    Sanitizer.behavior = "500"
    agent = chat_setup(app, db, monkeypatch, sanitizer)
    r = client_for(app, add_user(db, "ed", "editor")).post(
        "/chat", json={"message": "Schreibe an Max Mustermann"}
    )
    assert "HTTP 500" in r.get_data(as_text=True) and agent.inputs == []


def test_chat_without_a_configured_filter_is_not_blocked_by_the_filter_logic(
    app, db, monkeypatch
):
    agent = chat_setup(app, db, monkeypatch, "")
    client_for(app, add_user(db, "ed", "editor")).post(
        "/chat", json={"message": "Hallo Max Mustermann"}
    ).get_data()
    assert (
        agent.inputs[0]["messages"][-1]["content"] == "Hallo Max Mustermann"
    )  # ohne Filter unveraendert
