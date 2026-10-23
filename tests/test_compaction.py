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


from types import SimpleNamespace

import pytest
from flask import Flask
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import context_guard
from app.chat import compaction
from app.web.admin_routes import admin_bp
from app.web.api_routes import api_bp
from app.web.routes import bp as main_bp
from tests.test_access import add_user, client_for

ADMIN = {"id": 1, "role": "admin", "username": "admin"}


@pytest.fixture
def app(db):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    for bp in (main_bp, api_bp, admin_bp):
        app.register_blueprint(bp)
    return app


class FakeLLM:
    def __init__(self, text="ZUSAMMENFASSUNG"):
        self.text = text
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.text)

    def with_retry(self, **kwargs):
        return self


def rows(*pairs, start=1):
    return [
        SimpleNamespace(id=start + i, role=r, content=c)
        for i, (r, c) in enumerate(pairs)
    ]


def chat_pairs(n, size=1500):
    out = []
    for i in range(n):
        out.append(("user", f"Frage {i} " + "x" * size))
        out.append(("assistant", f"Antwort {i} " + "y" * size))
    return out


def chat(n, size=1500):
    return rows(*chat_pairs(n, size))


def test_budget_is_context_minus_output_with_five_percent_headroom(db):
    cfg = compaction.load_settings()
    b = compaction.compute_budget({"num_ctx": 8000, "max_tokens": 2000}, cfg)
    assert (b.window, b.output, b.usable, b.trigger) == (8000, 2000, 6000, 5700)
    assert b.keep == 1800  # 30 % des Budgets bleiben wortwoertlich


def test_without_max_tokens_a_share_of_the_context_is_reserved(db):
    b = compaction.compute_budget(
        {"num_ctx": 10000, "max_tokens": None}, compaction.load_settings()
    )
    assert b.output == 2000 and b.usable == 8000


def test_compaction_is_off_when_disabled_or_the_window_is_too_small(db):
    cfg = compaction.load_settings()
    assert compaction.compute_budget({"num_ctx": 1500, "max_tokens": 1000}, cfg) is None
    compaction.save_settings({"enabled": False})
    assert (
        compaction.compute_budget(
            {"num_ctx": 8000, "max_tokens": 2000}, compaction.load_settings()
        )
        is None
    )


def test_history_below_the_trigger_is_left_alone(db):
    b = compaction.compute_budget(
        {"num_ctx": 8000, "max_tokens": 2000}, compaction.load_settings()
    )
    llm = FakeLLM()
    msgs, note = compaction.fit_history(
        chat(2),
        None,
        "neue Frage",
        100,
        b,
        compaction.load_settings(),
        lambda: llm,
        lambda *a: None,
    )
    assert note is None and llm.prompts == [] and len(msgs) == 4


def test_history_above_the_trigger_is_summarized_once_and_the_recent_part_stays(db):
    cfg = compaction.load_settings()
    b = compaction.compute_budget({"num_ctx": 8000, "max_tokens": 2000}, cfg)
    history = chat(
        10
    )  # ~10000 geschaetzte Token -> deutlich ueber dem Trigger von 5700
    llm, saved = FakeLLM(), []
    msgs, note = compaction.fit_history(
        history,
        None,
        "neue Frage",
        100,
        b,
        cfg,
        lambda: llm,
        lambda t, i: saved.append((t, i)),
    )
    assert (
        len(llm.prompts) == 1
        and note["summarized"] > 0
        and note["after"] < note["before"]
    )
    assert saved and saved[0][0] == "ZUSAMMENFASSUNG"
    assert (
        msgs[0]["content"].startswith(compaction.SUMMARY_INTRO)
        and "ZUSAMMENFASSUNG" in msgs[0]["content"]
    )
    assert (
        msgs[1]["role"] == "assistant" and msgs[2]["role"] == "user"
    )  # Wechsel bleibt erhalten
    assert msgs[-1]["content"].startswith(
        "Antwort 9"
    )  # juengste Nachricht wortwoertlich
    assert not any(
        m["content"].startswith("Frage 0") for m in msgs[2:]
    )  # Aelteres steckt in der Zusammenfassung
    assert "Frage 0" in llm.prompts[0]


def test_summary_is_stored_and_reused_without_another_llm_call(db):
    sid = db.create_session("t").id
    for role, content in chat_pairs(10):
        db.save_message(sid, role, content)
    summary, all_rows = db.get_history(sid)
    assert summary is None and len(all_rows) == 20

    cfg = compaction.load_settings()
    b = compaction.compute_budget({"num_ctx": 8000, "max_tokens": 2000}, cfg)
    llm = FakeLLM()
    _msgs, note = compaction.fit_history(
        all_rows,
        summary,
        "neu",
        100,
        b,
        cfg,
        lambda: llm,
        lambda t, i: db.set_session_summary(sid, t, i),
    )
    assert note

    summary2, rest = db.get_history(sid)
    assert summary2 == "ZUSAMMENFASSUNG" and 0 < len(rest) < 20
    assert rest[0].role == "user"
    msgs2, note2 = compaction.fit_history(
        rest, summary2, "neu", 100, b, cfg, lambda: llm, lambda *a: None
    )
    assert note2 is None and len(llm.prompts) == 1  # keine zweite Zusammenfassung
    assert msgs2[0]["content"].startswith(compaction.SUMMARY_INTRO)


def test_history_contains_the_newest_messages_not_the_first_ten(db):
    sid = db.create_session("t").id
    for i in range(30):
        db.save_message(sid, "user" if i % 2 == 0 else "assistant", f"Nachricht {i}")
    _, got = db.get_history(sid)
    assert (
        got[0].content == "Nachricht 0"
        and got[-1].content == "Nachricht 29"
        and len(got) == 30
    )


def test_display_parts_do_not_reach_the_model(db):
    r = rows(
        ("user", "Frage"),
        ("assistant", '<think>geheim</think>\nAntwort\n\n<sources>[{"a":1}]</sources>'),
    )
    assert [m["content"] for m in compaction.history_messages(r, None)] == [
        "Frage",
        "Antwort",
    ]


def test_one_huge_message_is_clipped_when_nothing_older_can_be_summarized(db):
    cfg = compaction.load_settings()
    b = compaction.compute_budget({"num_ctx": 8000, "max_tokens": 2000}, cfg)
    huge = rows(("user", "z" * 60000))
    msgs, note = compaction.fit_history(
        huge, None, "neu", 100, b, cfg, lambda: FakeLLM(), lambda *a: None
    )
    assert note["clipped"] and compaction.estimate_tokens(msgs[0]["content"]) < b.usable


def test_prompt_template_and_langchain_format(db):
    text = compaction.render_prompt("A {x} {messages} B", "INHALT")
    assert text == "A {x} INHALT B"
    assert (
        compaction.prompt_for_langchain("A {x} {messages}").format(messages="M")
        == "A {x} M"
    )


# --- Ebene 2: innerhalb eines Agent-Laufs ---------------------------------------------------


def test_tool_results_larger_than_a_share_of_the_budget_are_clipped():
    mw = context_guard.ToolResultClipMiddleware(max_tokens=500)
    big = ToolMessage(
        content="d" * 20000, tool_call_id="1", name="document_search_tool"
    )
    small = ToolMessage(content="kurz", tool_call_id="2", name="document_search_tool")
    assert (
        compaction.estimate_tokens(mw.wrap_tool_call(None, lambda r: big).content) < 600
    )
    assert "gekürzt" in mw.wrap_tool_call(None, lambda r: big).content
    assert mw.wrap_tool_call(None, lambda r: small).content == "kurz"


def test_async_tool_results_are_clipped_too():
    import asyncio

    mw = context_guard.ToolResultClipMiddleware(max_tokens=100)
    big = ToolMessage(content="d" * 5000, tool_call_id="1", name="t")

    async def handler(_):
        return big

    assert "gekürzt" in asyncio.run(mw.awrap_tool_call(None, handler)).content


def test_token_counter_counts_text_and_tool_calls():
    msgs = [
        HumanMessage(content="a" * 300),
        AIMessage(
            content="b" * 30,
            tool_calls=[{"name": "t", "args": {"q": "c" * 90}, "id": "1"}],
        ),
    ]
    assert context_guard.count_message_tokens(msgs) >= 100 + 10 + 30


def test_middleware_is_built_from_the_budget(db):
    b = compaction.compute_budget(
        {"num_ctx": 8000, "max_tokens": 2000}, compaction.load_settings()
    )
    mw = context_guard.build_middleware(
        FakeLLM(), b, compaction.DEFAULT_PROMPT, overhead_tokens=700
    )
    assert len(mw) == 2 and mw[0].max_tokens == 2400


# --- Einstellungen --------------------------------------------------------------------------


def test_settings_api_is_admin_only_and_validates(app, db):
    ed = add_user(db, "ed", "editor")
    assert client_for(app, ed).get("/api/admin/compaction").status_code == 403
    assert client_for(app, ed).put("/api/admin/compaction", json={}).status_code == 403
    admin = client_for(app, ADMIN)
    got = admin.get("/api/admin/compaction").json
    assert (
        got["enabled"] is True
        and got["headroom_percent"] == 5
        and "{messages}" in got["prompt"]
    )
    assert (
        admin.put("/api/admin/compaction", json={"headroom_percent": 0}).status_code
        == 400
    )
    assert (
        admin.put("/api/admin/compaction", json={"keep_percent": "abc"}).status_code
        == 400
    )
    assert (
        admin.put(
            "/api/admin/compaction", json={"prompt": "ohne Platzhalter"}
        ).status_code
        == 400
    )
    assert admin.put("/api/admin/compaction", json={"prompt": "  "}).status_code == 400
    ok = admin.put(
        "/api/admin/compaction",
        json={"headroom_percent": 10, "enabled": False, "prompt": "Kurz: {messages}"},
    )
    assert (
        ok.status_code == 200
        and ok.json["headroom_percent"] == 10
        and ok.json["enabled"] is False
    )
    assert (
        ok.json["default_prompt"] == compaction.DEFAULT_PROMPT
    )  # Zuruecksetzen moeglich
    reset = admin.put(
        "/api/admin/compaction", json={"prompt": ok.json["default_prompt"]}
    )
    assert reset.json["prompt"] == compaction.DEFAULT_PROMPT


def test_default_instructions_api_is_admin_only_and_resettable(app, db):
    from app.agent import behavior

    ed = add_user(db, "ed2", "editor")
    assert client_for(app, ed).get("/api/admin/agent-behavior").status_code == 403
    assert (
        client_for(app, ed)
        .put("/api/admin/agent-behavior", json={"prompt": "x"})
        .status_code
        == 403
    )
    admin = client_for(app, ADMIN)
    got = admin.get("/api/admin/agent-behavior").json
    assert got["prompt"] == got["default_prompt"] == behavior.DEFAULT_BEHAVIOR
    assert (
        admin.put("/api/admin/agent-behavior", json={"prompt": "x" * 9000}).status_code
        == 400
    )
    ok = admin.put(
        "/api/admin/agent-behavior", json={"prompt": "Sprich wie die Marke."}
    )
    assert (
        ok.status_code == 200
        and ok.json["prompt"] == "Sprich wie die Marke."
        and ok.json["default_prompt"] == behavior.DEFAULT_BEHAVIOR
    )
    assert (
        admin.put(
            "/api/admin/agent-behavior", json={"prompt": ok.json["default_prompt"]}
        ).json["prompt"]
        == behavior.DEFAULT_BEHAVIOR
    )


def test_agent_is_rebuilt_when_the_budget_changes(db, monkeypatch):
    from app.llm import registry

    monkeypatch.setattr(registry, "build_chat_llm", lambda *a, **k: object())

    class FakeAgent(SimpleNamespace):
        @staticmethod
        def compose_prompt(extra):
            return "System"

    monkeypatch.setattr(registry, "DocumentAgent", lambda **k: FakeAgent(**k))
    monkeypatch.setattr(
        registry.DocumentAgent,
        "compose_prompt",
        FakeAgent.compose_prompt,
        raising=False,
    )
    resolved = registry.ResolvedModel(
        ref="r",
        display_name="r",
        connection_id=1,
        base_url="http://127.0.0.1:9/v1",
        api_key="k",
        model_id="m",
        params={"num_ctx": 8000, "max_tokens": 2000},
        system_prompt="",
        knowledge_ids=None,
    )
    registry.clear_caches()
    first = registry.get_agent(resolved, [])
    assert first.middleware
    compaction.save_settings({"headroom_percent": 10})
    assert registry.get_agent(resolved, []) is not first
    compaction.save_settings({"enabled": False})
    assert registry.get_agent(resolved, []).middleware is None
