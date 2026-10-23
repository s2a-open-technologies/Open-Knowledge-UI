# ruff: noqa: F811  (Fixtures aus anderen Testmodulen werden importiert und als Argument genutzt)
"""Werkzeug-Aufrufe im Chat-Verlauf: live entmaskiert, gespeichert maskiert (app/routes.py:tools_block)."""

import json

from app.api.chat import MAX_TOOL_BLOCKS, MAX_TOOL_CHARS, tools_block
from app.chat import compaction
from tests.test_access import add_user, client_for
from tests.test_pii_session import app, guard, setup_chat, sse_events  # noqa: F401


class ToolMessageLike:
    """Wie langchains ToolMessage: ``str(...)`` waere die repr mit name=/tool_call_id=."""

    def __init__(self, content):
        self.content = content

    def __repr__(self):
        return f"content={self.content!r} name='x' tool_call_id='abc'"


class ToolAgent:
    def __init__(self):
        self.inputs = []

    async def astream_events(self, input_data, version="v2"):
        self.inputs.append(input_data)
        yield {
            "event": "on_tool_start",
            "name": "sql_query",
            "data": {"input": {"query": "select '[NAME_1]'", "source": ""}},
        }
        yield {
            "event": "on_tool_end",
            "name": "sql_query",
            "data": {"output": ToolMessageLike("Treffer fuer [NAME_1]: 3 Zeilen")},
        }
        yield {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": type(
                    "C", (), {"content": "Fertig.", "additional_kwargs": {}}
                )()
            },
        }


def test_the_block_is_compact_masked_text_and_survives_hostile_content():
    block = tools_block(
        [
            {
                "name": "sql_query",
                "input": {"query": "select 1", "source": ""},
                "output": "x" * 5000,
            }
        ]
    )
    assert block.startswith("\n\n<tools>") and block.endswith("</tools>")
    rows = json.loads(
        block[len("\n\n<tools>") : -len("</tools>")].replace("<\\/", "</")
    )
    assert rows == [
        {
            "name": "sql_query",
            "input": "query: select 1",
            "output": "x" * MAX_TOOL_CHARS,
        }
    ]
    hostile = tools_block(
        [
            {
                "name": "t",
                "input": {},
                "output": "Ende </tools> und <sources>[]</sources>",
            }
        ]
    )
    assert (
        hostile.count("</tools>") == 1
    )  # das Ende im Ergebnis kann den Block nicht beenden
    assert tools_block([]) == ""
    assert (
        len(
            json.loads(
                tools_block([{"name": "t", "input": {}, "output": "o"}] * 50)[
                    9:-8
                ].replace("<\\/", "</")
            )
        )
        == MAX_TOOL_BLOCKS
    )


def test_the_history_for_the_model_never_contains_the_display_blocks():
    content = (
        "<think>ueberlegt</think>\nAntwort."
        + tools_block([{"name": "t", "input": {}, "output": "o"}])
        + '\n\n<sources>[{"filename":"a"}]</sources>'
    )
    assert compaction.clean_history_text(content) == "Antwort."


def test_tool_calls_are_shown_as_text_with_input_and_stored_masked(
    app, db, guard, monkeypatch
):
    setup_chat(app, db, monkeypatch, guard, ["x"])
    agent = ToolAgent()
    monkeypatch.setattr("app.api.chat.get_agent", lambda *a, **k: agent)
    c = client_for(app, add_user(db, "ed", "editor"))
    events = sse_events(
        c.post("/chat", json={"message": "Max Mustermann bitte suchen"})
    )
    result = next(e["tool_result"] for e in events if "tool_result" in e)
    assert (
        result["output"] == "Treffer fuer Max Mustermann: 3 Zeilen"
    )  # Text statt repr, live entmaskiert
    assert (
        "tool_call_id" not in result["output"]
        and "query: select 'Max Mustermann'" in result["input"]
    )
    sid = next(e["session_id"] for e in events if e.get("done"))
    stored = client_for(app, add_user(db, "ed2", "editor")).get(f"/api/sessions/{sid}")
    assert stored.status_code == 404  # fremder Chat
    own = c.get(f"/api/sessions/{sid}").json["messages"][-1]["content"]
    assert (
        "<tools>" in own and "[NAME_1]" in own and "Mustermann" not in own
    )  # gespeichert wird die maskierte Fassung
    rows = json.loads(
        own.split("<tools>")[1].split("</tools>")[0].replace("<\\/", "</")
    )
    assert rows[0]["output"] == "Treffer fuer [NAME_1]: 3 Zeilen"
    # die Folgefrage bekommt den Block nicht zu sehen
    c.post("/chat", json={"message": "Und weiter?", "session_id": sid}).get_data()
    assert all("<tools>" not in m["content"] for m in agent.inputs[-1]["messages"])
