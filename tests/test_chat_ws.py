"""Chat ueber /ws/chat: sofortiger Stopp, nichts wird gespeichert, leere neue Sessions verschwinden."""

import asyncio
import time
import uuid

import pytest

from app.auth import access
from app.chat import streams
from app.core import runtime
from app.llm import connections as store
from tests.test_access import add_user
from tests.web import chat_ws, client_for, make_app


class Chunk:
    def __init__(self, content):
        self.content = content
        self.additional_kwargs = {}


def token(text):
    return {"event": "on_chat_model_stream", "data": {"chunk": Chunk(text)}}


class SlowAgent:
    """Schreibt ein Token und wartet dann (wie ein Modell, das nachdenkt)."""

    def __init__(self, wait=None):
        self.wait = wait or (lambda: asyncio.sleep(30))
        self.finished = False

    async def astream_events(self, *a, **k):
        yield token("Hallo")
        await self.wait()
        self.finished = True
        yield token(" Welt")


class QuickAgent:
    async def astream_events(self, *a, **k):
        yield token("Fertig")


@pytest.fixture
def app(db):
    return make_app()


@pytest.fixture
def model(db, monkeypatch):
    from app.chat import service

    conn = store.create_connection("C", "http://x/v1", "k")
    agent = SlowAgent()
    monkeypatch.setattr(service, "get_agent", lambda resolved, tools: agent)
    monkeypatch.setattr(service, "resources_ready", lambda: True)
    runtime.state.agent_tools = []
    return {"ref": store.make_base_ref(conn["id"], "m"), "agent": agent}


def start(ws, stream_id, **body):
    ws.send_json({"type": "message", "stream_id": stream_id, "message": "Frage", **body})
    assert ws.receive_json() == {"type": "ready", "stream_id": stream_id}
    assert ws.receive_json() == {"token": "Hallo", "stream_id": stream_id}


def wait_for_end(ws):
    while True:
        event = ws.receive_json()
        if event.get("type") == "cancelled" or event.get("done") or "error" in event:
            return event


def wait_until(check, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.02)
    return check()


def test_a_new_chat_stops_at_once_saves_nothing_and_leaves_no_empty_session(app, db, model):
    alice = add_user(db, "alice", "user")
    c = client_for(app, alice)
    sid = uuid.uuid4().hex
    with c.websocket_connect("/ws/chat") as ws:
        start(ws, sid, model=model["ref"])
        began = time.monotonic()
        ws.send_json({"type": "stop", "stream_id": sid})
        assert wait_for_end(ws) == {"type": "cancelled", "stream_id": sid}
        assert time.monotonic() - began < 2  # nicht auf das Modell warten
    assert not model["agent"].finished
    assert wait_until(lambda: db.list_sessions(user_id=alice["id"]) == [])
    assert c.get("/api/sessions").json() == []


def test_stopping_an_existing_chat_keeps_its_history_unchanged(app, db, model):
    alice = add_user(db, "alice", "user")
    sess = db.create_session(user_id=alice["id"])
    db.save_message(sess.id, "user", "Erste Frage")
    db.save_message(sess.id, "assistant", "Erste Antwort")
    sid = uuid.uuid4().hex
    with client_for(app, alice).websocket_connect("/ws/chat") as ws:
        start(ws, sid, model=model["ref"], session_id=sess.id)
        ws.send_json({"type": "stop", "stream_id": sid})
        assert wait_for_end(ws)["type"] == "cancelled"
    assert db.get_chat_session(sess.id, user_id=alice["id"]) is not None
    assert [m.content for m in db.get_messages(sess.id)] == ["Erste Frage", "Erste Antwort"]


def test_a_running_blocking_tool_does_not_delay_the_stop(app, db, model, monkeypatch):
    """Ein synchrones Werkzeug (Thread) rechnet weiter, der Chat endet trotzdem sofort."""
    tool_done = []

    def slow_tool():
        time.sleep(1.5)
        tool_done.append(True)

    model["agent"].wait = lambda: asyncio.to_thread(slow_tool)
    sid = uuid.uuid4().hex
    with client_for(app, add_user(db, "ed", "editor")).websocket_connect("/ws/chat") as ws:
        start(ws, sid, model=model["ref"])
        began = time.monotonic()
        ws.send_json({"type": "stop", "stream_id": sid})
        assert wait_for_end(ws)["type"] == "cancelled"
        assert time.monotonic() - began < 1.0
    assert not model["agent"].finished  # das Ergebnis des Werkzeugs wird verworfen


def test_the_http_fallback_stops_only_own_streams(app, db, model):
    alice, bob = add_user(db, "alice", "user"), add_user(db, "bob", "user")
    sid = uuid.uuid4().hex
    with client_for(app, alice).websocket_connect("/ws/chat") as ws:
        start(ws, sid, model=model["ref"])
        # Bob kennt die ID, darf aber nicht stoppen (still ignoriert, wie eine unbekannte ID)
        assert client_for(app, bob).post("/chat/stop", json={"stream_id": sid}).json() == {
            "success": True
        }
        assert streams.registry.owner(sid) == alice["id"]
        assert client_for(app, alice).post("/chat/stop", json={"stream_id": sid}).status_code == 200
        assert wait_for_end(ws)["type"] == "cancelled"


def test_unknown_and_repeated_stops_are_harmless(app, db, model):
    alice = add_user(db, "alice", "user")
    c = client_for(app, alice)
    assert c.post("/chat/stop", json={"stream_id": "gibt-es-nicht"}).status_code == 200
    sid = uuid.uuid4().hex
    with c.websocket_connect("/ws/chat") as ws:
        start(ws, sid, model=model["ref"])
        ws.send_json({"type": "stop", "stream_id": sid})
        ws.send_json({"type": "stop", "stream_id": sid})
        assert wait_for_end(ws)["type"] == "cancelled"
        # die Verbindung bleibt nutzbar
        ws.send_json({"type": "stop", "stream_id": sid})
        ws.send_json({"type": "nix"})
        assert ws.receive_json() == {"error": "Unbekannte Nachricht"}


def test_only_one_stream_per_connection(app, db, model):
    sid, second = uuid.uuid4().hex, uuid.uuid4().hex
    with client_for(app, add_user(db, "ed", "editor")).websocket_connect("/ws/chat") as ws:
        start(ws, sid, model=model["ref"])
        ws.send_json({"type": "message", "stream_id": second, "message": "Noch eine"})
        assert ws.receive_json() == {
            "stream_id": second,
            "error": "Es läuft bereits eine Antwort.",
        }
        ws.send_json({"type": "stop", "stream_id": sid})
        assert wait_for_end(ws)["type"] == "cancelled"


def test_closing_the_socket_cancels_the_stream(app, db, model):
    alice = add_user(db, "alice", "user")
    sid = uuid.uuid4().hex
    with client_for(app, alice).websocket_connect("/ws/chat") as ws:
        start(ws, sid, model=model["ref"])
    assert wait_until(lambda: streams.registry.owner(sid) is None)
    assert not model["agent"].finished
    assert wait_until(lambda: db.list_sessions(user_id=alice["id"]) == [])


def test_a_finished_answer_is_saved(app, db, model, monkeypatch):
    from app.chat import service

    monkeypatch.setattr(service, "get_agent", lambda resolved, tools: QuickAgent())
    alice = add_user(db, "alice", "user")
    events = chat_ws(client_for(app, alice), message="Frage", model=model["ref"])
    done = events[-1]
    assert done["done"] is True
    assert [m.content for m in db.get_messages(done["session_id"])] == ["Frage", "Fertig"]


def test_errors_before_the_run_remove_the_new_session(app, db, model):
    alice = add_user(db, "alice", "user")
    ref = model["ref"]
    store.save_model_config(ref, {"is_public": False})  # Alice darf das Modell nicht nutzen
    assert not access.can_use_model(alice, ref)
    events = chat_ws(client_for(app, alice), message="Frage", model=ref)
    assert "Kein Zugriff" in events[-1]["error"]
    assert db.list_sessions(user_id=alice["id"]) == []


def test_sockets_need_a_login_and_the_same_origin(app, db):
    from starlette.websockets import WebSocketDisconnect

    from tests.web import anon

    with pytest.raises(WebSocketDisconnect):
        with anon(app).websocket_connect("/ws/chat") as ws:
            ws.receive_json()
    c = client_for(app, add_user(db, "ed", "editor"))
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect(
            "/ws/chat", headers={"origin": "https://boese.example.net"}
        ) as ws:
            ws.receive_json()
