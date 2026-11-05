"""Test-Helfer fuer die FastAPI-App: App bauen, angemeldete Clients, Chat ueber den WebSocket."""

import json
import uuid
from base64 import b64encode

import itsdangerous
from fastapi.testclient import TestClient

from app.core import runtime


def make_app(csrf: bool = False):
    """App ohne Qdrant, Modelle und Redis; CSRF aus (ausser ``csrf=True``)."""
    from app import create_app
    from app.core import database_service

    # Nie die echte Datenbank im Arbeitsverzeichnis anfassen: die Test-DB kommt aus der db-Fixture.
    if database_service.db_service is None:
        raise RuntimeError("make_app() braucht die db-Fixture (Test-Datenbank)")
    app = create_app(resources=False)
    app.state.csrf_enabled = csrf
    return app


def set_state(**values) -> None:
    """Felder des Laufzeit-Zustands setzen (frueher ``app.extensions.update``)."""
    for key, value in values.items():
        setattr(runtime.state, key, value)


def session_cookie(app, data: dict) -> str:
    """Session-Cookie wie die ``SessionMiddleware`` es signiert."""
    payload = b64encode(json.dumps(data).encode("utf-8"))
    return (
        itsdangerous.TimestampSigner(str(app.state.secret_key))
        .sign(payload)
        .decode("utf-8")
    )


def anon(app) -> TestClient:
    """Client ohne Anmeldung; Weiterleitungen werden nicht verfolgt (wie beim Flask-Testclient)."""
    return TestClient(app, follow_redirects=False)


def client_for(app, user: dict) -> TestClient:
    """Als ``user`` angemeldeter Client."""
    client = anon(app)
    client.cookies.set("session", session_cookie(app, {"user_id": user["id"]}))
    return client


class ChatEvents(list):
    """Events eines Chat-Streams; ``text`` ist alles als JSON-Text (fuer die Suche nach Meldungen)."""

    @property
    def text(self) -> str:
        return json.dumps(list(self), ensure_ascii=False)


def chat_ws(client: TestClient, stream_id: str | None = None, **body) -> ChatEvents:
    """Eine Frage ueber ``/ws/chat`` stellen und alle Events bis ``done``/``error``/``cancelled`` lesen."""
    stream_id = stream_id or uuid.uuid4().hex
    events = ChatEvents()
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "message", "stream_id": stream_id, **body})
        while True:
            event = ws.receive_json()
            if event.get("type") == "ready":
                continue
            events.append(event)
            if (
                event.get("done")
                or "error" in event
                or event.get("type") == "cancelled"
            ):
                break
    return events


def chat_text(events: list[dict]) -> str:
    """Gestreamter Antworttext aus den Events."""
    return "".join(e.get("token", "") for e in events)
