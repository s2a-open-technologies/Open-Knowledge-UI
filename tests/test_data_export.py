"""Datenexport (GET /api/me/export): nur eigene Daten, JSON mit Profil/Chats/Analysen."""

import json

from app.chat import summaries as summaries_store
from app.core import database_service
from app.core.models import Message
from tests.test_access import add_user, app, client_for  # noqa: F401


def _add_message(session_id: str, role: str, content: str) -> None:
    db = database_service.get_db_service().get_session()
    try:
        db.add(Message(session_id=session_id, role=role, content=content))
        db.commit()
    finally:
        db.close()


def test_export_requires_login(app):  # noqa: F811
    assert app.test_client().get("/api/me/export").status_code == 401


def test_export_contains_only_own_data(db, app):  # noqa: F811
    alice = add_user(db, "alice")
    bob = add_user(db, "bob")

    own = database_service.get_db_service().create_session(
        "Alice-Frage", user_id=alice["id"]
    )
    _add_message(own.id, "user", "Was ist RAG?")
    foreign = database_service.get_db_service().create_session(
        "Bob-Geheimnis", user_id=bob["id"]
    )
    _add_message(foreign.id, "user", "Fremdinhalt")

    summaries_store.user_dir(alice["id"], create=True)
    path = summaries_store.safe_path(alice["id"], "notiz.md")
    assert path is not None
    path.write_text("# Notiz\nInhalt von alice", encoding="utf-8")
    database_service.get_db_service().create_summary_file(
        "notiz.md", user_id=alice["id"]
    )

    client = client_for(app, alice)
    res = client.get("/api/me/export")
    assert res.status_code == 200
    assert (
        'filename="open-knowledge-ui-export-alice-'
        in res.headers["Content-Disposition"]
    )

    payload = json.loads(res.get_data(as_text=True))
    assert payload["format"] == "open-knowledge-ui-export"
    assert payload["profil"]["username"] == "alice"
    assert len(payload["chats"]) == 1
    chat = payload["chats"][0]
    assert chat["titel"] == "Alice-Frage"
    assert [m["inhalt"] for m in chat["nachrichten"]] == ["Was ist RAG?"]
    assert payload["meine_analysen"][0]["inhalt"] == "# Notiz\nInhalt von alice"

    assert "Fremdinhalt" not in res.get_data(as_text=True)
    assert all(c["titel"] != "Bob-Geheimnis" for c in payload["chats"])
