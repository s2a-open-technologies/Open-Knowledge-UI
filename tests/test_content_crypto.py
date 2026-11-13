"""Chat-Inhalte liegen verschluesselt in der Datenbank, der Code sieht Klartext (app/core/content_crypto.py)."""

import pytest
from sqlalchemy import text

from app.auth.user_admin import create_user, delete_user
from app.core import keys
from app.core.crypto import SecretKeyError
from app.core.models import ChatSession, Message
from tests.test_access import add_user, client_for
from tests.web import make_app

SECRET = "Abfindung fuer Frau Mueller: 85.000 EUR"


def _raw(db, sql, **params):
    with db.engine.connect() as conn:
        return conn.execute(text(sql), params).scalar()


def test_messages_title_and_summary_are_stored_encrypted_and_read_as_plaintext(db):
    alice = add_user(db, "alice")
    sid = db.create_session(f"Titel {SECRET}", user_id=alice["id"]).id
    saved = db.save_message(sid, "user", SECRET)
    db.set_session_summary(sid, f"Zusammenfassung: {SECRET}", saved.id)

    assert saved.content == SECRET  # Objekt nach dem Speichern: Klartext
    for sql in (
        "SELECT content FROM messages WHERE session_id = :s",
        "SELECT title FROM sessions WHERE id = :s",
        "SELECT summary FROM sessions WHERE id = :s",
    ):
        raw = _raw(db, sql, s=sid)
        assert raw.startswith(keys.TEXT_PREFIX) and "Mueller" not in raw

    assert [m.content for m in db.get_messages(sid)] == [SECRET]
    summary, rows = db.get_history(sid)
    assert summary == f"Zusammenfassung: {SECRET}" and rows == []
    assert db.get_chat_session(sid, user_id=alice["id"]).title == f"Titel {SECRET}"


def test_loading_does_not_cause_writes(db):
    alice = add_user(db, "alice")
    sid = db.create_session("Titel", user_id=alice["id"]).id
    db.save_message(sid, "user", SECRET)
    before = _raw(db, "SELECT content FROM messages WHERE session_id = :s", s=sid)
    s = db.get_session()
    try:
        s.query(Message).filter_by(session_id=sid).one()
        assert not s.dirty
        s.commit()
    finally:
        s.close()
    assert _raw(db, "SELECT content FROM messages WHERE session_id = :s", s=sid) == before


def test_updating_another_column_keeps_the_ciphertext(db):
    alice = add_user(db, "alice")
    sid = db.create_session("Titel", user_id=alice["id"]).id
    db.save_message(sid, "user", SECRET)
    db.update_session_title(sid, "Neuer Titel")
    assert db.get_chat_session(sid).title == "Neuer Titel"
    raw = _raw(db, "SELECT title FROM sessions WHERE id = :s", s=sid)
    assert raw.startswith(keys.TEXT_PREFIX)


def test_chats_without_owner_stay_plaintext(db):
    sid = db.create_session("Altbestand").id
    db.save_message(sid, "user", "alt")
    assert _raw(db, "SELECT content FROM messages WHERE session_id = :s", s=sid) == "alt"


def test_existing_plaintext_messages_remain_readable(db):
    alice = add_user(db, "alice")
    sid = db.create_session("T", user_id=alice["id"]).id
    with db.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO messages (session_id, role, content) VALUES (:s, 'user', 'vorher')"
            ),
            {"s": sid},
        )
    assert [m.content for m in db.get_messages(sid)] == ["vorher"]


def test_the_api_returns_plaintext(db):
    alice = add_user(db, "alice")
    sid = db.create_session("Mein Chat", user_id=alice["id"]).id
    db.save_message(sid, "user", SECRET)
    c = client_for(make_app(), alice)
    data = c.get(f"/api/sessions/{sid}").json()
    assert data["title"] == "Mein Chat"
    assert [m["content"] for m in data["messages"]] == [SECRET]


def test_deleting_the_user_shreds_the_key(db):
    boss = create_user("boss", "Boss", "geheim12", "admin")
    gf = create_user("gfchef", "GF", "geheim12", "user")
    sid = db.create_session("Vertraulich", user_id=gf["id"]).id
    db.save_message(sid, "user", SECRET)
    sealed = _raw(db, "SELECT content FROM messages WHERE session_id = :s", s=sid)
    uuid = keys.owner_uuid(keys.USER, gf["id"])

    delete_user(gf["id"], actor_id=boss["id"])

    assert _raw(db, "SELECT COUNT(*) FROM data_keys WHERE owner_uuid = :u", u=uuid) == 0
    keys.reset()
    with pytest.raises(SecretKeyError):  # z. B. eine Kopie aus einem alten Backup
        keys.unseal_text(keys.USER, uuid, sealed)
    s = db.get_session()
    try:
        assert s.query(ChatSession).filter_by(id=sid).first() is None
    finally:
        s.close()
