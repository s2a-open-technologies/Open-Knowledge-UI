# ruff: noqa: F811  (pytest-Fixtures werden importiert und als Argument genutzt)
"""Datenschutz: Datensparsamkeit, vollstaendiges Loeschen, Nachweis im Audit-Log, Aufbewahrungsfrist."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app.auth import audit
from app.auth.user_admin import create_user, delete_user, update_user
from app.chat import attachments, retention, summaries
from app.core.database_service import get_db_service
from app.core.models import (
    AuditLog,
    ChatAttachment,
    ChatSession,
    UserDocument,
    UserSummaryFile,
)
from app.knowledge import service as knowledge
from app.rag import models_runtime as mr
from tests.test_access import add_user, client_for
from tests.test_ingest import (  # noqa: F401
    DOC,
    FakeDense,
    FakeSparse,
    make_ingestion,
    no_enrichment,
    qd,
    scroll_all,
    set_setting,
)
from tests.test_my_uploads import upload_and_ingest
from tests.web import make_app, set_state


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    mr.reset_for_tests()
    monkeypatch.setattr(mr, "get_embedder", lambda spec: FakeDense())
    monkeypatch.setattr("app.wait_for_resources", lambda timeout=0: True, raising=False)
    yield
    mr.reset_for_tests()


@pytest.fixture
def app(db, qd):
    app = make_app()
    set_state(client=qd, sparse_embeddings=FakeSparse())
    return app


def _actions():
    return [e["action"] for e in audit.list_events(limit=500)]


# --- Datensparsamkeit ---------------------------------------------------------------------


def test_a_failed_chat_attachment_leaves_no_original_behind(db, qd):
    alice = add_user(db, "alice", "user")
    sid = db.create_session("Chat", user_id=alice["id"]).id
    att = db.create_attachment(sid, "leer.md", alice["id"])
    folder = attachments.session_dir(sid) / att.id
    folder.mkdir(parents=True)
    original = folder / "leer.md"
    original.write_text("", encoding="utf-8")  # kein lesbarer Inhalt -> Fehler

    attachments._ingest(att.id, sid, str(original), folder, FakeSparse())

    assert db.get_attachment(att.id).status == "error"
    assert not original.exists()


def test_deleting_a_document_after_reindex_also_removes_the_archived_original(
    app, db, qd, no_enrichment
):
    set_setting(db, "INGEST_KEEP_ORIGINAL", "true")
    ed = add_user(db, "ed", "editor")
    c = upload_and_ingest(app, qd, ed, "vertrag.md", DOC)
    archived = Path("data/processed_files/vertrag.md")
    assert archived.exists()
    # Nach einem Neu-Indexieren steht im Payload als source nur noch der Dateiname
    for p in scroll_all(qd):
        qd.set_payload("open_knowledge_ui", {"source": "vertrag.md"}, points=[p.id])

    doc = get_db_service().get_all_documents(knowledge.get_knowledge(None)["id"])[0]
    assert c.delete(f"/api/documents/{doc.id}").status_code == 200
    assert not archived.exists()


# --- Nutzer loeschen ----------------------------------------------------------------------


def test_deleting_a_user_removes_summaries_attachments_and_detaches_references(db, qd):
    boss = create_user("boss", "Boss", "geheim12", "admin")
    gf = create_user("gfchef", "GF", "geheim12", "user")
    sid = db.create_session("Vertraulich", user_id=gf["id"]).id
    db.create_attachment(sid, "a.md", gf["id"])
    folder = summaries.user_dir(gf["id"], create=True)
    (folder / "analyse.md").write_text("Gehalt GF", encoding="utf-8")
    s = db.get_session()
    s.add(UserSummaryFile(filename="analyse.md", user_id=gf["id"]))
    s.add(
        UserDocument(
            document_id="doc-1",
            filename="x.md",
            knowledge_id=knowledge.get_knowledge(None)["id"],
            uploaded_by=gf["id"],
        )
    )
    s.add(AuditLog(action="login_ok", actor_user_id=gf["id"], actor_name="gfchef"))
    s.commit()
    s.close()

    session_ids = delete_user(gf["id"], actor_id=boss["id"])
    for sid_ in session_ids:
        attachments.cleanup_session_attachments(sid_, qd)

    assert not folder.exists()
    s = db.get_session()
    try:
        assert s.query(UserSummaryFile).count() == 0
        assert s.query(UserDocument).filter_by(document_id="doc-1").one().uploaded_by is None
        login = s.query(AuditLog).filter_by(action="login_ok").one()
        assert login.actor_user_id is None and login.actor_name == "gfchef"
        assert s.query(ChatAttachment).filter_by(session_id=sid).count() == 0
    finally:
        s.close()


# --- Audit --------------------------------------------------------------------------------


def test_a_password_reset_by_an_admin_is_recorded_separately(db):
    boss = create_user("boss", "Boss", "geheim12", "admin")
    gf = create_user("gfchef", "GF", "geheim12", "user")

    update_user(gf["id"], actor_id=boss["id"], data={"display_name": "Chefin"})
    assert "password_reset_by_admin" not in _actions()

    update_user(gf["id"], actor_id=boss["id"], data={"password": "neuesgeheim1"})
    event = next(e for e in audit.list_events() if e["action"] == "password_reset_by_admin")
    assert event["object_id"] == str(gf["id"]) and event["actor_user_id"] == boss["id"]
    assert "neuesgeheim1" not in (event["detail"] or "")


@pytest.mark.parametrize(
    "method,path,expected",
    [
        ("PUT", "/api/admin/tools", "admin/tools"),
        ("POST", "/api/connections", "connections"),
        ("PATCH", "/api/connections/3", "connections"),
        ("PUT", "/api/models/config", "models/config"),
        ("DELETE", "/api/knowledge/4", "knowledge"),
        ("GET", "/api/admin/tools", None),
        ("POST", "/api/admin/tools/test", None),
        ("POST", "/api/connections/test", None),
        ("PUT", "/api/knowledge/4/access", None),  # eigenes Ereignis access_set
        ("POST", "/api/documents/upload", None),
        ("POST", "/api/sessions", None),
    ],
)
def test_config_change_target(method, path, expected):
    assert audit.config_change_target(method, path) == expected


def test_successful_config_changes_are_recorded_without_the_body(app, db):
    admin_user = add_user(db, "chef", "admin")
    c = client_for(app, admin_user)
    r = c.put("/api/admin/retention", json={"chat_retention_days": 30})
    assert r.status_code == 200
    event = next(e for e in audit.list_events() if e["action"] == "config_change")
    assert event["object_type"] == "admin/retention"
    assert event["actor_user_id"] == admin_user["id"]
    assert "30" not in (event["detail"] or "")

    before = _actions().count("config_change")
    assert c.put("/api/admin/retention", json={"chat_retention_days": -1}).status_code == 400
    assert _actions().count("config_change") == before  # abgelehnte Aenderung zaehlt nicht


def test_viewing_someone_elses_knowledge_document_is_recorded(app, db, qd, no_enrichment):
    ed = add_user(db, "ed", "editor")
    reader = add_user(db, "leser", "user")
    c = upload_and_ingest(app, qd, ed, "vertrag.md", DOC)
    doc = get_db_service().get_all_documents(knowledge.get_knowledge(None)["id"])[0]

    assert c.get(f"/api/documents/{doc.id}/markdown").status_code == 200
    assert "document_view" not in _actions()  # eigener Upload

    r = client_for(app, reader).get(f"/api/documents/{doc.id}/markdown")
    assert r.status_code == 200  # Standard-Knowledge ist oeffentlich
    event = next(e for e in audit.list_events() if e["action"] == "document_view")
    assert event["actor_user_id"] == reader["id"] and event["object_id"] == str(doc.id)


# --- Aufbewahrungsfrist -------------------------------------------------------------------


def _age_session(db, session_id, days):
    s = db.get_session()
    row = s.query(ChatSession).filter_by(id=session_id).one()
    row.updated_at = datetime.utcnow() - timedelta(days=days)
    s.commit()
    s.close()


def test_retention_is_off_by_default(db, qd):
    alice = add_user(db, "alice", "user")
    sid = db.create_session("Alt", user_id=alice["id"]).id
    _age_session(db, sid, 5000)
    assert retention.load_settings() == {"chat_retention_days": 0}
    assert retention.purge_expired(qd) == 0
    assert db.get_chat_session(sid) is not None


def test_retention_deletes_old_chats_with_attachments_and_keeps_recent_ones(db, qd):
    alice = add_user(db, "alice", "user")
    old = db.create_session("Alt", user_id=alice["id"]).id
    new = db.create_session("Neu", user_id=alice["id"]).id
    att = db.create_attachment(old, "a.md", alice["id"])
    folder = attachments.session_dir(old) / att.id
    folder.mkdir(parents=True)
    (folder / "a.md").write_text("# T\n\nVertraulicher Anhang.", encoding="utf-8")
    attachments._ingest(att.id, old, str(folder / "a.md"), folder, FakeSparse())
    _age_session(db, old, 40)
    _age_session(db, new, 2)

    retention.save_settings({"chat_retention_days": 30})
    assert retention.purge_expired(qd) == 1

    assert db.get_chat_session(old) is None and db.get_chat_session(new) is not None
    assert not attachments.session_dir(old).exists()
    assert db.get_attachment(att.id) is None
    leftover = [
        p
        for col in qd.get_collections().collections
        if col.name.startswith("user_uploads")
        for p in qd.scroll(col.name, limit=100, with_payload=True)[0]
    ]
    assert leftover == []


def test_retention_rejects_invalid_values(db):
    for bad in (-1, 99999, "abc"):
        with pytest.raises(retention.RetentionConfigError):
            retention.save_settings({"chat_retention_days": bad})
