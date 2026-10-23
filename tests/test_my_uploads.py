# ruff: noqa: F811  (pytest-Fixtures werden importiert und als Argument genutzt)
import io
from pathlib import Path

import pytest
from flask import Flask

from app.api import register_blueprints
from app.chat import attachments
from app.core.database_service import get_db_service
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
)


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    mr.reset_for_tests()
    monkeypatch.setattr(mr, "get_embedder", lambda spec: FakeDense())
    monkeypatch.setattr("app.wait_for_resources", lambda timeout=0: True, raising=False)
    yield
    mr.reset_for_tests()


@pytest.fixture
def app(db, qd):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    app.extensions.update(client=qd, sparse_embeddings=FakeSparse())
    register_blueprints(app)
    return app


def upload_and_ingest(app, qd, user, name, text):
    from app.auth import access

    access.add_grant(
        "knowledge", knowledge.get_knowledge(None)["id"], "user", user["id"], "write"
    )
    c = client_for(app, user)
    r = c.post(
        "/api/documents/upload",
        data={"files": (io.BytesIO(text.encode()), name)},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.json
    with make_ingestion(qd) as ing:
        ing.ingest_documents()
    return c


def test_knowledge_uploads_are_not_listed_in_my_uploads_but_the_uploader_can_delete_them_in_the_knowledge(
    app, db, qd, no_enrichment
):
    ed, other = add_user(db, "ed", "editor"), add_user(db, "other", "editor")
    c = upload_and_ingest(app, qd, ed, "vertrag.md", DOC)
    assert scroll_all(qd)
    doc = get_db_service().get_all_documents(knowledge.get_knowledge(None)["id"])[0]
    md_file = Path(doc.markdown_path)
    assert md_file.is_file()

    # "Meine Uploads" zeigt nur Chat-Anhaenge; Knowledge-Dokumente stehen im Knowledge
    assert c.get("/api/my-uploads").json == []
    assert c.get(f"/api/my-uploads/knowledge/{doc.id}/markdown").status_code == 404
    assert c.delete(f"/api/my-uploads/knowledge/{doc.id}").status_code == 404
    assert scroll_all(qd)  # nichts wurde geloescht
    assert (
        "Mietvertrag" in c.get(f"/api/documents/{doc.id}/markdown").json["markdown"]
    )  # Ansicht im Knowledge

    assert c.delete(f"/api/documents/{doc.id}").status_code == 200
    assert scroll_all(qd) == [] and not md_file.exists()
    assert get_db_service().get_all_documents(knowledge.get_knowledge(None)["id"]) == []
    assert client_for(app, other).get("/api/my-uploads").json == []


def test_uploader_may_delete_own_document_without_write_access_to_the_knowledge_but_others_may_not(
    app, db, qd, no_enrichment
):
    ed = add_user(db, "ed", "editor")
    c = upload_and_ingest(app, qd, ed, "vertrag.md", DOC)
    doc = get_db_service().get_all_documents(knowledge.get_knowledge(None)["id"])[0]
    stranger = add_user(db, "fremd", "user")
    # der Nutzer verliert die Editor-Rolle, bleibt aber Eigentuemer seiner Datei
    s = db.get_session()
    from app.core.models import User

    s.query(User).filter_by(id=ed["id"]).update({"role": "user"})
    s.commit()
    s.close()
    assert client_for(app, stranger).delete(f"/api/documents/{doc.id}").status_code in (
        403,
        404,
    )  # nicht der Eigentuemer
    assert scroll_all(qd)
    assert c.delete(f"/api/documents/{doc.id}").status_code == 200
    assert scroll_all(qd) == []


def test_knowledge_delete_route_removes_markdown_vectors_and_tables(
    app, db, qd, no_enrichment
):
    ed = add_user(db, "ed", "editor")
    c = upload_and_ingest(app, qd, ed, "vertrag.md", DOC)
    doc = get_db_service().get_all_documents(knowledge.get_knowledge(None)["id"])[0]
    md_file = Path(doc.markdown_path)
    assert c.delete(f"/api/documents/{doc.id}").status_code == 200
    assert scroll_all(qd) == [] and not md_file.exists()
    assert not list(Path("data/files").glob("*.uploader"))  # Begleitdatei aufgeraeumt


def test_chat_attachment_is_listed_viewable_and_individually_deletable(
    app, db, qd, monkeypatch
):
    alice, bob = add_user(db, "alice", "user"), add_user(db, "bob", "user")
    sid = db.create_session("Mein Chat", user_id=alice["id"]).id

    def add(name):
        att = db.create_attachment(sid, name, alice["id"])
        folder = attachments.session_dir(sid) / att.id
        folder.mkdir(parents=True)
        (folder / name).write_text(
            "# Anhang\n\nDie Kaution betraegt drei Monatsmieten.", encoding="utf-8"
        )
        attachments._ingest(att.id, sid, str(folder / name), folder, FakeSparse())
        return att

    a, b = add("a.md"), add("b.md")
    c = client_for(app, alice)
    items = c.get("/api/my-uploads").json
    assert {i["id"] for i in items} == {a.id, b.id} and all(
        i["kind"] == "chat" for i in items
    )
    assert items[0]["source"] == "Mein Chat" and items[0]["has_markdown"]
    assert "Kaution" in c.get(f"/api/my-uploads/chat/{a.id}/markdown").json["markdown"]

    assert client_for(app, bob).get("/api/my-uploads").json == []
    assert (
        client_for(app, bob).delete(f"/api/my-uploads/chat/{a.id}").status_code == 404
    )

    assert c.delete(f"/api/my-uploads/chat/{a.id}").status_code == 200
    remaining = [i["id"] for i in c.get("/api/my-uploads").json]
    assert remaining == [b.id]  # der andere Anhang desselben Chats bleibt
    points = [
        p
        for col in qd.get_collections().collections
        for p in qd.scroll(col.name, limit=100, with_payload=True)[0]
        if col.name.startswith("user_uploads")
    ]
    assert points and all(p.payload["attachment_id"] == b.id for p in points)
    assert not (attachments.session_dir(sid) / a.id).exists()


def test_deleting_the_chat_cleans_up_attachments_and_markdown(app, db, qd):
    alice = add_user(db, "alice", "user")
    sid = db.create_session("Chat", user_id=alice["id"]).id
    att = db.create_attachment(sid, "a.md", alice["id"])
    folder = attachments.session_dir(sid) / att.id
    folder.mkdir(parents=True)
    (folder / "a.md").write_text("# T\n\nText zum Anhang.", encoding="utf-8")
    attachments._ingest(att.id, sid, str(folder / "a.md"), folder, FakeSparse())
    attachments.cleanup_session_attachments(sid, qd)
    assert not attachments.session_dir(sid).exists()
    assert client_for(app, alice).get("/api/my-uploads").json == []
