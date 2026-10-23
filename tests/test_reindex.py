# ruff: noqa: F811  (pytest-Fixtures werden importiert und als Argument genutzt)
import json
from pathlib import Path

import pytest
from flask import Flask

from app.api.admin_routes import admin_bp
from app.api.api_routes import api_bp
from app.api.routes import bp as main_bp
from app.auth import access
from app.core.database_service import get_db_service
from app.ingest import reindex
from app.knowledge import service as knowledge
from app.rag import model_settings
from app.rag import models_runtime as mr
from tests.test_access import add_user, client_for
from tests.test_ingest import (  # noqa: F401  (Fixtures)
    DOC,
    FakeDense,
    FakeSparse,
    make_ingestion,
    no_enrichment,
    put_file,
    qd,
    scroll_all,
    set_setting,
)

NEW = mr.ModelSpec("hf", "neues/modell")


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    """Jedes Modell der Registry ist der Fake-Embedder (kein Download)."""
    mr.reset_for_tests()
    monkeypatch.setattr(mr, "get_embedder", lambda spec: FakeDense())
    yield
    mr.reset_for_tests()


def ingest_two(qd):
    put_file("a.md", DOC.format(pad="A"))
    put_file("b.md", DOC.format(pad="B"))
    with make_ingestion(qd) as ing:
        ing.ingest_documents()


def default_kb():
    return knowledge.get_knowledge(None)


def test_reindex_builds_a_new_collection_and_swaps(db, qd, no_enrichment):
    ingest_two(qd)
    before = scroll_all(qd)
    assert before
    model_settings.save_specs(NEW)
    assert model_settings.is_stale(default_kb())

    reindex.reindex_knowledge(default_kb(), qd, FakeSparse())

    kb = default_kb()
    assert kb["collection_name"] == "open_knowledge_ui_g1"
    assert not qd.collection_exists("open_knowledge_ui")
    after = scroll_all(qd, "open_knowledge_ui_g1")
    assert len(after) == len(before)
    assert {p.payload["file_group_id"] for p in after} == {
        p.payload["file_group_id"] for p in before
    }
    assert (
        json.loads(kb["embedding_spec"]) == NEW.to_dict()
        and kb["embedding_dimension"] == 7
    )
    assert not model_settings.is_stale(kb)
    assert kb["last_reindex_error"] is None


def test_a_second_reindex_numbers_the_generation_up(db, qd, no_enrichment):
    ingest_two(qd)
    reindex.reindex_knowledge(default_kb(), qd, FakeSparse())
    reindex.reindex_knowledge(default_kb(), qd, FakeSparse())
    assert default_kb()["collection_name"] == "open_knowledge_ui_g2"
    assert not qd.collection_exists("open_knowledge_ui_g1")


def test_restart_keeps_the_renamed_default_knowledge(db, qd, no_enrichment):
    ingest_two(qd)
    reindex.reindex_knowledge(default_kb(), qd, FakeSparse())
    db.init_default_data()
    assert [k["name"] for k in knowledge.list_knowledge()] == ["Standard"]


def test_failure_keeps_the_old_index_active(db, qd, no_enrichment, monkeypatch):
    ingest_two(qd)
    before = len(scroll_all(qd))
    model_settings.save_specs(NEW)

    def boom(self, *a, **kw):
        raise RuntimeError("Modell weg")

    monkeypatch.setattr("app.ingest.pipeline.DocumentIngestion.index_markdown", boom)
    with pytest.raises(RuntimeError):
        reindex.reindex_knowledge(default_kb(), qd, FakeSparse())

    assert default_kb()["collection_name"] == "open_knowledge_ui"
    assert len(scroll_all(qd)) == before
    assert [c.name for c in qd.get_collections().collections].count(
        "open_knowledge_ui_g1"
    ) == 0


def test_job_records_the_error_on_the_knowledge(db, qd, no_enrichment, monkeypatch):
    ingest_two(qd)
    monkeypatch.setattr(
        reindex,
        "reindex_knowledge",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("kaputt")),
    )
    reindex.start([default_kb()["id"]], qd, FakeSparse(), wait=True)
    st = reindex.status()
    assert not st["running"] and st["results"][0]["ok"] is False
    assert "kaputt" in default_kb()["last_reindex_error"]


def test_documents_without_markdown_are_reparsed_or_rebuilt_from_chunks(
    db, qd, no_enrichment
):
    ingest_two(qd)
    docs = get_db_service().get_all_documents(default_kb()["id"])
    # Altbestand simulieren: kein gespeichertes Markdown
    s = db.get_session()
    for d in s.query(type(docs[0])).all():
        Path(d.markdown_path).unlink()
        d.markdown_path = None
    s.commit()
    s.close()
    # a.md: Original im Archiv vorhanden; b.md: Original weg -> Notloesung aus den Chunks
    next(Path("data/processed_files").rglob("b.md")).unlink()

    reindex.reindex_knowledge(default_kb(), qd, FakeSparse())

    points = scroll_all(qd, "open_knowledge_ui_g1")
    by_file = {}
    for p in points:
        by_file.setdefault(p.payload["filename"], []).append(p)
    assert set(by_file) == {"a.md", "b.md"}
    assert any(
        p.payload["headers"] for p in by_file["a.md"]
    )  # neu geparst: Ueberschriften da
    assert all(
        Path(d.markdown_path).is_file()
        for d in get_db_service().get_all_documents(default_kb()["id"])
    )


def test_locked_while_running_and_one_job_at_a_time(db, qd, monkeypatch):
    import threading

    gate, entered = threading.Event(), threading.Event()

    def slow(k, client, sparse, progress):
        entered.set()
        gate.wait(5)

    monkeypatch.setattr(reindex, "reindex_knowledge", slow)
    kid = default_kb()["id"]
    reindex.start([kid], qd, FakeSparse())
    assert entered.wait(5)
    assert reindex.is_locked(kid) and not reindex.is_locked(kid + 99)
    with pytest.raises(reindex.ReindexBusy):
        reindex.start([kid], qd, FakeSparse())
    gate.set()
    for _ in range(100):
        if not reindex.status()["running"]:
            break
        threading.Event().wait(0.05)
    assert not reindex.is_locked(kid) and reindex.status()["results"][0]["ok"]


@pytest.fixture
def app(db, qd):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    app.extensions.update(client=qd, sparse_embeddings=FakeSparse())
    for bp in (main_bp, api_bp, admin_bp):
        app.register_blueprint(bp)
    return app


def test_api_permissions_status_and_lock(app, db, qd, monkeypatch):
    admin = {"id": 1, "role": "admin", "username": "admin"}
    viewer = add_user(db, "vera", "user")
    editor = add_user(db, "ed", "editor")
    kid = default_kb()["id"]
    access.add_grant(
        "knowledge", kid, "user", 1, "write"
    )  # Admins haben keinen Sonderzugriff
    assert client_for(app, viewer).post(
        f"/api/knowledge/{kid}/reindex"
    ).status_code in (403, 404)
    assert (
        client_for(app, editor).post(f"/api/knowledge/{kid}/reindex").status_code == 404
    )  # kein Schreibrecht
    assert client_for(app, editor).post("/api/reindex-all").status_code == 403
    assert client_for(app, viewer).get("/api/reindex/status").status_code == 200

    # nichts veraltet -> "Alles aktuell"
    assert client_for(app, admin).post("/api/reindex-all").json["started"] is False

    started = []
    monkeypatch.setattr(
        reindex, "start", lambda ids, c, s, wait=False: started.append(ids)
    )
    r = client_for(app, admin).post(f"/api/knowledge/{kid}/reindex")
    assert r.status_code == 202 and started == [[kid]]

    # Sperre: aendernde Aufrufe auf dem Knowledge -> 409
    monkeypatch.setattr(reindex, "is_locked", lambda i: i == kid)
    admin_c = client_for(app, admin)
    assert admin_c.patch(f"/api/knowledge/{kid}", json={"name": "X"}).status_code == 409
    assert (
        admin_c.post("/api/documents/ingest", json={"knowledge_id": kid}).status_code
        == 409
    )
    assert (
        admin_c.get(f"/api/knowledge/{kid}").status_code == 200
    )  # lesen bleibt moeglich


def test_knowledge_list_marks_stale(app, db):
    admin = client_for(app, {"id": 1, "role": "admin", "username": "admin"})
    assert admin.get("/api/knowledge").json[0]["stale"] is False
    model_settings.save_specs(NEW)
    assert admin.get("/api/knowledge").json[0]["stale"] is True


def test_reindex_keeps_uploader_and_upload_date(db, qd, no_enrichment):
    from tests.test_access import add_user

    ed = add_user(db, "ed", "editor")
    ingest_two(qd)
    s = db.get_session()
    from app.core.models import UserDocument

    stamp = None
    for doc in s.query(UserDocument).all():
        doc.uploaded_by = ed["id"]
        stamp = doc.uploaded_at
    s.commit()
    s.close()
    model_settings.save_specs(NEW)
    reindex.reindex_knowledge(default_kb(), qd, FakeSparse())
    payload = scroll_all(qd, "open_knowledge_ui_g1")[0].payload
    assert payload["uploaded_by"] == ed["id"]
    assert payload["uploaded_at"].startswith(stamp.date().isoformat())
    assert (
        payload["file_type"] == "md" and payload["knowledge_id"] == default_kb()["id"]
    )
