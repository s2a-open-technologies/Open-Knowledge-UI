import json

import pytest
from flask import Flask

from app.api import register_blueprints
from app.knowledge import service as knowledge
from app.llm import connections as store
from app.rag import model_settings
from app.rag import models_runtime as mr
from tests.test_access import add_user, client_for


@pytest.fixture(autouse=True)
def clean():
    mr.reset_for_tests()
    yield
    mr.reset_for_tests()


@pytest.fixture
def app(db):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    register_blueprints(app)
    return app


ADMIN = {"id": 1, "role": "admin", "username": "admin"}


class FakeEmbedder:
    dimension = 5


def test_fresh_install_defaults_to_huggingface(db):
    assert model_settings.embedding_spec().source == "hf"
    assert model_settings.reranker_spec().model == "BAAI/bge-reranker-base"
    default = knowledge.get_knowledge(None)
    assert json.loads(default["embedding_spec"])["source"] == "hf"
    assert default["index_signature"]


def test_only_admins_see_and_change_models(app, db):
    user = add_user(db, "bob", "editor")
    assert client_for(app, user).get("/api/admin/embedding").status_code == 403
    assert client_for(app, user).put("/api/admin/embedding", json={}).status_code == 403
    assert client_for(app, ADMIN).get("/api/admin/embedding").status_code == 200


def test_save_validates_and_switches_model(app, db, monkeypatch):
    loaded = []
    monkeypatch.setattr(
        mr, "load_in_background", lambda kind, spec: loaded.append((kind, spec.model))
    )
    admin = client_for(app, ADMIN)

    bad = admin.put(
        "/api/admin/embedding", json={"embedding": {"source": "endpoint", "model": "m"}}
    )
    assert bad.status_code == 400
    # fastembed bleibt waehlbar (z. B. um zu einem bestehenden Index zurueckzukehren)
    back = admin.put(
        "/api/admin/embedding",
        json={
            "embedding": {
                "source": "fastembed",
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            }
        },
    )
    assert back.status_code == 200 and back.json["embedding"]["source"] == "fastembed"
    assert (
        admin.put(
            "/api/admin/embedding", json={"ingest": {"chunk_size": 50}}
        ).status_code
        == 400
    )
    assert (
        admin.put(
            "/api/admin/embedding",
            json={"ingest": {"chunk_size": 500, "chunk_overlap": 600}},
        ).status_code
        == 400
    )
    assert (
        admin.put("/api/admin/embedding", json={"ingest": {"nope": 1}}).status_code
        == 400
    )

    r = admin.put(
        "/api/admin/embedding",
        json={
            "embedding": {"source": "hf", "model": "BAAI/bge-m3"},
            "reranker": {"source": "hf", "model": "BAAI/bge-reranker-v2-m3"},
            "ingest": {"chunk_size": 800, "chunk_overlap": 100, "keep_original": False},
        },
    )
    assert r.status_code == 200
    assert loaded[-2:] == [
        ("embedding", "BAAI/bge-m3"),
        ("reranker", "BAAI/bge-reranker-v2-m3"),
    ]
    body = r.json
    assert (
        body["embedding"]["model"] == "BAAI/bge-m3"
        and body["ingest"]["chunk_size"] == 800
    )
    assert body["ingest"]["keep_original"] is False
    # Das Standard-Knowledge wurde mit dem alten Modell indexiert -> veraltet
    assert [k["name"] for k in body["stale"]] == ["Standard"]


def test_existing_knowledge_keeps_its_own_model_until_reindexed(db):
    k = knowledge.create_knowledge("Alt", dimension=4)
    model_settings.save_specs(mr.ModelSpec("hf", "neues/modell"))
    fresh = knowledge.get_knowledge(k["id"])
    assert model_settings.knowledge_spec(fresh).model != "neues/modell"
    assert model_settings.is_stale(fresh)
    model_settings.stamp_knowledge(k["id"], mr.ModelSpec("hf", "neues/modell"), 9)
    fresh = knowledge.get_knowledge(k["id"])
    assert not model_settings.is_stale(fresh) and fresh["embedding_dimension"] == 9


def test_chunk_setting_change_marks_signed_knowledge_stale(db, app):
    k = knowledge.create_knowledge("Neu", dimension=4)
    assert not model_settings.is_stale(knowledge.get_knowledge(k["id"]))
    client_for(app, ADMIN).put(
        "/api/admin/embedding", json={"ingest": {"chunk_size": 700}}
    )
    assert model_settings.is_stale(knowledge.get_knowledge(k["id"]))


def test_legacy_knowledge_without_signature_is_not_stale_for_same_model(db):
    legacy = knowledge.get_knowledge(None)
    legacy.update(index_signature=None)
    assert not model_settings.is_stale(legacy)


def test_test_endpoint_reports_dimension_and_errors(app, db, monkeypatch):
    admin = client_for(app, ADMIN)
    monkeypatch.setattr(mr, "get_embedder", lambda spec: FakeEmbedder())
    ok = admin.post(
        "/api/admin/embedding/test",
        json={"kind": "embedding", "spec": {"source": "hf", "model": "x/y"}},
    )
    assert ok.status_code == 200 and ok.json["dimension"] == 5

    def boom(spec):
        raise mr.ModelLoadError("kein Netz")

    monkeypatch.setattr(mr, "get_embedder", boom)
    bad = admin.post(
        "/api/admin/embedding/test",
        json={"kind": "embedding", "spec": {"source": "hf", "model": "x/y"}},
    )
    assert bad.status_code == 502 and "kein Netz" in bad.json["error"]
    assert (
        admin.post(
            "/api/admin/embedding/test", json={"kind": "x", "spec": {}}
        ).status_code
        == 400
    )


def test_endpoint_spec_needs_an_existing_connection(app, db):
    admin = client_for(app, ADMIN)
    r = admin.put(
        "/api/admin/embedding",
        json={"embedding": {"source": "endpoint", "model": "m", "connection_id": 99}},
    )
    assert r.status_code == 400
    conn = store.create_connection("emb", "http://127.0.0.1:9/v1", "")
    r = admin.put(
        "/api/admin/embedding",
        json={
            "embedding": {
                "source": "endpoint",
                "model": "m",
                "connection_id": conn["id"],
            }
        },
    )
    assert r.status_code == 200 and r.json["embedding"]["connection_id"] == conn["id"]
