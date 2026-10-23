"""Alles, was ein Admin speichert, muss einen Neustart (neue DB-Verbindung) ueberleben."""

import pytest
from flask import Flask

from app.core import database_service
from app.llm import connections as store
from app.llm import registry
from app.rag import model_settings
from app.rag import models_runtime as mr
from app.web.admin_routes import admin_bp
from app.web.api_routes import api_bp
from app.web.routes import bp as main_bp
from tests.test_access import client_for

ADMIN = {"id": 1, "role": "admin", "username": "admin"}


@pytest.fixture(autouse=True)
def no_background_loading(monkeypatch):
    monkeypatch.setattr(mr, "load_in_background", lambda *a, **k: None)
    monkeypatch.setattr(
        registry, "fetch_remote_models", lambda c, force=False: ["m1", "m2"]
    )


@pytest.fixture
def app(db):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    for bp in (main_bp, api_bp, admin_bp):
        app.register_blueprint(bp)
    return app


def restart(db, monkeypatch, tmp_path):
    """Neustart simulieren: neues DatabaseService-Objekt auf derselben Datenbank."""
    url = db.engine.url.render_as_string(hide_password=False)
    monkeypatch.setattr(database_service, "db_service", None)
    return database_service.init_db_service(url)


def test_embedding_and_ingest_settings_survive_restart(app, db, monkeypatch, tmp_path):
    r = client_for(app, ADMIN).put(
        "/api/admin/embedding",
        json={
            "embedding": {"source": "hf", "model": "BAAI/bge-m3"},
            "reranker": {"source": "hf", "model": "BAAI/bge-reranker-v2-m3"},
            "ingest": {"chunk_size": 777, "keep_original": False},
        },
    )
    assert r.status_code == 200
    restart(db, monkeypatch, tmp_path)
    assert model_settings.embedding_spec().model == "BAAI/bge-m3"
    assert model_settings.reranker_spec().model == "BAAI/bge-reranker-v2-m3"
    assert model_settings.ingest_params()[0] == 777


def test_tool_settings_survive_restart_and_never_leak_keys(
    app, db, monkeypatch, tmp_path
):
    admin = client_for(app, ADMIN)
    r = admin.put(
        "/api/admin/tools",
        json={
            "stt": {
                "base_url": "http://stt:5001/v1",
                "model": "parakeet",
                "api_key": "geheim",
            },
            "search": {"url": "http://sx:8080", "categories": "general,news"},
            "pii": {"url": "http://pii:9500/x", "enabled": True, "api_key": "pii-key"},
        },
    )
    assert r.status_code == 200
    assert "geheim" not in r.get_data(as_text=True) and "pii-key" not in r.get_data(
        as_text=True
    )
    assert (
        r.json["stt"]["has_key"]
        and r.json["pii"]["has_key"]
        and r.json["web_search_available"]
    )

    restart(db, monkeypatch, tmp_path)
    body = client_for(app, ADMIN).get("/api/admin/tools").json
    assert (
        body["stt"]["base_url"] == "http://stt:5001/v1"
        and body["search"]["categories"] == "general,news"
    )
    assert body["pii"]["url"] == "http://pii:9500/x" and body["stt"]["has_key"]

    # leeres Key-Feld behaelt den Key, clear_key loescht ihn
    client_for(app, ADMIN).put(
        "/api/admin/tools",
        json={"stt": {"base_url": "http://stt:5001/v1", "model": "p"}},
    )
    assert client_for(app, ADMIN).get("/api/admin/tools").json["stt"]["has_key"] is True
    client_for(app, ADMIN).put(
        "/api/admin/tools",
        json={"stt": {"base_url": "http://stt:5001/v1", "clear_key": True}},
    )
    assert (
        client_for(app, ADMIN).get("/api/admin/tools").json["stt"]["has_key"] is False
    )


def test_model_alias_default_and_connection_survive_restart(
    app, db, monkeypatch, tmp_path
):
    admin = client_for(app, ADMIN)
    conn = store.create_connection("LAN", "http://127.0.0.1:9/v1", "k")
    ref = store.make_base_ref(conn["id"], "m1")
    assert (
        admin.put(
            "/api/models/config",
            json={"ref": ref, "name": "Hausmodell", "system_prompt": "Sei kurz"},
        ).status_code
        == 200
    )
    assert admin.put("/api/models/default", json={"ref": ref}).status_code == 200
    assert (
        admin.patch(
            f"/api/connections/{conn['id']}", json={"name": "LAN neu"}
        ).status_code
        == 200
    )

    restart(db, monkeypatch, tmp_path)
    cfg = store.get_model_config(ref)
    assert cfg["name"] == "Hausmodell" and cfg["system_prompt"] == "Sei kurz"
    assert (
        store.get_model_default() == ref
        if hasattr(store, "get_model_default")
        else registry.default_model_ref() == ref
    )
    assert store.get_connection(conn["id"])["name"] == "LAN neu"


def test_reads_do_not_leave_sessions_open(app, db):
    """Jeder Lesezugriff gibt seine Verbindung zurueck (sonst sperrt SQLite bzw. der Postgres-Pool laeuft voll)."""
    admin = client_for(app, ADMIN)
    pool = db.engine.pool
    for _ in range(40):
        assert admin.get("/api/admin/tools").status_code == 200
        assert admin.get("/api/admin/embedding").status_code == 200
        assert admin.get("/api/knowledge").status_code == 200
    assert pool.checkedout() == 0


def test_live_settings_see_changes_made_elsewhere(db):
    from app.core.settings_service import LiveSettingsService, SettingsService

    live = LiveSettingsService()
    session = db.get_session()
    SettingsService(session).set("SEARCH_SEARXNG_CATEGORIES", "news")
    session.close()
    assert (
        live.get("SEARCH_SEARXNG_CATEGORIES") == "news"
    )  # kein veralteter Cache einer Dauer-Session
    live.set("SEARCH_SEARXNG_CATEGORIES", "it")
    assert (
        live.get_int("NICHT_DA", 5) == 5
        and live.get("SEARCH_SEARXNG_CATEGORIES") == "it"
    )
