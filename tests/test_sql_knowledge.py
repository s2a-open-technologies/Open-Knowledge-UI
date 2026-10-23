import sqlite3

import pytest
from flask import Flask

from app.api.admin_routes import admin_bp
from app.api.api_routes import api_bp
from app.api.routes import bp as main_bp
from app.auth import access
from app.knowledge import service as knowledge
from tests.test_access import add_user, client_for

ADMIN = {"id": 1, "role": "admin", "username": "admin"}


@pytest.fixture
def app(db):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    for bp in (main_bp, api_bp, admin_bp):
        app.register_blueprint(bp)
    return app


@pytest.fixture
def shop(sql_worker):
    con = sqlite3.connect(sql_worker / "shop.db")
    con.executescript(
        "create table items (id integer primary key, name text, salary integer); insert into items (name, salary) values ('Stuhl', 1);"
    )
    con.commit()
    con.close()
    return sql_worker


def body(**over):
    src = {
        "dialect": "sqlite",
        "database": "shop.db",
        "tables": ["items"],
        "blocked_columns": {"items": ["salary"]},
    }
    src.update(over)
    return {"name": "Shop", "description": "Test", "kind": "sql", "source": src}


def create(app, payload, user=ADMIN):
    return client_for(app, user).post("/api/knowledge", json=payload)


def test_admin_creates_a_sql_knowledge_without_any_collection(app, db, shop):
    r = create(app, body())
    assert r.status_code == 201, r.json
    k = r.json
    assert k["kind"] == "sql" and k["collection_name"].startswith("sql_")
    assert k["embedding_dimension"] is None and k["stale"] is False
    assert k["source_config"]["blocked_columns"] == {"items": ["salary"]}
    assert knowledge.get_knowledge(k["id"])["kind"] == "sql"


def test_the_password_never_leaks_but_is_stored(app, db, shop, monkeypatch):
    from app.knowledge.sql import config as sqlcfg

    # Verbindungsprobe ohne echten Server: der Worker-Test wird ersetzt
    monkeypatch.setattr(
        "app.api.api_routes._sql_probe",
        lambda cfg: {"version": "x", "tables": [], "write_privilege": False},
    )
    r = create(
        app,
        body(
            dialect="postgresql",
            host="db.lan",
            port=5432,
            database="d",
            user="u",
            password="geheim",
        ),
    )
    assert r.status_code == 201, r.json
    assert "geheim" not in r.get_data(as_text=True)
    assert "geheim" not in client_for(app, ADMIN).get("/api/knowledge").get_data(
        as_text=True
    )
    assert r.json["source_config"]["has_password"] is True
    assert (
        sqlcfg.parse_config(
            knowledge.get_knowledge(r.json["id"])["source_config"]
        ).password
        == "geheim"
    )


def test_only_admins_create_or_test_sql_sources(app, db, shop):
    ed = add_user(db, "ed", "editor")
    assert create(app, body(), ed).status_code == 403
    assert (
        client_for(app, ed)
        .post("/api/knowledge/sql/test", json={"source": body()["source"]})
        .status_code
        == 403
    )


def test_invalid_sources_are_refused_with_a_reason(app, db, shop):
    r = create(
        app, body(host="169.254.169.254", dialect="postgresql", database="d", user="u")
    )
    assert r.status_code == 400 and "lokal" in r.json["error"].lower()
    r = create(app, body(database="gibt_es_nicht.db"))
    assert r.status_code == 400 and "nicht verfügbar" in r.json["error"]


def test_connection_test_lists_tables_and_warns_about_write_rights(app, db, shop):
    r = client_for(app, ADMIN).post(
        "/api/knowledge/sql/test", json={"source": body()["source"]}
    )
    assert r.status_code == 200 and r.json["success"] is True
    assert r.json["tables"][0]["name"] == "items" and r.json["write_privilege"] is False


def test_worker_down_gives_a_clear_503_without_address(app, db, shop):
    from app.agent.toolworker import client as wc

    def down(*a, **k):
        raise ConnectionError("http://tool-worker-sql:8090 refused")

    wc.set_override(wc.ToolWorkerClient("http://x", "t", down), "sql")
    try:
        r = client_for(app, ADMIN).post(
            "/api/knowledge/sql/test", json={"source": body()["source"]}
        )
    finally:
        wc.set_override(None, "sql")
    assert r.status_code == 503 and "tool-worker-sql" not in r.get_data(as_text=True)


def test_sql_knowledge_is_read_only_like_external_sources(app, db, shop):
    k = create(app, body()).json
    admin = client_for(app, ADMIN)
    q = f"?knowledge_id={k['id']}"
    assert admin.post(f"/api/documents/ingest{q}").status_code == 409
    assert admin.post(f"/api/knowledge/{k['id']}/reindex").status_code == 409
    assert admin.get(f"/api/vectordb{q}").status_code == 409


def test_visibility_follows_the_usual_grants(app, db, shop):
    k = create(app, body()).json
    ed = add_user(db, "ed", "editor")
    assert k["id"] not in [
        x["id"] for x in client_for(app, ed).get("/api/knowledge").json
    ]
    access.add_grant("knowledge", k["id"], "user", ed["id"], "read")
    assert k["id"] in [x["id"] for x in client_for(app, ed).get("/api/knowledge").json]


def test_changing_the_source_keeps_the_password_for_the_same_target(
    app, db, shop, monkeypatch
):
    from app.knowledge.sql import config as sqlcfg

    monkeypatch.setattr(
        "app.api.api_routes._sql_probe",
        lambda cfg: {"version": "x", "tables": [], "write_privilege": False},
    )
    k = create(
        app,
        body(
            dialect="postgresql",
            host="db.lan",
            port=5432,
            database="d",
            user="u",
            password="geheim",
        ),
    ).json
    src = {
        "dialect": "postgresql",
        "host": "db.lan",
        "port": 5432,
        "database": "d",
        "user": "u",
        "password": "",
        "tables": ["a"],
    }
    r = client_for(app, ADMIN).patch(f"/api/knowledge/{k['id']}", json={"source": src})
    assert r.status_code == 200, r.json
    stored = sqlcfg.parse_config(knowledge.get_knowledge(k["id"])["source_config"])
    assert stored.password == "geheim" and stored.tables == ("a",)
    src["host"] = "other.lan"
    r = client_for(app, ADMIN).patch(f"/api/knowledge/{k['id']}", json={"source": src})
    assert r.status_code == 200
    # anderes Ziel + leeres Passwort: das gespeicherte Passwort wandert NICHT mit
    assert (
        sqlcfg.parse_config(knowledge.get_knowledge(k["id"])["source_config"]).password
        == ""
    )
