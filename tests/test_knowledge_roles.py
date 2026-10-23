import pytest
from flask import Flask
from qdrant_client import QdrantClient

from app.auth import access
from app.auth.user_admin_routes import users_bp
from app.core.models import Group, GroupMember
from app.rag import knowledge
from app.rag import models_runtime as mr
from app.rag.routes import rag_bp
from app.web.admin_routes import admin_bp
from app.web.api_routes import api_bp
from app.web.routes import bp as main_bp
from tests.test_access import add_user, client_for


@pytest.fixture
def app(db, monkeypatch):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    app.extensions["client"] = QdrantClient(":memory:")
    for bp in (main_bp, api_bp, admin_bp, users_bp, rag_bp):
        app.register_blueprint(bp)

    class Emb:
        dimension = 7

    monkeypatch.setattr(mr, "get_embedder", lambda spec: Emb())
    return app


def make_group(db, name, *users):
    s = db.get_session()
    g = Group(name=name)
    s.add(g)
    s.flush()
    for u in users:
        s.add(GroupMember(group_id=g.id, user_id=u["id"]))
    s.commit()
    gid = g.id
    s.close()
    return gid


# --- Rolle "Benutzer": eigenes Knowledge ---------------------------------------------------
def test_a_plain_user_creates_manages_and_deletes_their_own_knowledge(app, db):
    bob = add_user(db, "bob", "user")
    c = client_for(app, bob)
    created = c.post("/api/knowledge", json={"name": "Meins"})
    assert created.status_code == 201, created.json
    kid = created.json["id"]
    assert created.json["can_write"] is True
    assert (
        c.patch(f"/api/knowledge/{kid}", json={"name": "Umbenannt"}).status_code == 200
    )
    assert kid in [k["id"] for k in c.get("/api/knowledge").json]
    assert c.get(f"/api/documents?knowledge_id={kid}").status_code == 200
    assert c.delete(f"/api/knowledge/{kid}").status_code == 200


def test_a_plain_user_cannot_touch_knowledge_of_others(app, db):
    bob = add_user(db, "bob", "user")
    other = knowledge.create_knowledge("Fremd")
    c = client_for(app, bob)
    assert (
        c.patch(f"/api/knowledge/{other['id']}", json={"name": "x"}).status_code == 404
    )
    assert c.delete(f"/api/knowledge/{other['id']}").status_code == 404
    assert (
        c.post(f"/api/documents/ingest?knowledge_id={other['id']}").status_code == 404
    )
    access.add_grant("knowledge", other["id"], "user", bob["id"], "read")  # nur lesen
    assert (
        c.patch(f"/api/knowledge/{other['id']}", json={"name": "x"}).status_code == 404
    )


def test_external_and_sql_sources_stay_admin_only(app, db):
    bob = add_user(db, "bob", "user")
    c = client_for(app, bob)
    assert (
        c.post(
            "/api/knowledge", json={"name": "E", "kind": "external", "source": {}}
        ).status_code
        == 403
    )
    assert (
        c.post(
            "/api/knowledge", json={"name": "S", "kind": "sql", "source": {}}
        ).status_code
        == 403
    )
    assert c.post("/api/knowledge/sql/test", json={"source": {}}).status_code == 403
    assert (
        c.post("/api/knowledge/external/test", json={"source": {}}).status_code == 403
    )


def test_sharing_stays_with_editors_and_admins(app, db):
    bob = add_user(db, "bob", "user")
    c = client_for(app, bob)
    kid = c.post("/api/knowledge", json={"name": "Meins"}).json["id"]
    assert c.get("/api/principals").status_code == 403  # Nutzerliste nicht fuer jeden
    assert c.get(f"/api/knowledge/{kid}/access").status_code == 403


def test_profiles_can_be_chosen_for_own_knowledge_only(app, db):
    bob = add_user(db, "bob", "user")
    c = client_for(app, bob)
    kid = c.post("/api/knowledge", json={"name": "Meins"}).json["id"]
    other = knowledge.create_knowledge("Fremd")
    assert (
        c.put(
            f"/api/knowledge/{kid}/retrieval-profile", json={"profile_id": None}
        ).status_code
        == 200
    )
    assert (
        c.put(
            f"/api/knowledge/{other['id']}/retrieval-profile", json={"profile_id": None}
        ).status_code
        == 404
    )


# --- Rolle "Editor": Knowledge der eigenen Gruppen -------------------------------------------
def test_an_editor_may_edit_all_knowledge_shared_with_their_groups(app, db):
    ed = add_user(db, "ed", "editor")
    gid = make_group(db, "Vertrieb", ed)
    shared = knowledge.create_knowledge("Vertrieb-Wissen")
    access.add_grant(
        "knowledge", shared["id"], "group", gid, "read"
    )  # nur Lese-Freigabe an die Gruppe
    assert access.can_access_knowledge(ed, shared["id"], "write") is True
    assert shared["id"] in access.accessible_knowledge_ids(ed, "write")
    c = client_for(app, ed)
    assert (
        c.patch(f"/api/knowledge/{shared['id']}", json={"name": "Neu"}).status_code
        == 200
    )
    assert (
        next(k for k in c.get("/api/knowledge").json if k["id"] == shared["id"])[
            "can_write"
        ]
        is True
    )


def test_the_group_rule_does_not_reach_other_groups_or_other_roles(app, db):
    ed = add_user(db, "ed", "editor")
    bob = add_user(db, "bob", "user")
    mine = make_group(db, "Mein Team", ed, bob)
    other_team = make_group(db, "Andere", add_user(db, "x", "editor"))
    k_mine = knowledge.create_knowledge("Team")
    k_other = knowledge.create_knowledge("Anderes Team")
    k_public = knowledge.create_knowledge("Oeffentlich")
    access.add_grant("knowledge", k_mine["id"], "group", mine, "read")
    access.add_grant("knowledge", k_other["id"], "group", other_team, "read")
    access.set_access("knowledge", k_public["id"], True, [])
    assert not access.can_access_knowledge(ed, k_other["id"], "write")
    assert not access.can_access_knowledge(
        ed, k_public["id"], "write"
    )  # oeffentlich = nur lesen
    assert access.can_access_knowledge(bob, k_mine["id"], "read")
    assert not access.can_access_knowledge(
        bob, k_mine["id"], "write"
    )  # Rolle "Benutzer": bleibt beim Freigabe-Level


# --- Inhaltsverzeichnis (index.md) im Knowledge ------------------------------------------------------
def test_the_index_is_shown_generated_on_demand_and_refreshable_with_write_access(
    app, db
):
    from app.ingest.index_doc import get_index

    bob = add_user(db, "bob", "user")
    reader = add_user(db, "leser", "user")
    c = client_for(app, bob)
    kid = c.post(
        "/api/knowledge", json={"name": "Vertraege", "description": "Mietvertraege"}
    ).json["id"]

    got = c.get(f"/api/knowledge/{kid}/index").json
    assert (
        got["enabled"] is True
        and "# Inhaltsverzeichnis: Vertraege" in got["markdown"]
        and got["updated_at"]
    )
    assert get_index(kid)[0] == got["markdown"]  # haengt am Knowledge

    # Lesen genuegt fuer die Ansicht, Neu erzeugen braucht Schreibrecht, Fremde sehen nichts
    access.add_grant("knowledge", kid, "user", reader["id"], "read")
    r = client_for(app, reader)
    assert "Vertraege" in r.get(f"/api/knowledge/{kid}/index").json["markdown"]
    assert r.post(f"/api/knowledge/{kid}/index/refresh").status_code == 404
    stranger = client_for(app, add_user(db, "fremd", "user"))
    assert stranger.get(f"/api/knowledge/{kid}/index").status_code == 404
    from app.ingest import index_doc

    index_doc._store(kid, "alt")
    assert (
        "Inhaltsverzeichnis"
        in c.post(f"/api/knowledge/{kid}/index/refresh").json["markdown"]
    )


def test_a_disabled_index_is_reported_and_remote_sources_have_none(app, db):
    from app.core.settings_service import SettingsService

    c = client_for(app, {"id": 1, "role": "admin", "username": "admin"})
    kid = c.post("/api/knowledge", json={"name": "K"}).json["id"]
    s = db.get_session()
    SettingsService(s).set("INGEST_INDEX_ENABLED", "false")
    s.commit()
    s.close()
    got = c.get(f"/api/knowledge/{kid}/index").json
    assert got["enabled"] is False and got["markdown"] is None
    sql = knowledge.create_sql_knowledge(
        "DB", "", {"dialect": "sqlite", "database": "x.db"}
    )
    access.add_grant(
        "knowledge", sql["id"], "user", 1, "write"
    )  # Admins sehen Knowledge nur mit Freigabe
    assert c.get(f"/api/knowledge/{sql['id']}/index").json["available"] is False
    assert c.post(f"/api/knowledge/{sql['id']}/index/refresh").status_code == 409
