import pytest
from flask import Flask
from werkzeug.security import generate_password_hash

from app.api.admin_routes import admin_bp
from app.api.api_routes import api_bp
from app.api.routes import bp as main_bp
from app.api.users import users_bp
from app.auth import access, user_admin
from app.auth.user_admin import AdminError
from app.core.models import User
from app.knowledge import service as knowledge
from app.llm import connections as store


def add_user(db, username, role="user", password="pw"):
    s = db.get_session()
    u = User(
        username=username,
        display_name=username,
        password_hash=generate_password_hash(password),
        role=role,
        is_active=True,
    )
    s.add(u)
    s.commit()
    uid = u.id
    s.close()
    return {"id": uid, "role": role, "username": username}


@pytest.fixture
def admin(db):
    return {"id": 1, "role": "admin", "username": "admin"}


@pytest.fixture
def app(db):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    for bp in (main_bp, api_bp, admin_bp, users_bp):
        app.register_blueprint(bp)
    return app


def client_for(app, user):
    c = app.test_client()
    with c.session_transaction() as s:
        s["user_id"] = user["id"]
    return c


# --- Knowledge-Zugriff ------------------------------------------------------


def test_private_knowledge_needs_a_grant(db, admin):
    kb = knowledge.create_knowledge("Privat")
    alice = add_user(db, "alice", "editor")
    assert not access.can_access_knowledge(alice, kb["id"], "read")
    # Auch Admins sehen fremdes, nicht freigegebenes Knowledge nicht
    assert not access.can_access_knowledge(admin, kb["id"], "read")
    assert kb["id"] not in access.accessible_knowledge_ids(admin)

    access.add_grant("knowledge", kb["id"], "user", alice["id"], "read")
    assert access.can_access_knowledge(alice, kb["id"], "read")
    assert not access.can_access_knowledge(alice, kb["id"], "write")  # read != write

    access.add_grant("knowledge", kb["id"], "user", alice["id"], "write")
    assert access.can_access_knowledge(alice, kb["id"], "write")


def test_public_knowledge_is_readable_but_not_writable(db):
    default = knowledge.get_knowledge(None)
    assert default["is_public"] is True
    bob = add_user(db, "bob")
    assert access.can_access_knowledge(bob, default["id"], "read")
    assert not access.can_access_knowledge(bob, default["id"], "write")


def test_group_membership_grants_access_and_removal_revokes_it(db):
    kb = knowledge.create_knowledge("Team")
    carol = add_user(db, "carol")
    group = user_admin.create_group("Recht")
    access.set_access(
        "knowledge",
        kb["id"],
        False,
        [{"principal_type": "group", "principal_id": group["id"], "access": "read"}],
    )
    assert not access.can_access_knowledge(carol, kb["id"])

    user_admin.set_group_members(group["id"], [carol["id"]])
    assert access.can_access_knowledge(carol, kb["id"])
    assert kb["id"] in access.accessible_knowledge_ids(carol)

    user_admin.set_group_members(group["id"], [])
    assert not access.can_access_knowledge(carol, kb["id"])


def test_unknown_knowledge_is_never_accessible(db, admin):
    assert not access.can_access_knowledge(admin, 999)


def test_set_access_replaces_grants_dedupes_and_validates(db):
    kb = knowledge.create_knowledge("X")
    u = add_user(db, "dora")
    access.set_access(
        "knowledge",
        kb["id"],
        False,
        [
            {"principal_type": "user", "principal_id": u["id"], "access": "read"},
            {"principal_type": "user", "principal_id": u["id"], "access": "write"},
        ],
    )
    grants = access.get_access("knowledge", kb["id"])["grants"]
    assert [(g["principal_id"], g["access"]) for g in grants] == [(u["id"], "write")]

    access.set_access("knowledge", kb["id"], True, [])
    assert access.get_access("knowledge", kb["id"]) == {"is_public": True, "grants": []}

    for bad in (
        [{"principal_type": "user", "principal_id": 999, "access": "read"}],
        [{"principal_type": "alien", "principal_id": 1, "access": "read"}],
        [{"principal_type": "user", "principal_id": u["id"], "access": "root"}],
        [{"principal_type": "user", "principal_id": "x", "access": "read"}],
    ):
        with pytest.raises(ValueError):
            access.set_access("knowledge", kb["id"], False, bad)
    assert access.set_access("knowledge", 999, False, []) is False


# --- Modell-Zugriff ---------------------------------------------------------


@pytest.fixture
def conn(db):
    return store.create_connection("C", "http://x/v1", "k")


def test_unconfigured_base_model_is_public(db, conn):
    eve = add_user(db, "eve")
    assert access.can_use_model(eve, store.make_base_ref(conn["id"], "qwen"))


def test_private_override_and_clone_need_grants(db, conn):
    eve = add_user(db, "eve")
    ref = store.make_base_ref(conn["id"], "qwen")
    store.save_model_config(ref, {"is_public": False})
    assert not access.can_use_model(eve, ref)

    clone = store.clone_model(ref, "Klon")
    assert clone["is_public"] is False  # Klone starten privat
    assert not access.can_use_model(eve, clone["ref"])

    rid = access.model_resource_id(clone["ref"])
    access.set_access(
        "model",
        rid,
        False,
        [{"principal_type": "user", "principal_id": eve["id"], "access": "read"}],
    )
    assert access.can_use_model(eve, clone["ref"])
    assert not access.can_use_model(eve, ref)  # Basis bleibt gesperrt
    assert not access.can_use_model(eve, "clone:999")


def test_admin_can_use_everything(db, conn, admin):
    ref = store.make_base_ref(conn["id"], "qwen")
    store.save_model_config(ref, {"is_public": False})
    assert access.can_use_model(admin, ref)


# --- Aufraeumen -------------------------------------------------------------


def test_grants_are_removed_with_their_principal_or_resource(db, conn):
    kb = knowledge.create_knowledge("Y")
    u = add_user(db, "finn")
    g = user_admin.create_group("G")
    ref = store.make_base_ref(conn["id"], "qwen")
    clone = store.clone_model(ref, "K")
    grants = [
        {"principal_type": "user", "principal_id": u["id"], "access": "read"},
        {"principal_type": "group", "principal_id": g["id"], "access": "read"},
    ]
    access.set_access("knowledge", kb["id"], False, grants)
    access.set_access("model", clone["id"], False, grants)

    user_admin.delete_group(g["id"])
    assert [
        x["principal_type"] for x in access.get_access("knowledge", kb["id"])["grants"]
    ] == ["user"]

    user_admin.delete_user(u["id"], actor_id=1)
    assert access.get_access("knowledge", kb["id"])["grants"] == []
    assert access.get_access("model", clone["id"])["grants"] == []

    # Ressource weg -> Grants weg
    access.add_grant("knowledge", kb["id"], "user", 1, "write")
    knowledge.delete_knowledge(kb["id"])
    from app.core.models import ResourceGrant

    s = store.get_db_service().get_session()
    assert (
        s.query(ResourceGrant)
        .filter_by(resource_type="knowledge", resource_id=kb["id"])
        .count()
        == 0
    )
    s.close()


# --- Benutzerverwaltung -----------------------------------------------------


def test_create_user_validation(db):
    ok = user_admin.create_user("Anna.K", "Anna", "geheim", "editor")
    assert ok["role"] == "editor" and "password" not in str(ok)
    for args, status in [
        (
            ("anna.k", "", "geheim", "user"),
            409,
        ),  # Duplikat ohne Beachtung der Schreibweise
        (("a b", "", "geheim", "user"), 400),
        (("ab", "", "geheim", "user"), 400),
        (("neu", "", "abc", "user"), 400),  # Passwort zu kurz
        (("neu", "", "geheim", "root"), 400),
    ]:
        with pytest.raises(AdminError) as e:
            user_admin.create_user(*args)
        assert e.value.status == status


def test_last_admin_and_self_protection(db):
    with pytest.raises(AdminError, match="letzte"):
        user_admin.update_user(1, actor_id=99, data={"role": "user"})
    with pytest.raises(AdminError, match="letzte"):
        user_admin.update_user(1, actor_id=99, data={"is_active": False})
    with pytest.raises(AdminError, match="letzte"):
        user_admin.delete_user(1, actor_id=99)
    with pytest.raises(AdminError, match="selbst"):
        user_admin.delete_user(1, actor_id=1)
    with pytest.raises(AdminError, match="Eigene"):
        user_admin.update_user(1, actor_id=1, data={"role": "editor"})

    second = add_user(db, "zweiter", "admin")
    # Mit zweitem Admin darf der erste herabgestuft werden - aber nicht von sich selbst.
    assert (
        user_admin.update_user(1, actor_id=second["id"], data={"role": "editor"})[
            "role"
        ]
        == "editor"
    )


def test_update_user_resets_password_and_name(db):
    u = add_user(db, "gina")
    user_admin.update_user(
        u["id"], 1, {"display_name": "Gina G.", "password": "neuespw"}
    )
    from app.auth.users import authenticate

    assert authenticate("gina", "neuespw")["display_name"] == "Gina G."
    assert authenticate("gina", "pw") is None


def test_delete_user_removes_chats_and_memberships(db):
    u = add_user(db, "hans")
    sess = db.create_session(user_id=u["id"])
    db.save_message(sess.id, "user", "hallo")
    g = user_admin.create_group("H")
    user_admin.set_group_members(g["id"], [u["id"]])

    removed = user_admin.delete_user(u["id"], actor_id=1)

    assert removed == [sess.id]
    assert db.get_chat_session(sess.id) is None
    assert user_admin.list_groups()[0]["member_ids"] == []


def test_group_validation(db):
    g = user_admin.create_group("Alpha", "x")
    with pytest.raises(AdminError) as e:
        user_admin.create_group("alpha")
    assert e.value.status == 409
    with pytest.raises(AdminError):
        user_admin.create_group("  ")
    with pytest.raises(AdminError, match="existiert nicht"):
        user_admin.set_group_members(g["id"], [999])
    assert user_admin.update_group(g["id"], {"name": "Beta"})["name"] == "Beta"


# --- API-Durchsetzung -------------------------------------------------------


def test_editor_creates_private_knowledge_only_they_can_manage(app, db):
    alice = add_user(db, "alice", "editor")
    bob = add_user(db, "bob", "editor")
    user = add_user(db, "uwe", "user")
    ca, cb, cu = (client_for(app, x) for x in (alice, bob, user))

    # Qdrant ist im Test nicht verfuegbar: die Anlage der Collection scheitert,
    # deshalb direkt ueber die Fachlogik anlegen und den Grant wie die Route setzen.
    kb = knowledge.create_knowledge("Alices")
    access.add_grant("knowledge", kb["id"], "user", alice["id"], "write")
    kid = kb["id"]

    ids = lambda c: {k["id"] for k in c.get("/api/knowledge").json}
    assert kid in ids(ca)
    assert kid not in ids(cb) and kid not in ids(cu)

    assert ca.get(f"/api/knowledge/{kid}").json["can_write"] is True
    assert cb.get(f"/api/knowledge/{kid}").status_code == 404
    assert cu.get(f"/api/knowledge/{kid}").status_code == 404

    # Verwaltende Endpunkte: fremder Editor bekommt 404, kein Datenleck.
    for method, path in [
        ("get", f"/api/documents?knowledge_id={kid}"),
        ("get", f"/api/data-files?knowledge_id={kid}"),
        ("get", f"/api/vectordb?knowledge_id={kid}"),
        ("post", f"/api/documents/ingest?knowledge_id={kid}"),
        ("post", f"/api/documents/upload?knowledge_id={kid}"),
        ("patch", f"/api/knowledge/{kid}"),
        ("delete", f"/api/knowledge/{kid}"),
        ("get", f"/api/knowledge/{kid}/access"),
        ("put", f"/api/knowledge/{kid}/access"),
    ]:
        res = getattr(cb, method)(path)
        assert res.status_code == 404, f"{method} {path} -> {res.status_code}"
        # Der Besitzer kommt durch die Rechtepruefung (egal was danach passiert).
    assert ca.get(f"/api/documents?knowledge_id={kid}").status_code == 200
    assert ca.get(f"/api/knowledge/{kid}/access").json["is_public"] is False


def test_sharing_with_a_group_grants_read_but_not_write(app, db):
    alice = add_user(db, "alice", "editor")
    bob = add_user(db, "bob", "editor")
    kb = knowledge.create_knowledge("Geteilt")
    access.add_grant("knowledge", kb["id"], "user", alice["id"], "write")
    group = user_admin.create_group("Team")
    user_admin.set_group_members(group["id"], [bob["id"]])
    ca, cb = client_for(app, alice), client_for(app, bob)

    res = ca.put(
        f"/api/knowledge/{kb['id']}/access",
        json={
            "is_public": False,
            "grants": [
                {
                    "principal_type": "group",
                    "principal_id": group["id"],
                    "access": "read",
                }
            ],
        },
    )
    assert res.status_code == 200 and res.json["grants"][0]["name"] == "Team"

    # Rolle "Benutzer": bleibt beim vergebenen Level (nur lesen)
    carol = add_user(db, "carol", "user")
    user_admin.set_group_members(group["id"], [bob["id"], carol["id"]])
    cc = client_for(app, carol)
    assert cc.get(f"/api/knowledge/{kb['id']}").json["can_write"] is False
    assert cc.get(f"/api/documents?knowledge_id={kb['id']}").status_code == 200  # lesen
    assert (
        cc.post(f"/api/documents/ingest?knowledge_id={kb['id']}").status_code == 404
    )  # schreiben
    # Editor: alles, was einer seiner Gruppen freigegeben ist, darf er bearbeiten (auch bei Lese-Freigabe)
    assert cb.get(f"/api/knowledge/{kb['id']}").json["can_write"] is True


def test_access_endpoint_rejects_bad_principals(app, db):
    alice = add_user(db, "alice", "editor")
    kb = knowledge.create_knowledge("K")
    access.add_grant("knowledge", kb["id"], "user", alice["id"], "write")
    res = client_for(app, alice).put(
        f"/api/knowledge/{kb['id']}/access",
        json={
            "grants": [
                {"principal_type": "user", "principal_id": 999, "access": "read"}
            ]
        },
    )
    assert res.status_code == 400


def test_user_and_group_admin_api_is_admin_only(app, db):
    editor = add_user(db, "ed", "editor")
    ce = client_for(app, editor)
    ca = client_for(app, {"id": 1})
    for method, path in [
        ("get", "/api/users"),
        ("post", "/api/users"),
        ("get", "/api/groups"),
        ("post", "/api/groups"),
        ("get", "/api/models/access?ref=base:1:x"),
    ]:
        assert (
            getattr(ce, method)(path, json={}).status_code == 403
            if method == "post"
            else getattr(ce, method)(path).status_code == 403
        )
    created = ca.post(
        "/api/users", json={"username": "neu", "password": "geheim", "role": "user"}
    )
    assert created.status_code == 201
    assert (
        ca.post(
            "/api/users", json={"username": "neu", "password": "geheim"}
        ).status_code
        == 409
    )
    assert (
        ca.patch("/api/users/1", json={"role": "user"}).status_code == 409
    )  # sich selbst
    assert ca.delete("/api/users/1").status_code == 409
    assert (
        ce.get("/api/principals").status_code == 200
    )  # Editoren brauchen Namen zum Freigeben


def test_everyone_can_change_their_own_password(app, db):
    u = add_user(db, "plain", "user")
    c = client_for(app, u)
    assert (
        c.post(
            "/api/auth/change-password",
            json={"old_password": "x", "new_password": "neuespw"},
        ).status_code
        == 403
    )
    assert (
        c.post(
            "/api/auth/change-password",
            json={"old_password": "pw", "new_password": "ab"},
        ).status_code
        == 400
    )
    assert (
        c.post(
            "/api/auth/change-password",
            json={"old_password": "pw", "new_password": "neuespw"},
        ).status_code
        == 200
    )


def test_models_listing_respects_model_access(app, db, conn, monkeypatch):
    from app.llm import registry

    monkeypatch.setattr(
        registry, "fetch_remote_models", lambda c, force=False: ["open", "secret"]
    )
    store.save_model_config(
        store.make_base_ref(conn["id"], "secret"), {"is_public": False}
    )
    u = add_user(db, "plain", "user")
    names = {
        m["model_id"]
        for m in client_for(app, u).get("/api/models").json["models"]
        if m["connection_name"] == "C"
    }
    assert names == {"open"}
    admin_names = {
        m["model_id"]
        for m in client_for(app, {"id": 1}).get("/api/models").json["models"]
        if m["connection_name"] == "C"
    }
    assert admin_names == {"open", "secret"}


def test_model_access_endpoint_creates_config_for_plain_base_model(app, db, conn):
    ca = client_for(app, {"id": 1})
    ref = store.make_base_ref(conn["id"], "qwen")
    assert ca.get(f"/api/models/access?ref={ref}").json == {
        "is_public": True,
        "grants": [],
    }
    u = add_user(db, "plain")
    res = ca.put(
        "/api/models/access",
        json={
            "ref": ref,
            "is_public": False,
            "grants": [
                {"principal_type": "user", "principal_id": u["id"], "access": "read"}
            ],
        },
    )
    assert res.status_code == 200 and res.json["is_public"] is False
    assert access.can_use_model(u, ref)
    assert not access.can_use_model(add_user(db, "other"), ref)


# --- Chat: Retrieval-Scope --------------------------------------------------


class FakeAgent:
    async def astream_events(self, *a, **k):
        if False:
            yield None


@pytest.fixture
def chat_env(app, db, conn, monkeypatch):
    """/chat ohne LLM: ein Fake-Agent, der Scope wird mitgeschrieben."""
    from app.api import routes

    seen = []
    monkeypatch.setattr(routes, "set_scope", lambda scope: seen.append(scope))
    monkeypatch.setattr(routes, "get_agent", lambda resolved, tools: FakeAgent())
    monkeypatch.setattr(routes, "resources_ready", lambda: True)
    app.extensions["agent_tools"] = []
    return seen


def chat(client, **body):
    res = client.post("/chat", json={"message": "Hallo", **body})
    res.get_data()  # Stream bis zum Ende lesen
    return res


def test_chat_scope_only_contains_readable_knowledge(app, db, conn, chat_env):
    mine = knowledge.create_knowledge("Meins")
    foreign = knowledge.create_knowledge("Fremd")
    alice = add_user(db, "alice", "editor")
    access.add_grant("knowledge", mine["id"], "user", alice["id"], "read")
    default = knowledge.get_knowledge(None)
    ref = store.make_base_ref(conn["id"], "m")
    c = client_for(app, alice)

    # Nutzer fordert ausdruecklich auch das fremde Knowledge an.
    chat(c, model=ref, knowledge_ids=[mine["id"], foreign["id"], default["id"]])
    assert set(chat_env[-1].collections) == {
        mine["collection_name"],
        "open_knowledge_ui",
    }

    chat(c, model=ref)  # None = alle lesbaren
    assert foreign["collection_name"] not in chat_env[-1].collections

    chat(c, model=ref, knowledge_ids=[foreign["id"]])
    assert chat_env[-1].collections == ()


def test_chat_denies_models_without_access_and_never_leaks_knowledge(
    app, db, conn, chat_env
):
    ref = store.make_base_ref(conn["id"], "geheim")
    store.save_model_config(ref, {"is_public": False})
    u = add_user(db, "plain")
    res = chat(client_for(app, u), model=ref)
    assert "Kein Zugriff" in res.get_data(as_text=True)
    assert chat_env == []  # weder Agent noch Scope wurden angefasst


def test_model_knowledge_is_searchable_without_own_grant(app, db, conn, chat_env):
    secret = knowledge.create_knowledge("Nur fuer das Modell")
    ref = store.make_base_ref(conn["id"], "team")
    store.save_model_config(ref, {"knowledge_ids": [secret["id"]], "is_public": True})
    u = add_user(db, "plain")  # kein Grant auf das Knowledge

    chat(
        client_for(app, u), model=ref, knowledge_ids=[]
    )  # Request waehlt bewusst nichts
    assert chat_env[-1].collections == (secret["collection_name"],)


def test_manager_cannot_lock_themselves_out_by_saving_a_list_without_themselves(
    app, db
):
    alice = add_user(db, "alice", "editor")
    bob = add_user(db, "bob", "editor")
    kb = knowledge.create_knowledge("Meins")
    access.add_grant("knowledge", kb["id"], "user", alice["id"], "write")
    ca = client_for(app, alice)

    # Liste ohne Alice selbst (nur Bob) -> Alice behaelt trotzdem Schreibrecht.
    res = ca.put(
        f"/api/knowledge/{kb['id']}/access",
        json={
            "is_public": False,
            "grants": [
                {"principal_type": "user", "principal_id": bob["id"], "access": "read"}
            ],
        },
    )
    assert res.status_code == 200
    assert access.can_access_knowledge(alice, kb["id"], "write")
    assert access.can_access_knowledge(bob, kb["id"], "read")
    assert (
        ca.put(
            f"/api/knowledge/{kb['id']}/access", json={"is_public": False, "grants": []}
        ).status_code
        == 200
    )
    assert access.can_access_knowledge(alice, kb["id"], "write")
    assert not access.can_access_knowledge(bob, kb["id"], "read")


def test_admin_keeps_write_access_when_saving_a_list_without_themselves(app, db):
    kb = knowledge.create_knowledge("Admin-Sicht")
    access.add_grant(
        "knowledge", kb["id"], "user", 1, "write"
    )  # Admins sehen Knowledge nur mit Freigabe
    ca = client_for(app, {"id": 1})
    res = ca.put(
        f"/api/knowledge/{kb['id']}/access", json={"is_public": False, "grants": []}
    )
    assert res.status_code == 200
    # Wer verwaltet, sperrt sich beim Speichern nicht aus (Admins haben keinen Sonderzugriff mehr)
    assert [
        g["principal_id"] for g in access.get_access("knowledge", kb["id"])["grants"]
    ] == [1]
    assert access.can_access_knowledge({"id": 1, "role": "admin"}, kb["id"], "write")
