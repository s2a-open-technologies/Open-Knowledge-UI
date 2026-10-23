import pytest
from flask import Flask
from werkzeug.security import check_password_hash, generate_password_hash

from app.auth import users
from app.core.models import ChatSession, User
from app.web.admin_routes import admin_bp
from app.web.api_routes import api_bp
from app.web.routes import bp as main_bp


@pytest.fixture(autouse=True)
def clean_throttle():
    users.reset_throttle()
    yield
    users.reset_throttle()


def add_user(db, username, password, role="user", active=True):
    s = db.get_session()
    s.add(
        User(
            username=username,
            display_name=username,
            password_hash=generate_password_hash(password),
            role=role,
            is_active=active,
        )
    )
    s.commit()
    s.close()


@pytest.fixture
def app(db):
    app = Flask(__name__)
    app.secret_key = "test"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(admin_bp)
    return app


def login(app, username, password):
    c = app.test_client()
    res = c.post("/api/auth/login", json={"username": username, "password": password})
    return c, res


# --- Seed ------------------------------------------------------------------


def test_admin_is_seeded_with_default_password(db):
    s = db.get_session()
    admin = s.query(User).one()
    assert (admin.username, admin.role, admin.is_active) == ("admin", "admin", True)
    assert check_password_hash(admin.password_hash, "admin")
    s.close()


def test_seeding_twice_creates_no_second_admin(db):
    db.init_default_data()
    db.init_default_data()
    s = db.get_session()
    assert s.query(User).count() == 1
    s.close()


def test_orphan_chats_are_assigned_to_admin(db):
    s = db.get_session()
    s.add(ChatSession(id="alt-1", title="Alt", user_id=None))
    s.commit()
    s.close()
    db.init_default_data()
    s = db.get_session()
    admin = s.query(User).one()
    assert s.query(ChatSession).filter_by(id="alt-1").one().user_id == admin.id
    s.close()


# --- Login -----------------------------------------------------------------


def test_login_succeeds_and_me_returns_user(app):
    c, res = login(app, "admin", "admin")
    assert res.status_code == 200 and res.json["user"]["role"] == "admin"
    me = c.get("/api/auth/me").json
    assert me["user"]["username"] == "admin" and "password_hash" not in str(me)


def test_username_is_case_insensitive(app):
    assert login(app, "ADMIN", "admin")[1].status_code == 200


@pytest.mark.parametrize(
    "user,pw", [("admin", "falsch"), ("niemand", "admin"), ("", "")]
)
def test_login_rejects_bad_credentials(app, user, pw):
    c, res = login(app, user, pw)
    assert res.status_code == 401
    assert c.get("/api/auth/me").status_code == 401


def test_inactive_user_cannot_login(app, db):
    add_user(db, "gesperrt", "pw", active=False)
    assert login(app, "gesperrt", "pw")[1].status_code == 401


def test_login_is_throttled_after_repeated_failures(app):
    for _ in range(users.MAX_FAILURES):
        assert login(app, "admin", "falsch")[1].status_code == 401
    # Jetzt ist selbst das richtige Passwort kurz gesperrt.
    assert login(app, "admin", "admin")[1].status_code == 429
    # Andere Benutzernamen sind nicht betroffen.
    assert login(app, "anderer", "x")[1].status_code == 401


def test_deactivating_a_user_ends_the_existing_session(app, db):
    add_user(db, "bob", "pw")
    c, _ = login(app, "bob", "pw")
    assert c.get("/api/auth/me").status_code == 200
    s = db.get_session()
    s.query(User).filter_by(username="bob").one().is_active = False
    s.commit()
    s.close()
    assert c.get("/api/auth/me").status_code == 401


def test_old_style_session_cookie_is_no_longer_valid(app):
    c = app.test_client()
    with c.session_transaction() as s:
        s["authenticated"] = True
    assert c.get("/api/auth/me").status_code == 401


# --- Rollen ----------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path,expected",
    [
        # (Rolle -> Status) wird unten pro Rolle geprueft
        ("get", "/api/connections", {"user": 403, "editor": 403, "admin": 200}),
        ("get", "/api/admin/tools", {"user": 403, "editor": 403, "admin": 200}),
        (
            "post",
            "/api/knowledge",
            {"user": 400, "editor": 400, "admin": 400},
        ),  # jede Rolle legt eigenes Knowledge an (ohne Name: 400)
        ("get", "/api/knowledge", {"user": 200, "editor": 200, "admin": 200}),
        ("get", "/api/models", {"user": 200, "editor": 200, "admin": 200}),
        ("get", "/api/sessions", {"user": 200, "editor": 200, "admin": 200}),
    ],
)
def test_role_matrix(app, db, method, path, expected):
    for role in ("user", "editor", "admin"):
        add_user(db, f"{role}1", "pw", role=role) if role != "admin" else None
    for role, status in expected.items():
        name = "admin" if role == "admin" else f"{role}1"
        pw = "admin" if role == "admin" else "pw"
        c, _ = login(app, name, pw)
        res = getattr(c, method)(path, json={}) if method == "post" else c.get(path)
        assert res.status_code == status, f"{role} {method} {path}"


def test_anonymous_is_rejected_everywhere(app):
    c = app.test_client()
    for path in ("/api/connections", "/api/models", "/api/sessions", "/api/knowledge"):
        assert c.get(path).status_code == 401


def test_normal_user_sees_models_without_prompts_or_config(app, db, monkeypatch):
    from app.llm import connections as store
    from app.llm import registry

    conn = store.create_connection("C", "http://x/v1", "k")
    ref = store.make_base_ref(conn["id"], "m")
    store.save_model_config(
        ref, {"system_prompt": "GEHEIM", "knowledge_ids": [1], "visible": True}
    )
    store.save_model_config(
        store.make_base_ref(conn["id"], "hidden"), {"visible": False}
    )
    monkeypatch.setattr(
        registry, "fetch_remote_models", lambda c, force=False: ["m", "hidden"]
    )
    add_user(db, "u", "pw")

    c, _ = login(app, "u", "pw")
    body = c.get("/api/models").get_data(as_text=True)
    assert "GEHEIM" not in body and '"config"' not in body
    mine = {
        m["model_id"]: m
        for m in c.get("/api/models").json["models"]
        if m["connection_name"] == "C"
    }
    assert "hidden" not in mine  # versteckte Modelle bleiben verborgen
    # Modell-Knowledge ist die Basis; eigene Auswahl bleibt erlaubt (Standard), die Basis wird mitgeliefert
    assert mine["m"]["knowledge_locked"] is False
    assert [k["id"] for k in mine["m"]["knowledge_base"]] == [1]
    store.save_model_config(ref, {"allow_extra_knowledge": False})
    locked = {
        m["model_id"]: m
        for m in c.get("/api/models").json["models"]
        if m["connection_name"] == "C"
    }
    assert (
        locked["m"]["knowledge_locked"] is True
    )  # Admin hat zusaetzliches Knowledge ausgeschaltet

    admin, _ = login(app, "admin", "admin")
    assert "GEHEIM" in admin.get("/api/models").get_data(as_text=True)


# --- Chats pro Nutzer ------------------------------------------------------


def test_chats_are_private_per_user(app, db):
    add_user(db, "a", "pw")
    add_user(db, "b", "pw")
    ca, _ = login(app, "a", "pw")
    cb, _ = login(app, "b", "pw")

    sid = ca.post("/api/sessions").json["id"]
    assert [s["id"] for s in ca.get("/api/sessions").json] == [sid]
    assert cb.get("/api/sessions").json == []

    assert cb.get(f"/api/sessions/{sid}").status_code == 404
    assert cb.get(f"/history/{sid}").status_code == 404
    assert cb.delete(f"/api/sessions/{sid}").status_code == 404
    assert ca.get(f"/api/sessions/{sid}").status_code == 200  # nichts passiert

    assert ca.delete(f"/api/sessions/{sid}").status_code == 200


def test_attachments_of_foreign_chats_are_not_accessible(app, db):
    add_user(db, "a", "pw")
    add_user(db, "b", "pw")
    ca, _ = login(app, "a", "pw")
    cb, _ = login(app, "b", "pw")
    sid = ca.post("/api/sessions").json["id"]
    att = db.create_attachment(sid, "x.pdf")

    assert ca.get(f"/api/chat/attachments/{att.id}").status_code == 200
    assert cb.get(f"/api/chat/attachments/{att.id}").status_code == 404


# --- Passwort --------------------------------------------------------------


def test_change_password_uses_the_users_own_hash(app, db):
    c, _ = login(app, "admin", "admin")
    bad = c.post(
        "/api/settings/change-password",
        json={"old_password": "x", "new_password": "neu123"},
    )
    assert bad.status_code == 403
    ok = c.post(
        "/api/settings/change-password",
        json={"old_password": "admin", "new_password": "neu123"},
    )
    assert ok.status_code == 200

    assert login(app, "admin", "admin")[1].status_code == 401
    assert login(app, "admin", "neu123")[1].status_code == 200


# --- Wildcards und Umgehung der Sperre --------------------------------------


@pytest.mark.parametrize("name", ["%", "a%", "ad%", "_dmin", "admi_", "%min", "adm%n"])
def test_wildcards_in_username_do_not_match(app, name):
    c, res = login(app, name, "admin")  # richtiges Passwort, falscher Name
    assert res.status_code == 401
    assert c.get("/api/auth/me").status_code == 401


def test_throttle_counts_per_account_not_per_spelling(app):
    variants = ["admin", "ADMIN", "Admin", " admin ", "aDmIn"]
    assert len(variants) == users.MAX_FAILURES
    for v in variants:
        assert login(app, v, "falsch")[1].status_code == 401
    # Egal wie geschrieben: das Konto ist jetzt gesperrt.
    for v in ("admin", "ADMIN", "aDMin"):
        assert login(app, v, "admin")[1].status_code == 429


def test_throttle_also_counts_per_ip(app):
    # Wechselnde Namen von einer IP: jeder Name bleibt einzeln unter dem
    # Kontolimit, die IP erreicht aber ihr Limit.
    for i in range(users.MAX_FAILURES_PER_IP):
        assert users.authenticate(f"name{i}", "x", client_ip="10.0.0.9") is None
    with pytest.raises(users.LoginThrottledError):
        users.authenticate("noch-ein-neuer-name", "x", client_ip="10.0.0.9")

    # Eine andere IP ist nicht betroffen.
    assert users.authenticate("irgendwer", "x", client_ip="10.0.0.10") is None
