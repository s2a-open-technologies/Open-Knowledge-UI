import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from flask import Flask
from ldap3 import MOCK_SYNC, OFFLINE_AD_2012_R2, Connection, Server

from app.auth import auth_external, auth_ldap, auth_oidc, users
from app.auth.user_admin_routes import users_bp
from app.core.models import Group, GroupMember, User
from app.web.admin_routes import admin_bp
from app.web.api_routes import api_bp
from app.web.routes import bp as main_bp
from tests.test_access import add_user, client_for

ADMIN = {"id": 1, "role": "admin", "username": "admin"}


@pytest.fixture(autouse=True)
def fresh():
    users.reset_throttle()
    auth_oidc._discovery.clear()
    yield


@pytest.fixture
def app(db):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    for bp in (main_bp, api_bp, admin_bp, users_bp):
        app.register_blueprint(bp)
    return app


def add_group(db, name):
    s = db.get_session()
    g = Group(name=name)
    s.add(g)
    s.commit()
    gid = g.id
    s.close()
    return gid


def members(db, username):
    s = db.get_session()
    try:
        u = s.query(User).filter_by(username=username).first()
        rows = s.query(GroupMember).filter_by(user_id=u.id).all()
        names = {s.query(Group).get(r.group_id).name: r.external for r in rows}
        return names
    finally:
        s.close()


# --- LDAP (ldap3 Mock) ------------------------------------------------------------------------


@pytest.fixture
def ldap_server(monkeypatch):
    server = Server("ldap.firma.test", get_info=OFFLINE_AD_2012_R2)
    seed = Connection(
        server,
        user="cn=svc,dc=firma,dc=test",
        password="svcpw",
        client_strategy=MOCK_SYNC,
    )
    seed.strategy.add_entry(
        "cn=svc,dc=firma,dc=test",
        {"userPassword": "svcpw", "objectClass": "person", "sAMAccountName": "svc"},
    )
    seed.strategy.add_entry(
        "cn=Alice Muster,ou=users,dc=firma,dc=test",
        {
            "userPassword": "alicepw",
            "objectClass": "person",
            "sAMAccountName": "alice",
            "displayName": "Alice Muster",
            "memberOf": [
                "CN=Vertrieb,OU=Groups,DC=firma,DC=test",
                "CN=Unbekannt,OU=Groups,DC=firma,DC=test",
            ],
        },
    )
    seed.strategy.add_entry(
        "cn=Bob Test,ou=users,dc=firma,dc=test",
        {
            "userPassword": "bobpw",
            "objectClass": "person",
            "sAMAccountName": "bob",
            "displayName": "Bob Test",
            "memberOf": [],
        },
    )
    monkeypatch.setattr(auth_ldap, "CLIENT_STRATEGY", MOCK_SYNC)
    monkeypatch.setattr(auth_ldap, "_server", lambda cfg, get_info=None: server)
    return server


LDAP = {
    "enabled": True,
    "url": "ldap://ldap.firma.test",
    "bind_dn": "cn=svc,dc=firma,dc=test",
    "bind_password": "svcpw",
    "base_dn": "dc=firma,dc=test",
}


def enable_ldap(app, **over):
    r = client_for(app, ADMIN).put("/api/admin/auth", json={"ldap": {**LDAP, **over}})
    assert r.status_code == 200, r.json
    return r


def login(app, username, password):
    return app.test_client().post(
        "/api/auth/login", json={"username": username, "password": password}
    )


def test_first_ldap_login_creates_the_account_with_the_default_role(
    app, db, ldap_server
):
    enable_ldap(app)
    r = login(app, "alice", "alicepw")
    assert r.status_code == 200 and r.json["user"]["role"] == "user"
    assert (
        r.json["user"]["display_name"] == "Alice Muster"
        and r.json["user"]["auth_source"] == "ldap"
    )
    again = login(app, "alice", "alicepw")
    assert again.json["user"]["id"] == r.json["user"]["id"]  # kein zweites Konto


def test_wrong_password_and_unknown_user_are_refused(app, db, ldap_server):
    enable_ldap(app)
    assert login(app, "alice", "falsch").status_code == 401
    assert login(app, "niemand", "x").status_code == 401
    assert (
        login(app, "alice", "").status_code == 401
    )  # leeres Passwort = anonymer Bind: nie
    s = db.get_session()
    assert s.query(User).filter_by(username="alice").count() == 0
    s.close()


def test_ldap_injection_in_the_username_does_not_widen_the_search(app, db, ldap_server):
    enable_ldap(app)
    assert login(app, "*", "alicepw").status_code == 401
    assert login(app, "alice)(sAMAccountName=*", "alicepw").status_code == 401
    assert login(app, "a*", "alicepw").status_code == 401


def test_groups_are_matched_by_name_never_created_and_only_external_ones_removed(
    app, db, ldap_server
):
    add_group(db, "vertrieb")  # Schreibweise egal
    manual = add_group(db, "Manuell")
    enable_ldap(app)
    login(app, "alice", "alicepw")
    assert members(db, "alice") == {"vertrieb": True}
    s = db.get_session()
    assert s.query(Group).filter_by(name="Unbekannt").count() == 0  # nie angelegt
    uid = s.query(User).filter_by(username="alice").first().id
    s.add(
        GroupMember(group_id=manual, user_id=uid, external=False)
    )  # Admin setzt manuell
    s.commit()
    s.close()
    # Der Anbieter meldet Vertrieb nicht mehr: nur die externe Mitgliedschaft verschwindet
    conn = Connection(
        ldap_server,
        user="cn=svc,dc=firma,dc=test",
        password="svcpw",
        client_strategy=MOCK_SYNC,
    )
    conn.bind()
    conn.strategy.remove_entry("cn=Alice Muster,ou=users,dc=firma,dc=test")
    conn.strategy.add_entry(
        "cn=Alice Muster,ou=users,dc=firma,dc=test",
        {
            "userPassword": "alicepw",
            "objectClass": "person",
            "sAMAccountName": "alice",
            "displayName": "Alice Muster",
            "memberOf": [],
        },
    )
    login(app, "alice", "alicepw")
    assert members(db, "alice") == {"Manuell": False}


def test_external_groups_are_stored_even_when_no_local_group_matches(
    app, db, ldap_server
):
    enable_ldap(app)
    login(app, "alice", "alicepw")
    s = db.get_session()
    stored = json.loads(
        s.query(User).filter_by(username="alice").first().external_groups
    )
    s.close()
    assert stored == ["Unbekannt", "Vertrieb"]


def test_local_accounts_stay_local_and_win_name_collisions(app, db, ldap_server):
    add_user(db, "alice", "editor", password="lokal")
    enable_ldap(app)
    assert login(app, "alice", "lokal").status_code == 200  # lokales Passwort
    assert (
        login(app, "alice", "alicepw").status_code == 401
    )  # LDAP-Passwort gilt fuer lokale Konten nicht
    assert login(app, "admin", "admin").status_code == 200  # Notzugang bleibt


def test_auto_create_can_be_switched_off_and_disabled_accounts_stay_locked(
    app, db, ldap_server
):
    enable_ldap(app)
    client_for(app, ADMIN).put("/api/admin/auth", json={"auto_create": False})
    assert login(app, "alice", "alicepw").status_code == 401
    client_for(app, ADMIN).put("/api/admin/auth", json={"auto_create": True})
    assert login(app, "alice", "alicepw").status_code == 200
    s = db.get_session()
    s.query(User).filter_by(username="alice").update({"is_active": False})
    s.commit()
    s.close()
    assert login(app, "alice", "alicepw").status_code == 401


def test_external_accounts_have_no_password_login_or_password_change(
    app, db, ldap_server
):
    enable_ldap(app)
    c = app.test_client()
    c.post("/api/auth/login", json={"username": "alice", "password": "alicepw"})
    me = c.get("/api/auth/me").json
    assert me["user"]["auth_source"] == "ldap"
    r = c.post(
        "/api/auth/change-password",
        json={"old_password": "alicepw", "new_password": "neuesPasswort"},
    )
    assert r.status_code in (400, 403)
    ids = client_for(app, ADMIN).get("/api/users").json
    uid = next(u["id"] for u in ids if u["username"] == "alice")
    assert (
        client_for(app, ADMIN)
        .patch(f"/api/users/{uid}", json={"password": "geheimGeheim1"})
        .status_code
        == 400
    )


def test_ldap_down_means_no_login_but_local_login_still_works(app, db, monkeypatch):
    monkeypatch.setattr(
        auth_ldap,
        "_server",
        lambda cfg, get_info=None: (_ for _ in ()).throw(auth_ldap.LdapError("weg")),
    )
    enable_ldap(app)
    assert login(app, "alice", "alicepw").status_code == 401
    assert login(app, "admin", "admin").status_code == 200


def test_secrets_never_leave_the_server_and_empty_fields_keep_them(
    app, db, ldap_server
):
    r = enable_ldap(app)
    text = r.get_data(as_text=True)
    assert "svcpw" not in text and r.json["ldap"]["has_bind_password"] is True
    client_for(app, ADMIN).put(
        "/api/admin/auth",
        json={"ldap": {"base_dn": "dc=firma,dc=test", "bind_password": ""}},
    )
    assert auth_external.load_config()["ldap"]["bind_password"] == "svcpw"
    assert "svcpw" not in client_for(app, ADMIN).get("/api/admin/auth").get_data(
        as_text=True
    )


def test_auth_settings_are_admin_only_and_validated(app, db):
    ed = add_user(db, "ed", "editor")
    assert client_for(app, ed).get("/api/admin/auth").status_code == 403
    assert client_for(app, ed).put("/api/admin/auth", json={}).status_code == 403
    admin = client_for(app, ADMIN)
    assert (
        admin.put(
            "/api/admin/auth", json={"ldap": {"enabled": True, "url": "http://x"}}
        ).status_code
        == 400
    )
    assert (
        admin.put(
            "/api/admin/auth",
            json={
                "ldap": {
                    "enabled": True,
                    "url": "ldap://169.254.169.254",
                    "base_dn": "dc=x",
                }
            },
        ).status_code
        == 400
    )
    assert (
        admin.put(
            "/api/admin/auth",
            json={
                "ldap": {
                    "enabled": True,
                    "url": "ldap://x",
                    "base_dn": "dc=x",
                    "user_filter": "(uid=x)",
                }
            },
        ).status_code
        == 400
    )
    assert (
        admin.put("/api/admin/auth", json={"default_role": "root"}).status_code == 400
    )
    assert (
        admin.put(
            "/api/admin/auth",
            json={"oidc": {"enabled": True, "issuer": "https://idp", "client_id": "a"}},
        ).status_code
        == 400
    )


def test_ldap_test_endpoint_reports_facts_and_does_not_log_in(app, db, ldap_server):
    admin = client_for(app, ADMIN)
    r = admin.post(
        "/api/admin/auth/test",
        json={
            "kind": "ldap",
            "values": LDAP,
            "test_user": "alice",
            "test_password": "alicepw",
        },
    )
    assert (
        r.status_code == 200
        and r.json["service_bind"]
        and r.json["user_found"]
        and r.json["login_ok"]
    )
    assert "Vertrieb" in r.json["groups"]
    s = db.get_session()
    assert (
        s.query(User).filter_by(username="alice").count() == 0
    )  # Test legt kein Konto an
    s.close()


# --- OIDC (lokaler Test-Anbieter) --------------------------------------------------------------


KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _b64(n: int) -> str:
    return (
        base64.urlsafe_b64encode(n.to_bytes((n.bit_length() + 7) // 8, "big"))
        .rstrip(b"=")
        .decode()
    )


class Idp:
    def __init__(self):
        self.claims = {
            "sub": "u-1",
            "preferred_username": "carol",
            "name": "Carol Test",
            "groups": ["Vertrieb", "Fremd"],
        }
        self.override = {}
        self.codes = {}
        idp = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, body, status=200):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                base = idp.url
                if self.path.endswith("/.well-known/openid-configuration"):
                    self._json(
                        {
                            "issuer": base,
                            "authorization_endpoint": f"{base}/authorize",
                            "token_endpoint": f"{base}/token",
                            "jwks_uri": f"{base}/jwks",
                            "code_challenge_methods_supported": ["S256"],
                        }
                    )
                elif self.path == "/jwks":
                    nums = KEY.public_key().public_numbers()
                    self._json(
                        {
                            "keys": [
                                {
                                    "kty": "RSA",
                                    "kid": "k1",
                                    "use": "sig",
                                    "alg": "RS256",
                                    "n": _b64(nums.n),
                                    "e": _b64(nums.e),
                                }
                            ]
                        }
                    )
                else:
                    self._json({}, 404)

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                form = parse_qs(self.rfile.read(length).decode())
                info = idp.codes.get(form["code"][0])
                if not info or form["client_secret"][0] != "s3cret":
                    return self._json({"error": "invalid_grant"}, 400)
                claims = {
                    **idp.claims,
                    "iss": idp.url,
                    "aud": "rag",
                    "iat": int(time.time()),
                    "exp": int(time.time()) + 300,
                    "nonce": info["nonce"],
                    **idp.override,
                }
                idp.pkce_ok = bool(form.get("code_verifier"))
                token = jwt.encode(
                    claims, KEY, algorithm="RS256", headers={"kid": "k1"}
                )
                self._json(
                    {"id_token": token, "access_token": "x", "token_type": "Bearer"}
                )

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def authorize(self, location: str) -> str:
        """Nimmt die Weiterleitung entgegen und liefert die Rueckruf-Adresse mit Code."""
        query = parse_qs(urlparse(location).query)
        code = f"code-{len(self.codes)}"
        self.codes[code] = {"nonce": query["nonce"][0]}
        assert query["code_challenge_method"] == ["S256"] and query["code_challenge"]
        return f"/api/auth/oidc/callback?code={code}&state={query['state'][0]}"


@pytest.fixture
def idp():
    server = Idp()
    yield server
    server.srv.shutdown()


def enable_oidc(app, idp, **over):
    block = {
        "enabled": True,
        "name": "Firma-SSO",
        "issuer": idp.url,
        "client_id": "rag",
        "client_secret": "s3cret",
        **over,
    }
    r = client_for(app, ADMIN).put(
        "/api/admin/auth", json={"oidc": block, "base_url": "http://app.test"}
    )
    assert r.status_code == 200, r.json


def sso(app, idp):
    c = app.test_client()
    first = c.get("/api/auth/oidc/login")
    assert first.status_code == 302 and first.location.startswith(
        idp.url + "/authorize"
    )
    return c, c.get(idp.authorize(first.location))


def test_providers_endpoint_tells_the_login_page_what_to_show(app, db, idp):
    assert app.test_client().get("/api/auth/providers").json == {
        "ldap": False,
        "oidc": {"enabled": False, "name": "SSO"},
    }
    enable_oidc(app, idp)
    assert app.test_client().get("/api/auth/providers").json["oidc"] == {
        "enabled": True,
        "name": "Firma-SSO",
    }


def test_oidc_login_creates_the_account_matches_groups_by_name_and_signs_in(
    app, db, idp
):
    add_group(db, "VERTRIEB")
    enable_oidc(app, idp)
    c, done = sso(app, idp)
    assert done.status_code == 302 and done.location.endswith("/")
    me = c.get("/api/auth/me").json["user"]
    assert (
        me["username"] == "carol"
        and me["auth_source"] == "oidc"
        and me["role"] == "user"
    )
    assert members(db, "carol") == {"VERTRIEB": True}
    s = db.get_session()
    assert s.query(Group).filter_by(name="Fremd").count() == 0
    s.close()
    assert idp.pkce_ok


def test_oidc_rejects_bad_state_wrong_audience_wrong_nonce_and_expired_tokens(
    app, db, idp
):
    enable_oidc(app, idp)
    c = app.test_client()
    first = c.get("/api/auth/oidc/login")
    cb = idp.authorize(first.location)
    bad_state = c.get(cb.split("state=")[0] + "state=falsch")
    assert (
        "login_error" in bad_state.location and c.get("/api/auth/me").status_code == 401
    )

    for override in (
        {"aud": "anderer"},
        {"nonce": "falsch"},
        {"exp": int(time.time()) - 600},
        {"iss": "http://boese.test"},
    ):
        idp.override = override
        c2 = app.test_client()
        done = c2.get(idp.authorize(c2.get("/api/auth/oidc/login").location))
        assert "login_error" in done.location, override
        assert c2.get("/api/auth/me").status_code == 401
    s = db.get_session()
    assert s.query(User).filter_by(username="carol").count() == 0
    s.close()


def test_oidc_token_signed_with_another_key_is_refused(app, db, idp, monkeypatch):
    enable_oidc(app, idp)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    real = jwt.encode

    def forge(claims, key, algorithm, headers):
        return real(claims, other, algorithm=algorithm, headers=headers)

    monkeypatch.setattr("tests.test_external_auth.jwt.encode", forge)
    c = app.test_client()
    done = c.get(idp.authorize(c.get("/api/auth/oidc/login").location))
    assert "login_error" in done.location and c.get("/api/auth/me").status_code == 401


def test_oidc_accounts_cannot_use_the_password_form_and_local_accounts_cannot_be_taken_over(
    app, db, idp
):
    enable_oidc(app, idp)
    sso(app, idp)
    assert login(app, "carol", "irgendwas").status_code == 401
    add_user(db, "dave", "editor", password="lokal")
    idp.claims = {**idp.claims, "sub": "u-2", "preferred_username": "dave"}
    c, done = sso(app, idp)
    assert "login_error" in done.location  # Name gehoert einem lokalen Konto
    assert c.get("/api/auth/me").status_code == 401


def test_oidc_disabled_or_unreachable_sends_the_user_back_to_the_login(app, db, idp):
    assert "login_error" in app.test_client().get("/api/auth/oidc/login").location
    enable_oidc(app, idp)
    idp.srv.shutdown()
    auth_oidc._discovery.clear()
    assert "login_error" in app.test_client().get("/api/auth/oidc/login").location


def test_oidc_test_endpoint_shows_the_callback_url(app, db, idp):
    r = client_for(app, ADMIN).post(
        "/api/admin/auth/test", json={"kind": "oidc", "values": {"issuer": idp.url}}
    )
    assert (
        r.status_code == 200 and r.json["jwks_reachable"] and r.json["pkce_supported"]
    )
    assert r.json["redirect_uri"].endswith("/api/auth/oidc/callback")
    bad = client_for(app, ADMIN).post(
        "/api/admin/auth/test",
        json={"kind": "oidc", "values": {"issuer": "http://127.0.0.1:9"}},
    )
    assert bad.status_code == 502


def test_group_name_helper_reads_cn_from_dns():
    assert auth_external.group_name_from("CN=Vertrieb,OU=G,DC=x") == "Vertrieb"
    assert auth_external.group_name_from("opendox-admin") == "opendox-admin"
    assert auth_external.group_name_from("a=b,c") == "a=b,c"


def test_oidc_never_links_an_existing_account_by_name_only(app, db, idp):
    enable_oidc(app, idp)
    sso(app, idp)  # carol, sub u-1
    s = db.get_session()
    first_id = s.query(User).filter_by(username="carol").first().id
    s.close()
    idp.claims = {
        **idp.claims,
        "sub": "u-evil",
    }  # anderer Mensch, gleicher preferred_username
    c, done = sso(app, idp)
    assert "login_error" in done.location and c.get("/api/auth/me").status_code == 401
    s = db.get_session()
    user = s.query(User).filter_by(username="carol").first()
    assert user.id == first_id and user.external_id == "u-1"  # nicht umgebogen
    s.close()
