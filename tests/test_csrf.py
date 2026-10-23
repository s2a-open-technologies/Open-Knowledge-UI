"""CSRF-Schutz der Cookie-Sitzung (app/csrf.py)."""

import pytest
from flask import Flask

from app.core.csrf import COOKIE, HEADER, init_csrf, same_origin
from app.web.api_routes import api_bp
from app.web.routes import bp as main_bp


@pytest.fixture
def app(db):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    init_csrf(app)
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)
    return app


def token_of(client):
    client.get("/api/auth/me")  # liefert 401, setzt aber Sitzung und Cookie
    return client.get_cookie(COOKIE).value


def test_changing_requests_without_a_valid_token_are_refused(app):
    c = app.test_client()
    token_of(c)
    for headers in ({}, {HEADER: "falsch"}):
        res = c.post("/api/auth/logout", json={}, headers=headers)
        assert res.status_code == 403 and "CSRF" in res.json["error"]
    for method in ("put", "patch", "delete"):
        assert getattr(c, method)("/api/knowledge/1", json={}).status_code == 403


def test_a_request_with_the_session_token_passes(app):
    c = app.test_client()
    token = token_of(c)
    assert (
        c.post("/api/auth/logout", json={}, headers={HEADER: token}).status_code == 200
    )
    # Login: der Token reicht, ueber falsche Zugangsdaten entscheidet erst die Anmeldung (401, nicht 403)
    c2 = app.test_client()
    token2 = token_of(c2)
    assert (
        c2.post(
            "/api/auth/login",
            json={"username": "x", "password": "y"},
            headers={HEADER: token2},
        ).status_code
        == 401
    )


def test_a_form_field_works_too(app):
    c = app.test_client()
    token = token_of(c)
    assert c.post("/login", data={"username": "x", "password": "y"}).status_code == 403
    assert (
        c.post(
            "/login", data={"username": "x", "password": "y", "csrf_token": token}
        ).status_code
        != 403
    )


def test_a_token_from_another_session_is_refused(app):
    a, b = app.test_client(), app.test_client()
    token_a = token_of(a)
    token_of(b)
    assert (
        b.post("/api/auth/logout", json={}, headers={HEADER: token_a}).status_code
        == 403
    )


def test_reading_requests_need_no_token_and_set_the_cookies(app):
    c = app.test_client()
    res = c.get("/api/auth/me")
    cookies = res.headers.getlist("Set-Cookie")
    csrf = next(h for h in cookies if h.startswith(COOKIE + "="))
    session = next(h for h in cookies if h.startswith("session="))
    assert (
        "HttpOnly" not in csrf and "SameSite=Lax" in csrf
    )  # das Skript der Seite muss ihn lesen koennen
    assert (
        "HttpOnly" in session and "SameSite=Lax" in session
    )  # der Sitzungs-Cookie nicht


def test_login_and_logout_rotate_the_token(app, db):
    c = app.test_client()
    first = token_of(c)
    assert (
        c.post("/api/auth/logout", json={}, headers={HEADER: first}).status_code == 200
    )
    second = c.get_cookie(COOKIE).value
    assert (
        second != first
    )  # session.clear() -> neues Token, die Antwort setzt den Cookie neu
    assert (
        c.post("/api/auth/logout", json={}, headers={HEADER: first}).status_code == 403
    )
    assert (
        c.post("/api/auth/logout", json={}, headers={HEADER: second}).status_code == 200
    )


def test_the_check_can_be_switched_off_for_tests(app):
    app.config["CSRF_ENABLED"] = False
    assert app.test_client().post("/api/auth/logout", json={}).status_code == 200


def test_websocket_origin_must_match_the_host():
    assert same_origin(None, "app.example.org")  # kein Browser
    assert same_origin("https://app.example.org", "app.example.org")
    assert not same_origin("https://boese.example.net", "app.example.org")
    assert not same_origin("https://app.example.org", None)
