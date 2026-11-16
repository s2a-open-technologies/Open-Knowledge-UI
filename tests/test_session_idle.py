"""Leerlauf-Zeitlimit: Cookie mit Frist, gleitende Erneuerung, Dauer-Abfragen zaehlen nicht."""

import re
import time

from app.core import session_idle
from tests.web import anon, make_app

USER = {"id": 1, "username": "anna", "role": "user"}


def _later(monkeypatch, seconds: float) -> None:
    """Die Uhr (auch die des Cookie-Signierers) um ``seconds`` vorstellen."""
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + seconds)


def _app(db):
    return make_app()


def _cookie_header(response) -> str | None:
    return next(
        (v for k, v in response.headers.multi_items() if k == "set-cookie" and v.startswith("session=")),
        None,
    )


def _login(app, db):
    from tests.test_users import add_user

    add_user(db, "anna", "geheim1")
    r = anon(app).post("/api/auth/login", json={"username": "anna", "password": "geheim1"})
    assert r.status_code == 200, r.text
    return r


def test_login_cookie_has_a_30_minute_lifetime(db):
    r = _login(_app(db), db)
    assert f"Max-Age={session_idle.IDLE_SECONDS}" in _cookie_header(r)
    assert session_idle.IDLE_SECONDS == 1800


def test_activity_renews_the_cookie_after_a_minute(db, monkeypatch):
    app = _app(db)
    login = _login(app, db)
    client = anon(app)
    client.cookies.set("session", re.match(r"session=([^;]+)", _cookie_header(login)).group(1))
    # die erste Anfrage nach dem Login vermerkt die Zeit; eine zweite gleich darauf erneuert nichts
    first = client.get("/api/sessions")
    client.cookies.set("session", re.match(r"session=([^;]+)", _cookie_header(first)).group(1))
    assert _cookie_header(client.get("/api/sessions")) is None
    _later(monkeypatch, 120)
    renewed = client.get("/api/sessions")
    assert renewed.status_code == 200
    assert f"Max-Age={session_idle.IDLE_SECONDS}" in _cookie_header(renewed)


def test_background_polling_does_not_renew_the_session(db, monkeypatch):
    app = _app(db)
    login = _login(app, db)
    client = anon(app)
    client.cookies.set("session", re.match(r"session=([^;]+)", _cookie_header(login)).group(1))
    _later(monkeypatch, 600)
    r = client.get("/api/ingestion-status")
    assert r.status_code == 200
    assert _cookie_header(r) is None


def test_keepalive_renews_and_needs_a_login(db, monkeypatch):
    app = _app(db)
    assert anon(app).post("/api/auth/keepalive").status_code == 401
    login = _login(app, db)
    client = anon(app)
    client.cookies.set("session", re.match(r"session=([^;]+)", _cookie_header(login)).group(1))
    _later(monkeypatch, 300)
    r = client.post("/api/auth/keepalive")
    assert r.status_code == 200
    assert f"Max-Age={session_idle.IDLE_SECONDS}" in _cookie_header(r)


def test_cookie_older_than_30_minutes_is_rejected_by_the_server(db, monkeypatch):
    app = _app(db)
    login = _login(app, db)
    value = re.match(r"session=([^;]+)", _cookie_header(login)).group(1)
    _later(monkeypatch, session_idle.IDLE_SECONDS + 60)
    r = anon(app).get("/api/sessions", headers={"Cookie": f"session={value}"})
    assert r.status_code == 401
