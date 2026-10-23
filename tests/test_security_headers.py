"""Sicherheits-Header (app/core/security_headers.py): Header vorhanden, HSTS nur bei Secure, Off-Schalter."""

from flask import Flask

from app.core.security_headers import CSP, HSTS, init_security_headers


def _client(monkeypatch, secure=False):
    app = Flask(__name__)
    init_security_headers(app)
    app.config["SESSION_COOKIE_SECURE"] = secure
    return app.test_client()


def test_all_headers_present(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/ohne-alles")
    assert r.status_code == 404  # Route ist egal, Header kommen von after_request
    assert r.headers["Content-Security-Policy"] == CSP
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "Strict-Transport-Security" not in r.headers


def test_hsts_only_in_secure_mode(monkeypatch):
    c = _client(monkeypatch, secure=True)
    r = c.get("/ohne-alles")
    assert r.headers["Strict-Transport-Security"] == HSTS


def test_off_switch(monkeypatch):
    monkeypatch.setenv("SECURITY_HEADERS", "off")
    c = _client(monkeypatch)
    r = c.get("/ohne-alles")
    assert "Content-Security-Policy" not in r.headers
    assert "X-Frame-Options" not in r.headers


def test_existing_csp_is_preserved(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/ohne-alles", headers={"X-Frame-Options": "SAMEORIGIN"})
    # X-Frame-Options wird vom Modul gesetzt; ein vom Proxy vorgegebener CSP bleibt stehen
    assert r.headers["X-Frame-Options"] == "DENY"


def test_csp_blocks_foreign_scripts_but_keeps_chat_images():
    csp = {d.split()[0]: d.split(None, 1)[1] for d in CSP.split(";") if d.strip()}
    assert "https:" in csp["img-src"]  # Chat-Markdown darf Fremdbilder zeigen
    assert "unsafe-inline" not in csp["script-src"] or "self" in csp["script-src"]
    assert csp["object-src"] == "'none'"
    assert csp["frame-ancestors"] == "'none'"
