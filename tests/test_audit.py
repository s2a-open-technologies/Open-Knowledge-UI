"""Audit-Log (app/auth/audit.py): Hooks bei Anmeldung/Verwaltung, Admin-API listet Ereignisse."""

import json

import pytest

from app.auth import audit
from app.auth import users as users_module
from app.auth.user_admin import create_user, delete_user, update_user
from tests.test_access import add_user, admin, app, client_for  # noqa: F401


def _actions():
    return [e["action"] for e in audit.list_events()]


def test_record_persists_event(db):
    audit.record(audit.LOGIN_OK, actor_name="alice", detail={"grundtest": True})
    events = audit.list_events()
    assert events[0]["action"] == "login_ok"
    assert events[0]["actor_name"] == "alice"
    assert events[0]["detail"] == '{"grundtest": true}'


def test_login_events(db):
    from app.auth.users import authenticate

    add_user(db, "carol", password="pw")

    assert authenticate("carol", "pw") is not None
    assert "login_ok" in _actions()

    assert authenticate("carol", "falsch") is None
    assert "login_failed" in _actions()

    assert authenticate("niemandda", "egal") is None
    assert "login_failed" in _actions()


def test_user_admin_hooks(db):
    boss = create_user("boss", "Boss", "geheim1", "admin")
    created = create_user("dave", "Dave", "geheim1", "editor")
    assert "user_create" in _actions()

    update_user(created["id"], actor_id=boss["id"], data={"display_name": "Dave B."})
    assert "user_update" in _actions()
    update = next(e for e in audit.list_events() if e["action"] == "user_update")
    assert update["object_id"] == str(created["id"])

    delete_user(created["id"], actor_id=boss["id"])
    assert "user_delete" in _actions()


def test_admin_audit_endpoint(db, admin, app):  # noqa: F811
    audit.record(audit.LOGIN_OK, actor_name="verifizierer")

    editor = add_user(db, "eddy", "editor")
    assert client_for(app, editor).get("/api/admin/audit").status_code == 403

    admin_client = client_for(app, admin)
    res = admin_client.get("/api/admin/audit?limit=10")
    assert res.status_code == 200
    payload = json.loads(res.get_data(as_text=True))
    assert any(e["action"] == "login_ok" for e in payload["events"])

    res = admin_client.get("/api/admin/audit?limit=10&action=user_create")
    payload = json.loads(res.get_data(as_text=True))
    assert all(e["action"] == "user_create" for e in payload["events"])
    assert "login_ok" not in [e["action"] for e in payload["events"]]


def test_logout_records_event(db, app):  # noqa: F811
    user = add_user(db, "erika")
    client = client_for(app, user)
    assert client.post("/api/auth/logout").status_code == 200
    assert "logout" in _actions()


def test_throttle_event_recorded(db, monkeypatch):
    from app.auth.users import authenticate

    users_module._failures.clear()  # Modulstate aus frueheren Tests (user:1-Kollision)
    add_user(db, "fritz", password="pw")
    monkeypatch.setattr(users_module, "MAX_FAILURES", 1)
    assert authenticate("fritz", "falsch") is None
    with pytest.raises(users_module.LoginThrottledError):
        authenticate("fritz", "falsch")
    assert "login_throttled" in _actions()
