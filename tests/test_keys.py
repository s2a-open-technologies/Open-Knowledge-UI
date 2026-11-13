"""Datenschluessel je Nutzer/Knowledge (app/core/keys.py) und Rotation des Master-Schluessels."""

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import text

from app.core import crypto, key_rotation, keys
from app.core.crypto import SecretKeyError
from app.core.models import DataKey, EncryptionKeyState, Knowledge, User
from app.core.settings_service import SettingsService
from app.llm import connections
from tests.test_access import add_user


@pytest.fixture(autouse=True)
def fresh_keys():
    keys.reset()
    yield
    keys.reset()


def _switch_key(monkeypatch, new: str, previous: str | None = None):
    monkeypatch.setenv("ENCRYPTION_KEY", new)
    if previous is None:
        monkeypatch.delenv("ENCRYPTION_KEY_PREVIOUS", raising=False)
    else:
        monkeypatch.setenv("ENCRYPTION_KEY_PREVIOUS", previous)
    crypto.reset()
    keys.reset()


def _rotate(db):
    return key_rotation.run_key_rotation_if_needed(db.engine, SettingsService.encrypted_keys())


# --- UUIDs --------------------------------------------------------------------------------


def test_users_and_knowledge_get_a_stable_unique_uuid(db):
    a, b = add_user(db, "alice"), add_user(db, "bob")
    s = db.get_session()
    try:
        ua = s.get(User, a["id"]).uuid
        ub = s.get(User, b["id"]).uuid
        kb = s.query(Knowledge).first().uuid
    finally:
        s.close()
    assert len({ua, ub, kb}) == 3 and all(len(u) == 36 for u in (ua, ub, kb))
    assert keys.owner_uuid(keys.USER, a["id"]) == ua


# --- Ver- und Entschluesseln --------------------------------------------------------------


def test_text_roundtrip_is_bound_to_its_owner(db):
    a, b = add_user(db, "alice"), add_user(db, "bob")
    sealed = keys.seal_for(keys.USER, a["id"], "Gehalt der GF: 12.345 €")
    assert sealed.startswith(keys.TEXT_PREFIX) and "Gehalt" not in sealed
    assert keys.unseal_for(keys.USER, a["id"], sealed) == "Gehalt der GF: 12.345 €"
    assert keys.seal_for(keys.USER, a["id"], sealed) == sealed  # nicht doppelt
    with pytest.raises(SecretKeyError):
        keys.unseal_for(keys.USER, b["id"], sealed)  # fremder Eigentuemer (AAD)


def test_plaintext_from_before_encryption_is_returned_unchanged(db):
    a = add_user(db, "alice")
    assert keys.unseal_for(keys.USER, a["id"], "alter Klartext") == "alter Klartext"
    assert keys.unseal(keys.USER, "x", b"alte Datei") == b"alte Datei"


def test_blob_roundtrip_and_tamper_detection(db):
    a = add_user(db, "alice")
    uuid = keys.owner_uuid(keys.USER, a["id"])
    blob = keys.seal(keys.USER, uuid, b"%PDF-1.7 vertraulich")
    assert keys.is_sealed(blob) and b"vertraulich" not in blob
    assert keys.unseal(keys.USER, uuid, blob) == b"%PDF-1.7 vertraulich"
    tampered = blob[:-1] + bytes([blob[-1] ^ 1])
    with pytest.raises(SecretKeyError):
        keys.unseal(keys.USER, uuid, tampered)


def test_one_key_per_owner_wrapped_with_the_master_key(db):
    a = add_user(db, "alice")
    keys.seal_for(keys.USER, a["id"], "eins")
    keys.reset()
    keys.seal_for(keys.USER, a["id"], "zwei")
    with db.engine.connect() as conn:
        rows = conn.execute(text("SELECT wrapped_key FROM data_keys")).all()
    assert len(rows) == 1 and crypto.is_encrypted(rows[0][0])


def test_shredding_makes_content_unreadable(db):
    a = add_user(db, "alice")
    sealed = keys.seal_for(keys.USER, a["id"], "geheim")
    assert keys.shred(keys.USER, keys.owner_uuid(keys.USER, a["id"])) == 1
    with pytest.raises(SecretKeyError):
        keys.unseal_for(keys.USER, a["id"], sealed)


def test_data_key_rotation_keeps_old_content_readable(db):
    a = add_user(db, "alice")
    uuid = keys.owner_uuid(keys.USER, a["id"])
    old = keys.seal_text(keys.USER, uuid, "alt")
    assert keys.rotate_data_key(keys.USER, uuid) == 2
    new = keys.seal_text(keys.USER, uuid, "neu")
    keys.reset()
    assert keys.unseal_text(keys.USER, uuid, old) == "alt"
    assert keys.unseal_text(keys.USER, uuid, new) == "neu"


def test_a_wrong_master_key_never_returns_empty_content(db, monkeypatch):
    a = add_user(db, "alice")
    sealed = keys.seal_for(keys.USER, a["id"], "geheim")
    _switch_key(monkeypatch, Fernet.generate_key().decode())
    with pytest.raises(SecretKeyError):
        keys.unseal_for(keys.USER, a["id"], sealed)


# --- Rotation des Master-Schluessels -------------------------------------------------------


def test_encrypted_columns_come_from_the_models():
    cols = {(t, c) for t, c, _pk in key_rotation.encrypted_columns()}
    assert {
        ("connections", "api_key"),
        ("knowledge", "source_config"),
        ("data_keys", "wrapped_key"),
    } <= cols


def test_first_start_stores_the_baseline_and_a_second_start_changes_nothing(db):
    # init_default_data hat die Ausgangslage schon gespeichert
    assert _rotate(db) == "unchanged"
    s = db.get_session()
    try:
        assert s.query(EncryptionKeyState).count() == 1
    finally:
        s.close()


def test_rotation_reencrypts_secrets_and_data_keys(db, monkeypatch):
    old_key = crypto.primary_key().decode()
    a = add_user(db, "alice")
    conn = connections.create_connection("C", "http://x/v1", "sk-geheim")
    sealed = keys.seal_for(keys.USER, a["id"], "Chatverlauf")

    new_key = Fernet.generate_key().decode()
    _switch_key(monkeypatch, new_key, previous=old_key)
    assert _rotate(db) == "rotated"

    _switch_key(monkeypatch, new_key)  # alter Schluessel weg
    assert connections.get_connection(conn["id"], with_key=True)["api_key"] == "sk-geheim"
    assert keys.unseal_for(keys.USER, a["id"], sealed) == "Chatverlauf"
    s = db.get_session()
    try:
        state = s.query(EncryptionKeyState).one()
        assert state.key_fingerprint == key_rotation.fingerprint(new_key.encode())
        assert s.query(DataKey).count() == 1
    finally:
        s.close()
    assert _rotate(db) == "unchanged"


def test_a_wrong_previous_key_leaves_everything_as_it_was(db, monkeypatch):
    connections.create_connection("C", "http://x/v1", "sk-geheim")
    s = db.get_session()
    before = s.query(EncryptionKeyState).one().key_fingerprint
    s.close()

    _switch_key(
        monkeypatch, Fernet.generate_key().decode(), previous=Fernet.generate_key().decode()
    )
    with pytest.raises(SecretKeyError):
        _rotate(db)
    s = db.get_session()
    try:
        assert s.query(EncryptionKeyState).one().key_fingerprint == before
    finally:
        s.close()


def test_a_new_key_without_previous_waits(db, monkeypatch):
    _switch_key(monkeypatch, Fernet.generate_key().decode())
    assert _rotate(db) == "waiting"


def test_an_interrupted_rotation_resumes(db, monkeypatch):
    old_key = crypto.primary_key().decode()
    c1 = connections.create_connection("A", "http://a/v1", "sk-a")
    c2 = connections.create_connection("B", "http://b/v1", "sk-b")
    new_key = Fernet.generate_key().decode()
    _switch_key(monkeypatch, new_key, previous=old_key)
    # Teilweise schon umgeschluesselt (Abbruch nach dem ersten Wert), Fingerprint noch alt
    with db.engine.begin() as conn:
        value = conn.execute(
            text("SELECT api_key FROM connections WHERE id = :id"), {"id": c1["id"]}
        ).scalar()
        conn.execute(
            text("UPDATE connections SET api_key = :v WHERE id = :id"),
            {"v": crypto.rotate(value), "id": c1["id"]},
        )
    assert _rotate(db) == "rotated"
    _switch_key(monkeypatch, new_key)
    assert connections.get_connection(c1["id"], with_key=True)["api_key"] == "sk-a"
    assert connections.get_connection(c2["id"], with_key=True)["api_key"] == "sk-b"


def test_status_reports_pending_rotation(db, monkeypatch):
    assert key_rotation.status()["rotation_pending"] is False
    _switch_key(monkeypatch, Fernet.generate_key().decode())
    assert key_rotation.status()["rotation_pending"] is True
