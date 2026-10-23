"""Zugangsdaten in der Datenbank verschluesseln (app/crypto.py)."""

import os
import stat

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import text

from app.core import crypto
from app.core.settings_service import SettingsService
from app.llm import connections as store
from app.rag import knowledge


def raw(db, sql, **params):
    with db.engine.connect() as conn:
        return conn.execute(text(sql), params).scalar()


def test_roundtrip_prefix_and_passthrough():
    token = crypto.encrypt("geheim")
    assert token.startswith("enc:v1:") and "geheim" not in token
    assert crypto.decrypt(token) == "geheim"
    assert crypto.encrypt(token) == token  # nichts doppelt verschluesseln
    assert crypto.encrypt("") == "" and crypto.encrypt(None) is None
    assert crypto.decrypt("klartext") == "klartext"  # Altbestand ohne Praefix


def test_a_wrong_or_missing_key_never_yields_an_empty_value(monkeypatch):
    token = crypto.encrypt("geheim")
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    crypto.reset()
    with pytest.raises(crypto.SecretKeyError, match="ENCRYPTION_KEY"):
        crypto.decrypt(token)


def test_key_rotation_reads_old_values_and_writes_with_the_first_key(monkeypatch):
    old = os.environ["ENCRYPTION_KEY"]
    token = crypto.encrypt("geheim")
    new = Fernet.generate_key().decode()
    monkeypatch.setenv("ENCRYPTION_KEY", f"{new},{old}")
    crypto.reset()
    assert crypto.decrypt(token) == "geheim"
    fresh = crypto.encrypt("anderes")
    monkeypatch.setenv(
        "ENCRYPTION_KEY", new
    )  # alter Schluessel entfernt: neue Werte bleiben lesbar
    crypto.reset()
    assert crypto.decrypt(fresh) == "anderes"


def test_without_a_key_a_private_key_file_is_created_once(tmp_path, monkeypatch):
    monkeypatch.delenv("ENCRYPTION_KEY")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    crypto.reset()
    token = crypto.encrypt("x")
    path = crypto.key_file_path()
    assert (
        path.exists()
        and stat.S_IMODE(path.stat().st_mode) == 0o600
        and crypto.key_source() == "file"
    )
    content = path.read_bytes()
    crypto.reset()
    assert (
        crypto.decrypt(token) == "x" and path.read_bytes() == content
    )  # wiederverwendet, nicht neu erzeugt


def test_an_invalid_env_key_is_reported(monkeypatch):
    monkeypatch.setenv("ENCRYPTION_KEY", "kein-schluessel")
    crypto.reset()
    with pytest.raises(ValueError):
        crypto.encrypt("x")


def test_connection_keys_are_encrypted_at_rest_and_plain_in_use(db):
    c = store.create_connection("C", "http://x/v1", "sk-top-secret")
    stored = raw(db, "SELECT api_key FROM connections WHERE id = :i", i=c["id"])
    assert stored.startswith("enc:v1:") and "sk-top-secret" not in stored
    assert (
        store.get_connection(c["id"], with_key=True)["api_key"] == "sk-top-secret"
    )  # in Benutzung entschluesselt
    store.update_connection(c["id"], api_key="sk-neu")
    assert "sk-neu" not in raw(
        db, "SELECT api_key FROM connections WHERE id = :i", i=c["id"]
    )


def test_source_credentials_are_encrypted_at_rest(db):
    cfg = {"url": "http://q:6333", "collection": "c", "api_key": "qdrant-secret"}
    from app.rag import model_settings

    k = knowledge.create_external_knowledge(
        "Fremd", "", model_settings.embedding_spec(), 4, cfg
    )
    stored = raw(db, "SELECT source_config FROM knowledge WHERE id = :i", i=k["id"])
    assert stored.startswith("enc:v1:") and "qdrant-secret" not in stored
    assert (
        "qdrant-secret" in knowledge.get_knowledge(k["id"])["source_config"]
    )  # gelesen wird entschluesselt


def test_sensitive_settings_are_encrypted_and_others_are_not(db):
    s = db.get_session()
    service = SettingsService(s)
    service.set("PII_FILTER_API_KEY", "pii-secret")
    service.set("CHAT_MAX_TOKENS", "123")
    assert (
        service.get("PII_FILTER_API_KEY") == "pii-secret"
        and service.get("CHAT_MAX_TOKENS") == "123"
    )
    s.close()
    stored = raw(db, "SELECT value FROM settings WHERE key = 'PII_FILTER_API_KEY'")
    assert stored.startswith("enc:v1:") and "pii-secret" not in stored
    assert raw(db, "SELECT value FROM settings WHERE key = 'CHAT_MAX_TOKENS'") == "123"


def test_existing_plaintext_is_encrypted_once_and_a_wrong_key_stops_startup(
    db, monkeypatch
):
    with db.engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO connections (name, base_url, api_key, enabled, kind) VALUES ('Alt', 'http://a/v1', 'klartext-key', 1, 'llm')"
            )
        )
        conn.execute(
            text("UPDATE settings SET value = 'alt-klartext' WHERE key = 'STT_API_KEY'")
        )
    assert crypto.encrypt_existing(db.engine, SettingsService.encrypted_keys()) >= 2
    assert raw(db, "SELECT api_key FROM connections WHERE name = 'Alt'").startswith(
        "enc:v1:"
    )
    assert raw(db, "SELECT value FROM settings WHERE key = 'STT_API_KEY'").startswith(
        "enc:v1:"
    )
    assert (
        crypto.encrypt_existing(db.engine, SettingsService.encrypted_keys()) == 0
    )  # idempotent

    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    crypto.reset()
    with pytest.raises(crypto.SecretKeyError):
        crypto.encrypt_existing(db.engine, SettingsService.encrypted_keys())
    with pytest.raises(crypto.SecretKeyError):
        db.init_default_data()  # falscher Schluessel: Start bricht ab statt leere Zugangsdaten zu nutzen
