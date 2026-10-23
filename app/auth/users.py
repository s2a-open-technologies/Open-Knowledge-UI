# Open Knowledge UI - A privacy-focused, self-hosted RAG system
# SPDX-FileCopyrightText: 2026 S2A Open Technologies gGmbH
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 S2A Open Technologies gGmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Benutzer, Rollen und Anmeldung.

Rollen sind fest und aufsteigend: ``user`` < ``editor`` < ``admin``.
Rueckgaben sind dicts (nie ORM-Objekte oder Passwort-Hashes).
"""

import json
import threading
import time
from logging import getLogger

from sqlalchemy import func
from werkzeug.security import check_password_hash, generate_password_hash

from app.core.database_service import get_db_service
from app.core.models import ChatSession, User
from app.core.settings_service import SettingsService

logger = getLogger(__name__)

ROLES = ("user", "editor", "admin")

# Fehlversuche: nach MAX_FAILURES (pro Konto) bzw. MAX_FAILURES_PER_IP (pro
# Client-IP) innerhalb WINDOW_SECONDS wird die Anmeldung voruebergehend gesperrt.
# Gezaehlt wird pro aufgeloestem Konto (nicht pro eingegebener Schreibweise) und
# zusaetzlich pro IP, damit sich der Schutz nicht durch wechselnde Namen oder
# Gross-/Kleinschreibung umgehen laesst. In-Memory, ein gunicorn-Worker
# (siehe AGENTS.md).
MAX_FAILURES = 5
MAX_FAILURES_PER_IP = 20
WINDOW_SECONDS = 300
_failures: dict[str, list[float]] = {}
_failures_lock = threading.Lock()

# Hash eines Zufallswerts: Pruefung auch bei unbekanntem Namen, damit die
# Antwortzeit nicht verraet, ob ein Benutzer existiert.
_DUMMY_HASH = generate_password_hash("dummy-nicht-verwendet")


def role_at_least(role: str, required: str) -> bool:
    return ROLES.index(role) >= ROLES.index(required) if role in ROLES else False


def _user_dict(u: User) -> dict:
    try:
        groups = json.loads(u.external_groups) if u.external_groups else []
    except ValueError:
        groups = []
    return {
        "id": u.id,
        "username": u.username,
        "display_name": u.display_name or u.username,
        "role": u.role,
        "is_active": bool(u.is_active),
        "auth_source": u.auth_source or "local",
        "external_groups": groups,
    }


def ensure_admin_user(db_session) -> None:
    """Seedet beim ersten Start den Benutzer ``admin`` (idempotent) und ordnet
    Chats ohne Besitzer ihm zu.

    Das Passwort ist das bisherige App-Passwort (``APP_PASSWORD_HASH``; Default
    ``admin``), damit bestehende Installationen ihr gesetztes Passwort behalten.
    """
    admin = db_session.query(User).filter_by(role="admin").order_by(User.id).first()
    if admin is None:
        password_hash = SettingsService(db_session).get("APP_PASSWORD_HASH") or (
            generate_password_hash("admin")
        )
        admin = User(
            username="admin",
            display_name="Administrator",
            password_hash=password_hash,
            role="admin",
            is_active=True,
        )
        db_session.add(admin)
        db_session.flush()
        logger.info("Standard-Benutzer 'admin' angelegt")

    db_session.query(ChatSession).filter(ChatSession.user_id.is_(None)).update(
        {ChatSession.user_id: admin.id}
    )


def get_user(user_id: int) -> dict | None:
    db = get_db_service().get_session()
    try:
        u = db.query(User).filter_by(id=user_id).first()
        return _user_dict(u) if u else None
    finally:
        db.close()


def _throttled(key: str, limit: int) -> bool:
    now = time.time()
    with _failures_lock:
        recent = [t for t in _failures.get(key, []) if now - t < WINDOW_SECONDS]
        _failures[key] = recent
        return len(recent) >= limit


def _record_failure(key: str) -> None:
    with _failures_lock:
        _failures.setdefault(key, []).append(time.time())


def reset_throttle() -> None:
    with _failures_lock:
        _failures.clear()


class LoginThrottledError(Exception):
    """Zu viele Fehlversuche fuer diesen Benutzernamen."""


def authenticate(
    username: str, password: str, client_ip: str | None = None
) -> dict | None:
    """Prueft Zugangsdaten.

    Der Benutzername wird exakt (ohne Gross-/Kleinschreibung, ohne Platzhalter)
    verglichen.

    Returns:
        Den Benutzer oder ``None`` bei falschen Daten bzw. deaktiviertem Konto.

    Raises:
        LoginThrottledError: bei zu vielen Fehlversuchen fuer das Konto oder die IP.
    """
    name = (username or "").strip().lower()
    ip_key = f"ip:{client_ip}" if client_ip else None

    db = get_db_service().get_session()
    try:
        u = (
            db.query(User).filter(func.lower(User.username) == name).first()
            if name
            else None
        )
        # Pro Konto zaehlen (auch bei Schreibvarianten), unbekannte Namen pro Name.
        account_key = f"user:{u.id}" if u else f"name:{name}"

        if _throttled(account_key, MAX_FAILURES) or (
            ip_key and _throttled(ip_key, MAX_FAILURES_PER_IP)
        ):
            from app.auth import audit

            audit.record(
                audit.LOGIN_THROTTLED,
                actor_user_id=u.id if u else None,
                actor_name=name or None,
                detail={"grund": "gesperrt", "client_ip": client_ip},
            )
            raise LoginThrottledError()

        source = (u.auth_source or "local") if u else None
        if u is None or source == "local":
            ok = check_password_hash(
                u.password_hash if u else _DUMMY_HASH, password or ""
            )
            if u and ok and u.is_active:
                with _failures_lock:
                    _failures.pop(account_key, None)
                from app.auth import audit

                audit.record(audit.LOGIN_OK, actor_user_id=u.id, actor_name=u.username)
                return _user_dict(u)
        # LDAP: bestehendes LDAP-Konto oder unbekannter Name (lokale Konten bleiben lokal)
        if u is None or source == "ldap":
            ldap_user = _try_ldap(name or username, password)
            if ldap_user is not None:
                with _failures_lock:
                    _failures.pop(account_key, None)
                return ldap_user
        # OIDC-Konten haben kein Passwort hier: sie melden sich nur ueber den Anbieter an
    finally:
        db.close()

    from app.auth import audit

    audit.record(
        audit.LOGIN_FAILED,
        actor_name=name or None,
        detail={"grund": "falsche_angaben"},
    )
    _record_failure(account_key)
    if ip_key:
        _record_failure(ip_key)
    return None


def _try_ldap(username: str, password: str) -> dict | None:
    """LDAP-Anmeldung, falls eingerichtet. Fehler des Verzeichnisses gelten als nicht angemeldet."""
    try:
        from app.auth import external as auth_external
        from app.auth import ldap as auth_ldap

        cfg = auth_external.load_config()
        if not cfg["ldap"]["enabled"] or not username or not password:
            return None
        found = auth_ldap.authenticate_ldap(cfg["ldap"], username, password)
        if found is None:
            return None
        return auth_external.login_external_user(
            "ldap",
            found["external_id"],
            found["username"],
            found["display_name"],
            found["groups"],
        )
    except Exception as e:  # LdapError, ExternalLoginError, ...
        logger.warning(f"LDAP-Anmeldung fehlgeschlagen: {type(e).__name__}: {e}")
        return None


def verify_password(user_id: int, password: str) -> bool:
    db = get_db_service().get_session()
    try:
        u = db.query(User).filter_by(id=user_id).first()
        if u and (u.auth_source or "local") != "local":
            return False  # externe Konten haben hier kein Passwort
        return bool(u and check_password_hash(u.password_hash, password or ""))
    finally:
        db.close()


def set_password(user_id: int, new_password: str) -> bool:
    db = get_db_service().get_session()
    try:
        u = db.query(User).filter_by(id=user_id).first()
        if not u or (u.auth_source or "local") != "local":
            return False  # Passwort externer Konten verwaltet der Anbieter
        u.password_hash = generate_password_hash(new_password)
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
