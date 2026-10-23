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
Gemeinsamer Kern der externen Anmeldung (LDAP und OIDC).

- Konfiguration liegt in ``settings`` (``AUTH_LDAP``, ``AUTH_OIDC`` als JSON, Geheimnisse
  sensibel, nie in Antworten): ``load_config`` / ``public_config`` / ``save_config``.
- ``login_external_user`` legt das Konto beim ersten Login an und gleicht die Gruppen ab.
- Gruppen werden **nie importiert**: ``sync_groups`` ordnet nur ueber den Namen vorhandene
  lokale Gruppen zu und legt nie eine an. Nur vom Login gesetzte Mitgliedschaften
  (``group_members.external``) raeumt der Abgleich wieder ab, manuelle bleiben.
"""

import json
import secrets
from logging import getLogger

from sqlalchemy import func
from werkzeug.security import generate_password_hash

from app.auth.users import ROLES, _user_dict
from app.core.database_service import get_db_service
from app.core.models import Group, GroupMember, User
from app.core.settings_service import SettingsService

logger = getLogger(__name__)

SECRET_FIELDS = {"ldap": ("bind_password",), "oidc": ("client_secret",)}
SOURCES = ("ldap", "oidc")

LDAP_DEFAULTS = {
    "enabled": False,
    "url": "",
    "starttls": False,
    "verify_tls": True,
    "ca_file": "",
    "bind_dn": "",
    "bind_password": "",
    "base_dn": "",
    "user_filter": "(&(objectClass=person)(sAMAccountName={username}))",
    "username_attr": "sAMAccountName",
    "name_attr": "displayName",
    "group_attr": "memberOf",
    "timeout": 10,
}
OIDC_DEFAULTS = {
    "enabled": False,
    "name": "SSO",
    "issuer": "",
    "client_id": "",
    "client_secret": "",
    "scopes": "openid profile email",
    "username_claim": "preferred_username",
    "name_claim": "name",
    "groups_claim": "groups",
    "timeout": 10,
}


class AuthConfigError(ValueError):
    """Ungueltige Einstellung (lesbare Meldung fuer die Oberflaeche)."""


def _read(key: str) -> dict:
    session = get_db_service().get_session()
    try:
        raw = SettingsService(session).get(key, "{}") or "{}"
    finally:
        session.close()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def load_config() -> dict:
    """Aktuelle Einstellungen der Anmeldung (mit Geheimnissen, nur intern verwenden)."""
    session = get_db_service().get_session()
    try:
        s = SettingsService(session)
        role = s.get("AUTH_DEFAULT_ROLE", "user") or "user"
        return {
            "ldap": {**LDAP_DEFAULTS, **_read("AUTH_LDAP")},
            "oidc": {**OIDC_DEFAULTS, **_read("AUTH_OIDC")},
            "default_role": role if role in ROLES else "user",
            "auto_create": s.get_bool("AUTH_AUTO_CREATE", True),
            "base_url": (s.get("APP_BASE_URL", "") or "").strip().rstrip("/"),
        }
    finally:
        session.close()


def public_config(cfg: dict | None = None) -> dict:
    """Wie ``load_config``, aber ohne Geheimnisse (nur ``has_<feld>``)."""
    cfg = cfg or load_config()
    out = {k: v for k, v in cfg.items() if k not in SOURCES}
    for source in SOURCES:
        block = dict(cfg[source])
        for secret in SECRET_FIELDS[source]:
            block[f"has_{secret}"] = bool(block.pop(secret, ""))
        out[source] = block
    return out


def _clean_block(source: str, incoming: dict, current: dict) -> dict:
    defaults = LDAP_DEFAULTS if source == "ldap" else OIDC_DEFAULTS
    block = dict(current)
    for key, default in defaults.items():
        if key not in incoming:
            continue
        value = incoming[key]
        if key in SECRET_FIELDS[source]:
            if value:  # leer = gespeicherten Wert behalten
                block[key] = str(value)
            continue
        if isinstance(default, bool):
            block[key] = bool(value)
        elif isinstance(default, int):
            try:
                block[key] = max(1, min(120, int(value)))
            except (TypeError, ValueError):
                raise AuthConfigError(f"{key}: ganze Zahl erwartet") from None
        else:
            block[key] = str(value or "").strip()
    if incoming.get("clear_secret"):
        for secret in SECRET_FIELDS[source]:
            block[secret] = ""
    return block


def save_config(data: dict) -> dict:
    """Speichert die Einstellungen; leere Geheimnis-Felder behalten den gespeicherten Wert."""
    from app.llm.registry import RemoteModelsError, _reject_link_local

    current = load_config()
    updates: dict[str, str] = {}
    for source, key in (("ldap", "AUTH_LDAP"), ("oidc", "AUTH_OIDC")):
        if source in data:
            block = _clean_block(source, data[source] or {}, current[source])
            target = block["url"] if source == "ldap" else block["issuer"]
            if block["enabled"] and not target:
                raise AuthConfigError(
                    "LDAP: Adresse fehlt" if source == "ldap" else "OIDC: Issuer fehlt"
                )
            if target:
                scheme_ok = (
                    target.startswith(("ldap://", "ldaps://"))
                    if source == "ldap"
                    else target.startswith(("http://", "https://"))
                )
                if not scheme_ok:
                    raise AuthConfigError(
                        "LDAP: Adresse muss mit ldap:// oder ldaps:// beginnen"
                        if source == "ldap"
                        else "OIDC: Issuer muss mit http:// oder https:// beginnen"
                    )
                try:
                    _reject_link_local(
                        target.replace("ldaps://", "https://").replace(
                            "ldap://", "http://"
                        )
                    )
                except RemoteModelsError as e:
                    raise AuthConfigError(str(e)) from None
            if (
                source == "oidc"
                and block["enabled"]
                and not (block["client_id"] and block["client_secret"])
            ):
                raise AuthConfigError(
                    "OIDC: Client-ID und Client-Secret werden gebraucht"
                )
            if source == "ldap" and block["enabled"] and not block["base_dn"]:
                raise AuthConfigError("LDAP: Basis-DN fehlt")
            if (
                source == "ldap"
                and block["enabled"]
                and "{username}" not in block["user_filter"]
            ):
                raise AuthConfigError(
                    "LDAP: Der Benutzerfilter braucht den Platzhalter {username}"
                )
            updates[key] = json.dumps(block)
    if "default_role" in data:
        if data["default_role"] not in ROLES:
            raise AuthConfigError("Ungültige Standardrolle")
        updates["AUTH_DEFAULT_ROLE"] = data["default_role"]
    if "auto_create" in data:
        updates["AUTH_AUTO_CREATE"] = "true" if data["auto_create"] else "false"
    if "base_url" in data:
        url = str(data["base_url"] or "").strip().rstrip("/")
        if url and not url.startswith(("http://", "https://")):
            raise AuthConfigError(
                "Die App-Adresse muss mit http:// oder https:// beginnen"
            )
        updates["APP_BASE_URL"] = url
    session = get_db_service().get_session()
    try:
        SettingsService(session).set_batch(updates)
    finally:
        session.close()
    return public_config()


# --------------------------------------------------------------------------
# Konten und Gruppen
# --------------------------------------------------------------------------


def group_name_from(value: str) -> str:
    """Gruppenname aus einem Wert: bei einer LDAP-DN (``CN=Vertrieb,OU=...``) der CN."""
    text = (value or "").strip()
    if "=" in text and "," in text:
        first = text.split(",", 1)[0]
        if first.lower().startswith("cn="):
            return first[3:].strip()
    return text


def sync_groups(db, user_id: int, external_groups: list[str]) -> list[str]:
    """Ordnet den Benutzer vorhandenen lokalen Gruppen gleichen Namens zu.

    Legt nie Gruppen an. Vom Login gesetzte Mitgliedschaften, die der Anbieter nicht mehr
    meldet, werden entfernt; manuell gesetzte bleiben. Liefert die Namen der Zuordnung.
    """
    wanted = {group_name_from(g).lower() for g in external_groups if group_name_from(g)}
    local = {g.name.lower(): g for g in db.query(Group)}
    target_ids = {local[name].id for name in wanted if name in local}
    rows = {m.group_id: m for m in db.query(GroupMember).filter_by(user_id=user_id)}
    for gid in target_ids - rows.keys():
        db.add(GroupMember(group_id=gid, user_id=user_id, external=True))
    for gid, member in rows.items():
        if member.external and gid not in target_ids:
            db.delete(member)
    return sorted(local[n].name for n in wanted if n in local)


def _unique_username(db, wanted: str) -> str | None:
    name = (wanted or "").strip()
    if not name:
        return None
    exists = db.query(User).filter(func.lower(User.username) == name.lower()).first()
    return None if exists else name


class ExternalLoginError(Exception):
    """Externes Konto darf sich nicht anmelden (lesbare Meldung fuer das Log, nicht fuer Nutzer)."""


def login_external_user(
    source: str, external_id: str, username: str, display_name: str, groups: list[str]
) -> dict:
    """Legt das Konto beim ersten Login an, aktualisiert es sonst und gleicht die Gruppen ab.

    Raises:
        ExternalLoginError: Konto gesperrt, Neuanlage abgeschaltet oder Namenskollision mit einem
            lokalen Konto (lokale Konten bleiben lokal).
    """
    if source not in SOURCES:
        raise ExternalLoginError(f"Unbekannte Quelle {source}")
    cfg = load_config()
    db = get_db_service().get_session()
    try:
        user = (
            db.query(User)
            .filter_by(auth_source=source, external_id=external_id)
            .first()
        )
        if user is None and source == "ldap":
            # LDAP: der DN kann sich beim Umbenennen aendern, das Verzeichnis hat den Namen bestaetigt.
            # OIDC verknuepft nie ueber den Namen (preferred_username ist aenderbar und nicht eindeutig):
            # nur ueber die unveraenderliche sub, sonst waere Kontoübernahme moeglich.
            user = (
                db.query(User)
                .filter(
                    func.lower(User.username) == username.lower(),
                    User.auth_source == source,
                )
                .first()
            )
        if user is None:
            if not cfg["auto_create"]:
                raise ExternalLoginError("Automatisches Anlegen ist abgeschaltet")
            name = _unique_username(db, username)
            if name is None:
                raise ExternalLoginError(
                    f"Der Name „{username}“ ist bereits an ein anderes Konto vergeben"
                )
            user = User(
                username=name,
                display_name=display_name or name,
                password_hash=generate_password_hash(
                    secrets.token_urlsafe(32)
                ),  # unbrauchbar
                role=cfg["default_role"],
                is_active=True,
                auth_source=source,
            )
            db.add(user)
            db.flush()
        if not user.is_active:
            raise ExternalLoginError("Das Konto ist deaktiviert")
        user.external_id = external_id
        if display_name:
            user.display_name = display_name
        user.external_groups = json.dumps(
            sorted({g.strip() for g in groups if g and g.strip()})
        )
        sync_groups(db, user.id, groups)
        from app.auth import audit

        audit.record(
            audit.LOGIN_EXTERNAL_OK, actor_user_id=user.id, actor_name=user.username
        )
        db.commit()
        result = _user_dict(user)
        result["external_groups"] = json.loads(user.external_groups)
        return result
    except Exception as e:
        from app.auth import audit

        audit.record(
            audit.LOGIN_EXTERNAL_FAILED,
            actor_name=username,
            detail={"quelle": source, "grund": type(e).__name__},
        )
        db.rollback()
        raise
    finally:
        db.close()
