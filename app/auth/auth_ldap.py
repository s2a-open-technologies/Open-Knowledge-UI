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
Anmeldung gegen LDAP/Active Directory (``ldap3``).

Ablauf: optionaler Service-Bind -> Benutzer per Filter suchen (Eingabe mit
``escape_filter_chars`` maskiert) -> Bind als gefundener DN mit dem Passwort -> Anzeigename und
Gruppen lesen. TLS-Zertifikate werden standardmaessig geprueft.
"""

from logging import getLogger

from ldap3 import ALL, SUBTREE, SYNC, Connection, Server, Tls
from ldap3.core.exceptions import LDAPException
from ldap3.utils.conv import escape_filter_chars

from app.auth.auth_external import group_name_from

logger = getLogger(__name__)

# Testnaht: die Tests setzen hier die Mock-Strategie von ldap3
CLIENT_STRATEGY = SYNC


class LdapError(Exception):
    """Verbindungs- oder Konfigurationsproblem (lesbare Meldung fuer den Admin-Test)."""


def _server(cfg: dict, get_info=ALL) -> Server:
    from app.llm.registry import RemoteModelsError, _reject_link_local

    url = cfg["url"]
    try:
        _reject_link_local(
            url.replace("ldaps://", "https://").replace("ldap://", "http://")
        )
    except RemoteModelsError as e:
        raise LdapError(str(e)) from None
    import ssl

    tls = None
    if url.startswith("ldaps://") or cfg.get("starttls"):
        tls = Tls(
            validate=ssl.CERT_REQUIRED
            if cfg.get("verify_tls", True)
            else ssl.CERT_NONE,
            ca_certs_file=cfg.get("ca_file") or None,
        )
    return Server(
        url,
        use_ssl=url.startswith("ldaps://"),
        tls=tls,
        get_info=get_info,
        connect_timeout=cfg["timeout"],
    )


def _open(cfg: dict, user: str | None, password: str | None, server=None) -> Connection:
    server = server or _server(cfg)
    conn = Connection(
        server,
        user=user,
        password=password,
        receive_timeout=cfg["timeout"],
        raise_exceptions=False,
        auto_bind=False,
        client_strategy=CLIENT_STRATEGY,
    )
    try:
        conn.open()  # liefert nichts zurueck, Fehler kommen als Ausnahme
    except LDAPException as e:
        raise LdapError(
            f"Server nicht erreichbar: {type(e).__name__}: {e}"[:300]
        ) from None
    if conn.closed:
        raise LdapError(
            f"Server nicht erreichbar: {conn.last_error or 'keine Antwort'}"
        )
    if (
        cfg.get("starttls")
        and not cfg["url"].startswith("ldaps://")
        and not conn.start_tls()
    ):
        raise LdapError(f"StartTLS fehlgeschlagen: {conn.last_error}")
    return conn


def _first(value) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value or "").strip()


def find_user(cfg: dict, username: str) -> dict | None:
    """Sucht den Benutzer (Service-Bind oder anonym). Liefert ``{dn, username, name, groups}``."""
    conn = _open(cfg, cfg.get("bind_dn") or None, cfg.get("bind_password") or None)
    try:
        if cfg.get("bind_dn") and not conn.bind():
            raise LdapError(
                f"Service-Bind fehlgeschlagen: {conn.result.get('description', conn.last_error)}"
            )
        flt = cfg["user_filter"].replace("{username}", escape_filter_chars(username))
        attrs = [
            a for a in (cfg["username_attr"], cfg["name_attr"], cfg["group_attr"]) if a
        ]
        ok = conn.search(
            cfg["base_dn"], flt, search_scope=SUBTREE, attributes=attrs, size_limit=2
        )
        if not ok or not conn.entries:
            return None
        if len(conn.entries) > 1:
            logger.warning(
                "LDAP: der Benutzerfilter liefert mehrere Treffer, Anmeldung abgelehnt"
            )
            return None
        entry = conn.entries[0]
        data = entry.entry_attributes_as_dict
        groups = (
            [group_name_from(str(g)) for g in (data.get(cfg["group_attr"]) or [])]
            if cfg.get("group_attr")
            else []
        )
        return {
            "dn": entry.entry_dn,
            "username": _first(data.get(cfg["username_attr"])) or username,
            "name": _first(data.get(cfg["name_attr"])),
            "groups": [g for g in groups if g],
        }
    finally:
        conn.unbind()


def authenticate_ldap(cfg: dict, username: str, password: str) -> dict | None:
    """Prueft Benutzername und Passwort gegen das Verzeichnis.

    Returns:
        ``{external_id, username, display_name, groups}`` oder ``None`` bei falschen Daten.
    Raises:
        LdapError: bei Verbindungs-/Konfigurationsproblemen (nicht bei falschem Passwort).
    """
    if not username or not password:  # leeres Passwort waere ein anonymer Bind
        return None
    try:
        found = find_user(cfg, username)
        if found is None:
            return None
        conn = _open(cfg, found["dn"], password)
        try:
            if not conn.bind():
                return None
        finally:
            conn.unbind()
    except LDAPException as e:
        raise LdapError(f"{type(e).__name__}: {e}"[:300]) from None
    return {
        "external_id": found["dn"],
        "username": found["username"],
        "display_name": found["name"],
        "groups": found["groups"],
    }


def probe_ldap(cfg: dict, username: str = "", password: str = "") -> dict:
    """Fakten fuer den Admin: Server erreichbar, Service-Bind, optional Probe-Login (ohne Sitzung)."""
    result: dict = {"reachable": False}
    try:
        conn = _open(cfg, cfg.get("bind_dn") or None, cfg.get("bind_password") or None)
        result["reachable"] = True
        try:
            if cfg.get("bind_dn"):
                result["service_bind"] = bool(conn.bind())
                if not result["service_bind"]:
                    raise LdapError(
                        f"Service-Bind fehlgeschlagen: {conn.result.get('description', conn.last_error)}"
                    )
        finally:
            conn.unbind()
        if username:
            found = find_user(cfg, username)
            result["user_found"] = found is not None
            if found:
                result["groups"] = found["groups"]
                result["display_name"] = found["name"]
                if password:
                    result["login_ok"] = (
                        authenticate_ldap(cfg, username, password) is not None
                    )
    except LDAPException as e:
        raise LdapError(f"{type(e).__name__}: {e}"[:300]) from None
    return result
