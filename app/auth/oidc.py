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
Anmeldung per OpenID Connect (Authorization-Code-Fluss mit PKCE).

Ablauf: ``start_login`` (Discovery, ``state``/``nonce``/PKCE in der Session, Redirect-URL) ->
``finish_login`` (State pruefen, Code tauschen, ID-Token ueber JWKS verifizieren:
Signatur, ``iss``, ``aud``, ``exp``, ``nonce``). Alle Adressen des Anbieters werden gegen
Link-Local-Ziele geprueft, Weiterleitungen nicht verfolgt.
"""

import base64
import hashlib
import secrets
import time
from logging import getLogger
from urllib.parse import urlencode

import jwt
import requests

logger = getLogger(__name__)

_DISCOVERY_TTL = 300
_discovery: dict[str, tuple[float, dict]] = {}
ALGORITHMS = ["RS256", "RS384", "RS512", "ES256", "ES384", "PS256"]


class OidcError(Exception):
    """Fehler im Ablauf (lesbare Meldung; Nutzern zeigt die App nur eine allgemeine)."""


def _check(url: str) -> str:
    from app.llm.registry import RemoteModelsError, _reject_link_local

    try:
        _reject_link_local(url)
    except RemoteModelsError as e:
        raise OidcError(str(e)) from None
    return url


def _get(url: str, timeout: int) -> dict:
    try:
        resp = requests.get(_check(url), timeout=timeout, allow_redirects=False)
    except requests.RequestException as e:
        raise OidcError(f"{url} nicht erreichbar: {type(e).__name__}") from None
    if resp.status_code != 200:
        raise OidcError(f"{url}: HTTP {resp.status_code}")
    try:
        return resp.json()
    except ValueError:
        raise OidcError(f"{url}: keine gültige JSON-Antwort") from None


def discover(cfg: dict, force: bool = False) -> dict:
    issuer = cfg["issuer"].rstrip("/")
    cached = _discovery.get(issuer)
    if cached and not force and time.time() - cached[0] < _DISCOVERY_TTL:
        return cached[1]
    doc = _get(f"{issuer}/.well-known/openid-configuration", cfg["timeout"])
    if doc.get("issuer", "").rstrip("/") != issuer:
        raise OidcError(
            "Der Issuer der Discovery-Antwort passt nicht zur eingetragenen Adresse"
        )
    for key in ("authorization_endpoint", "token_endpoint", "jwks_uri"):
        if not doc.get(key):
            raise OidcError(f"Discovery ohne {key}")
        _check(doc[key])
    _discovery[issuer] = (time.time(), doc)
    return doc


def redirect_uri(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/auth/oidc/callback"


def start_login(cfg: dict, base_url: str, session) -> str:
    """Bereitet den Login vor und liefert die Adresse des Anbieters (State/Nonce/PKCE in der Session)."""
    doc = discover(cfg)
    state, nonce = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    session["oidc"] = {
        "state": state,
        "nonce": nonce,
        "verifier": verifier,
        "at": time.time(),
    }
    query = urlencode(
        {
            "response_type": "code",
            "client_id": cfg["client_id"],
            "redirect_uri": redirect_uri(base_url),
            "scope": cfg["scopes"],
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    return f"{doc['authorization_endpoint']}?{query}"


def _decode(jwks, id_token: str, cfg: dict, doc: dict, nonce: str) -> dict:
    key = jwks.get_signing_key_from_jwt(id_token).key
    claims = jwt.decode(
        id_token,
        key,
        algorithms=ALGORITHMS,
        audience=cfg["client_id"],
        issuer=doc["issuer"],
        options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        leeway=30,
    )
    if not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
        raise OidcError("Nonce stimmt nicht überein")
    return claims


def finish_login(cfg: dict, base_url: str, session, args: dict) -> dict:
    """Schliesst den Login ab. Liefert ``{external_id, username, display_name, groups}``."""
    saved = session.pop("oidc", None)
    if not saved or time.time() - saved["at"] > 600:
        raise OidcError("Anmeldung abgelaufen, bitte erneut starten")
    if args.get("error"):
        raise OidcError(f"Anbieter meldet: {args.get('error')}")
    if not args.get("state") or not secrets.compare_digest(
        str(args["state"]), saved["state"]
    ):
        raise OidcError("State stimmt nicht überein")
    code = args.get("code")
    if not code:
        raise OidcError("Kein Code vom Anbieter")
    doc = discover(cfg)
    try:
        resp = requests.post(
            _check(doc["token_endpoint"]),
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri(base_url),
                "client_id": cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "code_verifier": saved["verifier"],
            },
            timeout=cfg["timeout"],
            allow_redirects=False,
        )
    except requests.RequestException as e:
        raise OidcError(
            f"Token-Endpoint nicht erreichbar: {type(e).__name__}"
        ) from None
    if resp.status_code != 200:
        raise OidcError(f"Token-Endpoint: HTTP {resp.status_code}")
    id_token = (resp.json() or {}).get("id_token")
    if not id_token:
        raise OidcError("Keine ID-Token in der Antwort")
    claims = _verify_token(cfg, doc, id_token, saved["nonce"])

    name = str(
        claims.get(cfg["username_claim"]) or claims.get("email") or claims["sub"]
    )
    raw_groups = claims.get(cfg["groups_claim"]) or []
    groups = (
        [raw_groups] if isinstance(raw_groups, str) else [str(g) for g in raw_groups]
    )
    return {
        "external_id": str(claims["sub"]),
        "username": name,
        "display_name": str(claims.get(cfg["name_claim"]) or ""),
        "groups": groups,
    }


def _verify_token(cfg: dict, doc: dict, id_token: str, nonce: str) -> dict:
    try:
        jwks = jwt.PyJWKClient(
            _check(doc["jwks_uri"]), timeout=cfg["timeout"], cache_keys=True
        )
        return _decode(jwks, id_token, cfg, doc, nonce)
    except jwt.PyJWTError as e:
        raise OidcError(f"ID-Token ungültig: {type(e).__name__}") from None


def probe_oidc(cfg: dict, base_url: str) -> dict:
    """Fakten fuer den Admin: Discovery erreichbar, Endpunkte, Rueckruf-Adresse."""
    doc = discover(cfg, force=True)
    _get(doc["jwks_uri"], cfg["timeout"])
    return {
        "issuer": doc["issuer"],
        "authorization_endpoint": doc["authorization_endpoint"],
        "token_endpoint": doc["token_endpoint"],
        "jwks_reachable": True,
        "redirect_uri": redirect_uri(base_url) if base_url else None,
        "pkce_supported": "S256"
        in (doc.get("code_challenge_methods_supported") or ["S256"]),
    }
