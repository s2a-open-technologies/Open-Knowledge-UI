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
CSRF-Schutz fuer die Cookie-Sitzung (ASGI-Middleware).

Die Anmeldung steckt in einem Session-Cookie, den der Browser bei jeder Anfrage mitschickt, auch wenn
sie von einer fremden Seite ausgeloest wird. Darum:

* Jede Sitzung hat ein zufaelliges Token (``session["csrf_token"]``). Es steht zusaetzlich im Cookie
  ``csrf_token`` (lesbar fuer das Skript der eigenen Seite, nicht fuer Fremdseiten).
* Jede aendernde Anfrage (POST, PUT, PATCH, DELETE) muss es im Header ``X-CSRF-Token`` mitschicken;
  es wird mit dem Token der Sitzung verglichen. Eine fremde Seite kann den Cookie weder lesen noch
  den Header setzen.
* Zusaetzlich ist der Session-Cookie ``SameSite=Lax`` (siehe ``cookie_secure`` und ``create_app``):
  er geht bei fremden Seiten nicht mit POST-Anfragen und nicht mit WebSocket-Verbindungen mit.
  WebSockets pruefen ausserdem ``Origin`` (:func:`same_origin`).

Nach Anmeldung/Abmeldung (``session.clear()``) bekommt die Sitzung ein neues Token; die Antwort setzt
den Cookie neu. Abschaltbar ueber ``app.state.csrf_enabled`` (nur Tests).
"""

import hmac
import json
import os
import secrets
from http.cookies import SimpleCookie
from urllib.parse import urlparse

COOKIE = "csrf_token"
HEADER = "X-CSRF-Token"
UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}
_REFUSED = {
    "success": False,
    "error": "Sicherheitsprüfung fehlgeschlagen (CSRF-Token fehlt oder ist ungültig). "
    "Bitte die Seite neu laden.",
}


def cookie_secure() -> bool:
    """``Secure``-Flag fuer die Cookies (HTTPS-Bereitstellung), nur aus der Umgebung."""
    return os.getenv("SESSION_COOKIE_SECURE", "").strip().lower() in ("1", "true", "yes", "on")


def same_origin(origin: str | None, host: str | None) -> bool:
    """Stammt ``Origin`` von derselben Adresse wie ``Host`` (fuer WebSockets; fehlender Origin = kein Browser)."""
    if not origin:
        return True
    try:
        return bool(host) and urlparse(origin).netloc.lower() == host.lower()
    except ValueError:
        return False


def _token(session: dict) -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def _header(scope, name: str) -> str:
    wanted = name.lower().encode("latin-1")
    for key, value in scope.get("headers", []):
        if key == wanted:
            return value.decode("latin-1")
    return ""


class CSRFMiddleware:
    """Liegt innerhalb der ``SessionMiddleware`` (braucht ``scope["session"]``)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not getattr(scope["app"].state, "csrf_enabled", True):
            return await self.app(scope, receive, send)
        session = scope["session"]
        if scope["method"] in UNSAFE:
            expected = session.get("csrf_token")
            sent = _header(scope, HEADER)
            if not (expected and sent and hmac.compare_digest(str(expected), str(sent))):
                body = json.dumps(_REFUSED).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        if scope["path"].startswith("/static/"):
            return await self.app(scope, receive, send)

        cookies = SimpleCookie(_header(scope, "cookie"))
        sent_cookie = cookies[COOKIE].value if COOKIE in cookies else None

        async def send_with_cookie(message):
            if message["type"] == "http.response.start":
                token = _token(session)  # nach dem Handler: Login/Logout haben die Sitzung neu gemacht
                if sent_cookie != token:
                    cookie = f"{COOKIE}={token}; Path=/; SameSite=Lax"
                    if cookie_secure():
                        cookie += "; Secure"
                    message.setdefault("headers", [])
                    message["headers"] = list(message["headers"]) + [
                        (b"set-cookie", cookie.encode("latin-1"))
                    ]
            await send(message)

        await self.app(scope, receive, send_with_cookie)
