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
CSRF-Schutz fuer die Cookie-Sitzung.

Die Anmeldung steckt in einem Session-Cookie, den der Browser bei jeder Anfrage mitschickt, auch wenn
sie von einer fremden Seite ausgeloest wird. Darum:

* Jede Sitzung hat ein zufaelliges Token (``session["csrf_token"]``). Es steht zusaetzlich im Cookie
  ``csrf_token`` (lesbar fuer das Skript der eigenen Seite, nicht fuer Fremdseiten).
* Jede aendernde Anfrage (POST, PUT, PATCH, DELETE) muss es im Header ``X-CSRF-Token`` (oder als
  Formularfeld ``csrf_token``) mitschicken; es wird mit dem Token der Sitzung verglichen. Eine fremde
  Seite kann den Cookie weder lesen noch den Header setzen.
* Zusaetzlich ist der Session-Cookie ``SameSite=Lax`` (siehe ``configure_cookies``): er geht bei
  fremden Seiten nicht mit POST-Anfragen und nicht mit WebSocket-Verbindungen mit.

Nach Anmeldung/Abmeldung (``session.clear()``) bekommt die Sitzung ein neues Token; die Antwort setzt
den Cookie neu.
"""

import hmac
import os
import secrets
from urllib.parse import urlparse

from flask import current_app, jsonify, request, session

COOKIE = "csrf_token"
HEADER = "X-CSRF-Token"
FIELD = "csrf_token"
UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}


def _token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def configure_cookies(app) -> None:
    """Session-Cookie: nicht per Skript lesbar, nicht bei fremden Seiten; ``Secure`` per Umgebungsvariable."""
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = (
        app.config.get("SESSION_COOKIE_SAMESITE") or "Lax"
    )
    if os.getenv("SESSION_COOKIE_SECURE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        app.config["SESSION_COOKIE_SECURE"] = True


def same_origin(origin: str | None, host: str | None) -> bool:
    """Stammt ``Origin`` von derselben Adresse wie ``Host`` (fuer WebSockets; fehlender Origin = kein Browser)."""
    if not origin:
        return True
    try:
        return bool(host) and urlparse(origin).netloc.lower() == host.lower()
    except ValueError:
        return False


def init_csrf(app) -> None:
    configure_cookies(app)

    @app.before_request
    def check_csrf():
        if request.method not in UNSAFE or not current_app.config.get(
            "CSRF_ENABLED", True
        ):
            return None
        expected = session.get("csrf_token")
        sent = request.headers.get(HEADER) or request.form.get(FIELD) or ""
        if expected and sent and hmac.compare_digest(str(expected), str(sent)):
            return None
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Sicherheitsprüfung fehlgeschlagen (CSRF-Token fehlt oder ist ungültig). "
                    "Bitte die Seite neu laden.",
                }
            ),
            403,
        )

    @app.after_request
    def issue_cookie(response):
        if (
            not current_app.config.get("CSRF_ENABLED", True)
            or request.endpoint == "static"
        ):
            return response
        token = _token()
        if request.cookies.get(COOKIE) != token:
            response.set_cookie(
                COOKIE,
                token,
                httponly=False,  # das Skript der eigenen Seite muss es lesen, um es als Header zu senden
                samesite="Lax",
                secure=bool(current_app.config.get("SESSION_COOKIE_SECURE")),
                path="/",
            )
        return response
