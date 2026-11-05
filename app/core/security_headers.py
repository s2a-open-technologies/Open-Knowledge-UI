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
Sicherheits-Header auf jeder Antwort (Middleware, in ``create_app`` registriert).

Setzt pro Antwort:

* ``X-Content-Type-Options: nosniff`` — kein MIME-Sniffing,
* ``X-Frame-Options: DENY`` und CSP ``frame-ancestors 'none'`` — keine Einbettung in Frames,
* ``Referrer-Policy: strict-origin-when-cross-origin`` — duennt den Verweisaufenthalt,
* ``Content-Security-Policy`` — Skripte und Verbindungen nur von gleicher Herkunft. Fuer
  Skripte und Stile bleibt ``'unsafe-inline'`` stehen, weil der Theme-Bootstrap inline in
  ``index.html`` laeuft; der Schutz bleibt, dass Skripte von fremden Urspruegen geblockt
  werden. Fremde Bilder im Chat-Markdown sind erlaubt (``img-src ... https:``), weil der
  Markdown-Renderer sie freigibt (rehypeSanitize-Standard).
* ``Strict-Transport-Security`` nur bei HTTPS-Bereitstellung (``SESSION_COOKIE_SECURE``).

Mit `SECURITY_HEADERS=off` (Umgebung) abschaltbar — wenn ein Reverse Proxy die Header
selbst setzt und doppelte Header Probleme machen.
"""

# Standard library
import os

CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob: https:; "
    "font-src 'self' data:; "
    "connect-src 'self'; "
    "worker-src 'self' blob:; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'none'; "
    "form-action 'self'"
)

HSTS = "max-age=31536000; includeSubDomains"


def enabled() -> bool:
    return (os.getenv("SECURITY_HEADERS") or "on").strip().lower() != "off"


class SecurityHeadersMiddleware:
    """Setzt die Header auf jeder HTTP-Antwort (ASGI)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not enabled():
            return await self.app(scope, receive, send)
        from app.core.csrf import cookie_secure

        secure = getattr(scope["app"].state, "cookie_secure", None)
        secure = cookie_secure() if secure is None else secure

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = [
                    (k, v) for k, v in message.get("headers", []) if k.lower() not in _SET
                ]
                present = {k.lower() for k, _ in headers}
                # Nicht ueberschreiben, wenn ein Proxy/Handler schon einen CSP bzw. HSTS setzt.
                if b"content-security-policy" not in present:
                    headers.append((b"content-security-policy", CSP.encode()))
                headers += [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                ]
                if secure and b"strict-transport-security" not in present:
                    headers.append((b"strict-transport-security", HSTS.encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


_SET = {b"x-content-type-options", b"x-frame-options", b"referrer-policy"}
