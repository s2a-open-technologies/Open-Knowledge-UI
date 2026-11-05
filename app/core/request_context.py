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

"""Request-Kontext ohne Framework-Globals: offene DB-Sessions, Client-Adresse, angemeldeter Nutzer.

Die Middleware in ``create_app`` startet je Request einen Kontext und schliesst am Ende alle
Sessions, die darin per ``DatabaseService.get_session`` geoeffnet wurden (auch wenn der Aufrufer
es vergisst). Sync-Handler und -Dependencies laufen im Threadpool mit kopiertem Kontext; deshalb
liegen Sessions und Nutzer in veraenderbaren Objekten, die alle Kopien gemeinsam sehen.
"""

from contextvars import ContextVar

_sessions: ContextVar[list | None] = ContextVar("request_db_sessions", default=None)
_client_ip: ContextVar[str | None] = ContextVar("request_client_ip", default=None)
_actor: ContextVar[dict | None] = ContextVar("request_actor", default=None)


def begin(client_ip: str | None):
    """Kontext fuer einen Request starten; das Ergebnis geht an :func:`end`."""
    return _sessions.set([]), _client_ip.set(client_ip), _actor.set({"user": None})


def end(tokens) -> None:
    sessions_token, ip_token, actor_token = tokens
    for session in _sessions.get() or []:
        try:
            session.close()  # idempotent; bereits geschlossene Sessions schaden nicht
        except Exception:
            pass
    _sessions.reset(sessions_token)
    _client_ip.reset(ip_token)
    _actor.reset(actor_token)


def track_session(session) -> None:
    """Session dem laufenden Request zuordnen (ausserhalb eines Requests: nichts tun)."""
    opened = _sessions.get()
    if opened is not None:
        opened.append(session)


def client_ip() -> str | None:
    return _client_ip.get()


def set_actor(user: dict | None) -> None:
    """Angemeldeten Nutzer des Requests merken (fuer das Audit-Log)."""
    holder = _actor.get()
    if holder is not None:
        holder["user"] = user


def actor() -> dict | None:
    holder = _actor.get()
    return holder["user"] if holder else None


class RequestContextMiddleware:
    """Je HTTP-Request einen Kontext starten und am Ende offene DB-Sessions schliessen (ASGI)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        client = scope.get("client")
        tokens = begin(client[0] if client else None)
        status = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
            if status["code"] < 400:
                await _audit_config_change(scope.get("method", ""), scope.get("path", ""))
        finally:
            end(tokens)


async def _audit_config_change(method: str, path: str) -> None:
    """Erfolgreiche Konfigurationsaenderungen protokollieren (ohne Request-Body: er kann
    Zugangsdaten enthalten). Best-effort wie das ganze Audit-Log."""
    from app.auth import audit

    target = audit.config_change_target(method, path)
    if target is None or actor() is None:
        return
    from starlette.concurrency import run_in_threadpool

    try:
        await run_in_threadpool(
            audit.record,
            audit.CONFIG_CHANGE,
            object_type=target,
            object_id=path,
            detail={"methode": method.upper()},
        )
    except Exception:  # noqa: BLE001 — Audit darf nie den Request brechen
        pass
