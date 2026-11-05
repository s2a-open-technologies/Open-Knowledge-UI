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
Authentifizierung und Rollenpruefung (FastAPI-Dependencies).

Das Session-Cookie (Starlette ``SessionMiddleware``) speichert nur ``user_id``. Benutzer und Rolle
werden pro Request aus der DB geladen, damit Aenderungen (Deaktivieren, Rollenwechsel) sofort
wirken. Fehler: 401 ``{"error": "Nicht authentifiziert"}``, 403 ``{"error": "Keine Berechtigung"}``.

In Handlern::

    def route(user: CurrentUser): ...           # angemeldet
    def route(user: Editor): ...                # mindestens editor
    def route(user: Admin): ...                 # admin
"""

from typing import Annotated

from fastapi import Depends
from starlette.requests import HTTPConnection

from app.api.errors import ApiError
from app.core import request_context
from app.auth.users import get_user, role_at_least

_UNSET = object()


def current_user(conn: HTTPConnection) -> dict | None:
    """Angemeldeter, aktiver Benutzer des Requests (auch WebSocket) oder ``None`` (pro Request gecacht)."""
    cached = getattr(conn.state, "current_user", _UNSET)
    if cached is not _UNSET:
        return cached
    user = None
    user_id = conn.session.get("user_id")
    if user_id is not None:
        user = get_user(user_id)
        if user is None or not user["is_active"]:
            conn.session.pop("user_id", None)
            user = None
    conn.state.current_user = user
    request_context.set_actor(user)
    return user


def require_user(conn: HTTPConnection) -> dict:
    user = current_user(conn)
    if user is None:
        raise ApiError(401, "Nicht authentifiziert")
    return user


def require_role(role: str):
    """Dependency: 401 ohne Anmeldung, 403 bei zu niedriger Rolle (``editor`` erlaubt auch ``admin``)."""

    def dependency(user: Annotated[dict, Depends(require_user)]) -> dict:
        if not role_at_least(user["role"], role):
            raise ApiError(403, "Keine Berechtigung")
        return user

    return dependency


def login_session(conn: HTTPConnection, user_id: int) -> None:
    """Neue Sitzung fuer ``user_id`` (alte Inhalte verwerfen: verhindert Session-Fixation)."""
    conn.session.clear()
    conn.session["user_id"] = user_id
    conn.state.current_user = _UNSET


OptionalUser = Annotated[dict | None, Depends(current_user)]
CurrentUser = Annotated[dict, Depends(require_user)]
Editor = Annotated[dict, Depends(require_role("editor"))]
Admin = Annotated[dict, Depends(require_role("admin"))]
