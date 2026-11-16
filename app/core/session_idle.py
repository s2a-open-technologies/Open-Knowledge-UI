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
"""Leerlauf-Zeitlimit der Sitzung: 30 Minuten ohne Aktivitaet, danach ist eine Anmeldung noetig.

Das Session-Cookie traegt ``Max-Age``. Starlette schreibt es aber nur neu, wenn sich die Sitzung
geaendert hat; ohne diese Middleware liefe die Frist also ab dem Login, egal was der Nutzer tut.
Sie vermerkt deshalb bei jeder Anfrage eines angemeldeten Nutzers die Zeit (hoechstens einmal pro
Minute), das Cookie wird neu ausgestellt und die Frist beginnt von vorn.

Dauer-Abfragen im Hintergrund (``EXEMPT_PATHS``) verlaengern die Sitzung nicht, sonst liefe sie
nie ab. Der Chat laeuft ueber einen WebSocket und kann kein Cookie erneuern; dafuer meldet das
Frontend echte Bedienung ueber ``POST /api/auth/keepalive``.
"""

import time

IDLE_SECONDS = 30 * 60
# Hoechstens so oft wird das Cookie neu ausgestellt
REFRESH_EVERY_SECONDS = 60
# Abfragen, die der Browser von allein im Takt stellt (keine Bedienung)
EXEMPT_PATHS = frozenset({"/api/ingestion-status"})


class SessionIdleMiddleware:
    """Liegt innerhalb der ``SessionMiddleware`` (braucht ``scope["session"]``)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] not in EXEMPT_PATHS:
            session = scope["session"]
            now = int(time.time())
            if (
                session.get("user_id") is not None
                and now - int(session.get("seen", 0)) >= REFRESH_EVERY_SECONDS
            ):
                session["seen"] = now  # markiert die Sitzung als geaendert: Cookie wird neu gesetzt
        await self.app(scope, receive, send)
