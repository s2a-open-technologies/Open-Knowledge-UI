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

"""Seiten ohne /api: SPA-Auslieferung (mit 404-Fallback fuer Client-Routen) und Health-Check."""

import os
from logging import getLogger
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app import resources_ready
from app.api.schemas.pages import Health
from app.core.database_service import get_db_service
from app.core.settings_service import SettingsService

router = APIRouter()
logger = getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


def serve_spa() -> HTMLResponse:
    """Liefert die gebaute React-SPA (app/static/index.html).

    Der Vite-Build schreibt direkt nach ``app/static/`` (siehe
    ``frontend/vite.config.ts``); die App serviert dieses Verzeichnis unter
    ``/static/...``. Fällt mit einer klaren Meldung zurück, falls das Frontend
    noch nicht gebaut wurde (``cd frontend && npm run build``)."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        return HTMLResponse(
            "<h1>Frontend nicht gebaut</h1>"
            "<p>Bitte zuerst ausführen: <code>cd frontend &amp;&amp; npm install "
            "&amp;&amp; npm run build</code></p>",
            status_code=503,
        )

    # Server-seitigen Theme-Default (UI_THEME) in die SPA-Index einbetten,
    # damit index.html vor dem ersten Paint den richtigen Fallback hat.
    try:
        settings = SettingsService(get_db_service().get_session())
        ui_theme = settings.get("UI_THEME", "system") or "system"
    except Exception:
        ui_theme = "system"

    html = index_path.read_text(encoding="utf-8")
    return HTMLResponse(html.replace("__RAG_UI_THEME__", ui_theme))


@router.get("/health", response_model=Health)
def health():
    """Health check endpoint for Docker (Liveness). ready=False waehrend
    die Embedding-Modelle im Hintergrund geladen werden."""
    from app.core import redis_bus
    from app.core.version import get_version

    return Health(
        status="healthy",
        ready=resources_ready(),
        version=get_version(),
        redis=redis_bus.status(),
    )


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    # Die React-SPA übernimmt das Routing (Chat unter '/'). Der Login-Status
    # wird clientseitig über /api/auth/me geprüft.
    return serve_spa()


# Alle nicht von der API/Assets bedienten GET-Pfade sind clientseitige
# React-Router-Routen und liefern dieselbe index.html aus. Bewusst über den
# 404-Handler (echter Catch-all, siehe app/api/errors.py) statt einer Routen-Whitelist:
# neue SPA-Seiten funktionieren damit automatisch, ohne hier etwas nachzutragen.
_RESERVED_PREFIXES = ("/api/", "/static/", "/ws/")


_RESERVED_EXACT = {"/health"}


def spa_fallback(request: Request) -> HTMLResponse | None:
    """index.html fuer Client-Routen; ``None`` = echte 404 (API, Assets, Nicht-GET, Dateien)."""
    path = request.url.path

    # Backend-/API-/Asset-Pfade niemals mit der SPA überschreiben.
    if (
        request.method != "GET"
        or path.startswith(_RESERVED_PREFIXES)
        or path in _RESERVED_EXACT
    ):
        return None

    # Nur Navigation (HTML) oder extension-lose Pfade als SPA ausliefern;
    # fehlende Dateien mit Endung bleiben echte 404s.
    accepts_html = "text/html" in request.headers.get("accept", "")
    looks_like_file = "." in os.path.basename(path)
    if accepts_html or not looks_like_file:
        return serve_spa()
    return None
