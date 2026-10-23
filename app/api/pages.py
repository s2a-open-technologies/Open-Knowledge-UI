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

"""Seiten ohne /api: SPA-Auslieferung, Login/Logout (Formular), Health-Check."""

import os
from logging import getLogger

from flask import (
    Blueprint,
    Response,
    current_app,
    flash,
    redirect,
    request,
    session,
    url_for,
)

from app import resources_ready
from app.auth.users import LoginThrottledError, authenticate
from app.core.database_service import get_db_service
from app.core.settings_service import SettingsService

bp = Blueprint("main", __name__)
logger = getLogger(__name__)


_STATIC_DIR = None


def serve_spa():
    """Liefert die gebaute React-SPA (app/static/index.html).

    Der Vite-Build schreibt direkt nach ``app/static/`` (siehe
    ``frontend/vite.config.ts``); Flask serviert dieses Verzeichnis unter
    ``/static/...``. Fällt mit einer klaren Meldung zurück, falls das Frontend
    noch nicht gebaut wurde (``cd frontend && npm run build``)."""
    global _STATIC_DIR
    if _STATIC_DIR is None:
        _STATIC_DIR = os.path.join(current_app.root_path, "static")

    index_path = os.path.join(_STATIC_DIR, "index.html")
    if not os.path.isfile(index_path):
        return (
            (
                "<h1>Frontend nicht gebaut</h1>"
                "<p>Bitte zuerst ausführen: <code>cd frontend &amp;&amp; npm install "
                "&amp;&amp; npm run build</code></p>"
            ),
            503,
        )

    # Server-seitigen Theme-Default (UI_THEME) in die SPA-Index einbetten,
    # damit index.html vor dem ersten Paint den richtigen Fallback hat.
    try:
        settings = SettingsService(get_db_service().get_session())
        ui_theme = settings.get("UI_THEME", "system") or "system"
    except Exception:
        ui_theme = "system"

    with open(index_path, encoding="utf-8") as fh:
        html = fh.read()
    html = html.replace("__RAG_UI_THEME__", ui_theme)
    return Response(html, mimetype="text/html")


@bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint for Docker (Liveness). ready=False waehrend
    die Embedding-Modelle im Hintergrund geladen werden."""
    from app.core.version import get_version

    return {
        "status": "healthy",
        "ready": resources_ready(),
        "version": get_version(),
    }, 200


@bp.route("/", methods=["GET"])
def index():
    # Die React-SPA übernimmt das Routing (Chat unter '/'). Der Login-Status
    # wird clientseitig über /api/auth/me geprüft.
    return serve_spa()


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        try:
            user = authenticate(
                request.form.get("username", ""),
                request.form.get("password", ""),
                client_ip=request.remote_addr,
            )
        except LoginThrottledError:
            user = None
            flash("Zu viele Fehlversuche. Bitte später erneut versuchen.")
        if user:
            session.clear()
            session["user_id"] = user["id"]
            return redirect(url_for("main.index"))

        flash("Benutzername oder Passwort falsch")

    return serve_spa()


@bp.route("/logout", methods=["GET"])
def logout():
    from app.auth import audit

    audit.record(audit.LOGOUT)
    session.clear()
    return redirect(url_for("main.login"))


# Alle nicht von Flask/API/Assets bedienten GET-Pfade sind clientseitige
# React-Router-Routen und liefern dieselbe index.html aus. Bewusst über einen
# 404-Handler (echter Catch-all) statt einer Routen-Whitelist: neue SPA-Seiten
# funktionieren damit automatisch, ohne hier etwas nachzutragen.
_RESERVED_PREFIXES = ("/api/", "/static/", "/ws/")


_RESERVED_EXACT = {"/health", "/logout"}


@bp.app_errorhandler(404)
def spa_fallback(error):
    path = request.path

    # Backend-/API-/Asset-Pfade niemals mit der SPA überschreiben.
    if (
        request.method != "GET"
        or path.startswith(_RESERVED_PREFIXES)
        or path in _RESERVED_EXACT
    ):
        return error.get_response()

    # Nur Navigation (HTML) oder extension-lose Pfade als SPA ausliefern;
    # fehlende Dateien mit Endung bleiben echte 404s.
    accepts_html = "text/html" in request.headers.get("Accept", "")
    looks_like_file = "." in os.path.basename(path)
    if accepts_html or not looks_like_file:
        return serve_spa()

    return error.get_response()
