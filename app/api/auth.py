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

"""Anmeldung (lokal, OIDC), eigenes Konto (Passwort, Datenexport) und Version."""

from logging import getLogger

from flask import (
    Blueprint,
    Response,
    jsonify,
    redirect,
    request,
    session,
)

from app.auth.auth import auth_required_api, current_user
from app.auth.users import (
    LoginThrottledError,
    authenticate,
    set_password,
    verify_password,
)
from app.chat import data_export

bp = Blueprint("auth_api", __name__, url_prefix="/api")
logger = getLogger(__name__)


@bp.route("/version", methods=["GET"])
def app_version():
    """Version der Anwendung (öffentlich, z. B. für Support und Monitoring)."""
    from app.core.version import get_version

    return jsonify({"version": get_version()})


@bp.route("/auth/me", methods=["GET"])
@auth_required_api
def auth_me():
    return jsonify({"authenticated": True, "user": current_user()})


@bp.route("/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json() or {}
    try:
        user = authenticate(
            data.get("username", ""),
            data.get("password", ""),
            client_ip=request.remote_addr,
        )
    except LoginThrottledError:
        return jsonify(
            {
                "success": False,
                "error": "Zu viele Fehlversuche. Bitte in einigen Minuten erneut versuchen.",
            }
        ), 429

    if user is None:
        return jsonify(
            {"success": False, "error": "Benutzername oder Passwort falsch"}
        ), 401

    # Neue Session statt Wiederverwendung (verhindert Session-Fixation).
    session.clear()
    session["user_id"] = user["id"]
    return jsonify({"success": True, "user": user})


@bp.route("/auth/providers", methods=["GET"])
def auth_providers():
    """Welche externen Anmeldungen der Login-Bildschirm anbieten soll (ohne Anmeldung abrufbar)."""
    from app.auth import external as auth_external

    cfg = auth_external.load_config()
    return jsonify(
        {
            "ldap": bool(cfg["ldap"]["enabled"]),
            "oidc": {
                "enabled": bool(cfg["oidc"]["enabled"]),
                "name": cfg["oidc"]["name"] or "SSO",
            },
        }
    )


def _public_base_url() -> str:
    from app.auth import external as auth_external

    return auth_external.load_config()["base_url"] or request.host_url.rstrip("/")


@bp.route("/auth/oidc/login", methods=["GET"])
def auth_oidc_login():
    from app.auth import external as auth_external
    from app.auth import oidc as auth_oidc

    cfg = auth_external.load_config()["oidc"]
    if not cfg["enabled"]:
        return redirect("/?login_error=oidc_off")
    try:
        return redirect(auth_oidc.start_login(cfg, _public_base_url(), session))
    except auth_oidc.OidcError as e:
        logger.warning(f"OIDC-Start fehlgeschlagen: {e}")
        return redirect("/?login_error=oidc")


@bp.route("/auth/oidc/callback", methods=["GET"])
def auth_oidc_callback():
    from app.auth import external as auth_external
    from app.auth import oidc as auth_oidc

    cfg = auth_external.load_config()["oidc"]
    if not cfg["enabled"]:
        return redirect("/?login_error=oidc_off")
    try:
        found = auth_oidc.finish_login(cfg, _public_base_url(), session, request.args)
        user = auth_external.login_external_user(
            "oidc",
            found["external_id"],
            found["username"],
            found["display_name"],
            found["groups"],
        )
    except (auth_oidc.OidcError, auth_external.ExternalLoginError) as e:
        logger.warning(f"OIDC-Anmeldung abgelehnt: {e}")
        session.pop("oidc", None)
        return redirect("/?login_error=oidc")
    session.clear()  # neue Session (verhindert Session-Fixation)
    session["user_id"] = user["id"]
    return redirect("/")


@bp.route("/auth/logout", methods=["POST"])
def auth_logout():
    from app.auth import audit

    audit.record(audit.LOGOUT)
    session.clear()
    return jsonify({"success": True})


@bp.route("/settings/change-password", methods=["POST"])
@auth_required_api
def change_password():
    data = request.get_json() or {}
    old_password = data.get("old_password", "")
    new_password = data.get("new_password", "")

    if not new_password or len(new_password) < 4:
        return jsonify(
            {"success": False, "error": "Passwort muss mindestens 4 Zeichen haben"}
        ), 400

    user = current_user()
    if not verify_password(user["id"], old_password):
        return jsonify({"success": False, "error": "Altes Passwort ist falsch"}), 403

    set_password(user["id"], new_password)
    from app.auth import audit

    audit.record(
        audit.PASSWORD_CHANGE,
        actor_user_id=user["id"],
        object_type="user",
        object_id=user["id"],
    )
    return jsonify({"success": True})


@bp.route("/me/export", methods=["GET"])
@auth_required_api
def export_my_data():
    """Alle persoenlichen Daten des angemeldeten Kontos als JSON (Auskunft/Übertragbarkeit)."""
    user = current_user()
    try:
        data = data_export.build_export(user["id"])
    except LookupError:
        return jsonify({"success": False, "error": "Konto nicht gefunden"}), 404
    response = Response(
        data_export.to_response_bytes(data),
        content_type="application/json",
    )
    response.headers["Content-Disposition"] = (
        f'attachment; filename="{data_export.export_filename(user["username"])}"'
    )
    from app.auth import audit

    audit.record(
        audit.DATA_EXPORT,
        actor_user_id=user["id"],
        object_type="user",
        object_id=user["id"],
    )
    return response
