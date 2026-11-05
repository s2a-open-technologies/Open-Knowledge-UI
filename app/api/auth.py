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

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response

from app.api.errors import ApiError
from app.api.schemas.auth import (
    ChangePasswordRequest,
    LoginRequest,
    LoginResponse,
    Me,
    Providers,
    Version,
)
from app.api.schemas.common import Success
from app.auth.auth import CurrentUser, login_session
from app.auth.users import (
    LoginThrottledError,
    authenticate,
    set_password,
    verify_password,
)
from app.chat import data_export

router = APIRouter(prefix="/api", tags=["auth"])
logger = getLogger(__name__)


@router.get("/version", response_model=Version)
def app_version():
    """Version der Anwendung (öffentlich, z. B. für Support und Monitoring)."""
    from app.core.version import get_version

    return Version(version=get_version())


@router.get("/auth/me", response_model=Me)
def auth_me(user: CurrentUser):
    return {"authenticated": True, "user": user}


@router.post("/auth/login", response_model=LoginResponse)
def auth_login(request: Request, body: LoginRequest | None = None):
    body = body or LoginRequest()
    try:
        user = authenticate(
            body.username,
            body.password,
            client_ip=request.client.host if request.client else None,
        )
    except LoginThrottledError:
        raise ApiError(
            429,
            "Zu viele Fehlversuche. Bitte in einigen Minuten erneut versuchen.",
            success=False,
        )

    if user is None:
        raise ApiError(401, "Benutzername oder Passwort falsch", success=False)

    # Neue Session statt Wiederverwendung (verhindert Session-Fixation).
    login_session(request, user["id"])
    return {"success": True, "user": user}


@router.get("/auth/providers", response_model=Providers)
def auth_providers():
    """Welche externen Anmeldungen der Login-Bildschirm anbieten soll (ohne Anmeldung abrufbar)."""
    from app.auth import external as auth_external

    cfg = auth_external.load_config()
    return {
        "ldap": bool(cfg["ldap"]["enabled"]),
        "oidc": {
            "enabled": bool(cfg["oidc"]["enabled"]),
            "name": cfg["oidc"]["name"] or "SSO",
        },
    }


def _public_base_url(request: Request) -> str:
    from app.auth import external as auth_external

    return auth_external.load_config()["base_url"] or str(request.base_url).rstrip("/")


@router.get("/auth/oidc/login", include_in_schema=False)
def auth_oidc_login(request: Request):
    from app.auth import external as auth_external
    from app.auth import oidc as auth_oidc

    cfg = auth_external.load_config()["oidc"]
    if not cfg["enabled"]:
        return RedirectResponse("/?login_error=oidc_off", status_code=302)
    try:
        return RedirectResponse(
            auth_oidc.start_login(cfg, _public_base_url(request), request.session),
            status_code=302,
        )
    except auth_oidc.OidcError as e:
        logger.warning(f"OIDC-Start fehlgeschlagen: {e}")
        return RedirectResponse("/?login_error=oidc", status_code=302)


@router.get("/auth/oidc/callback", include_in_schema=False)
def auth_oidc_callback(request: Request):
    from app.auth import external as auth_external
    from app.auth import oidc as auth_oidc

    cfg = auth_external.load_config()["oidc"]
    if not cfg["enabled"]:
        return RedirectResponse("/?login_error=oidc_off", status_code=302)
    try:
        found = auth_oidc.finish_login(
            cfg, _public_base_url(request), request.session, dict(request.query_params)
        )
        user = auth_external.login_external_user(
            "oidc",
            found["external_id"],
            found["username"],
            found["display_name"],
            found["groups"],
        )
    except (auth_oidc.OidcError, auth_external.ExternalLoginError) as e:
        logger.warning(f"OIDC-Anmeldung abgelehnt: {e}")
        request.session.pop("oidc", None)
        return RedirectResponse("/?login_error=oidc", status_code=302)
    login_session(request, user["id"])  # neue Session (verhindert Session-Fixation)
    return RedirectResponse("/", status_code=302)


@router.post("/auth/logout", response_model=Success)
def auth_logout(request: Request):
    from app.auth import audit

    audit.record(audit.LOGOUT)
    request.session.clear()
    return Success()


@router.post("/settings/change-password", response_model=Success)
def change_password(user: CurrentUser, body: ChangePasswordRequest | None = None):
    body = body or ChangePasswordRequest()
    if not body.new_password or len(body.new_password) < 4:
        raise ApiError(400, "Passwort muss mindestens 4 Zeichen haben", success=False)

    if not verify_password(user["id"], body.old_password):
        raise ApiError(403, "Altes Passwort ist falsch", success=False)

    set_password(user["id"], body.new_password)
    from app.auth import audit

    audit.record(
        audit.PASSWORD_CHANGE,
        actor_user_id=user["id"],
        object_type="user",
        object_id=user["id"],
    )
    return Success()


@router.get("/me/export")
def export_my_data(user: CurrentUser):
    """Alle persoenlichen Daten des angemeldeten Kontos als JSON (Auskunft/Übertragbarkeit)."""
    try:
        data = data_export.build_export(user["id"])
    except LookupError:
        raise ApiError(404, "Konto nicht gefunden", success=False)
    response = Response(
        data_export.to_response_bytes(data),
        media_type="application/json",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{data_export.export_filename(user["username"])}"'
            )
        },
    )
    from app.auth import audit

    audit.record(
        audit.DATA_EXPORT,
        actor_user_id=user["id"],
        object_type="user",
        object_id=user["id"],
    )
    return response
