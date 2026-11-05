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

"""Fehlerantworten der API: immer JSON mit ``error`` (das Frontend liest genau dieses Feld).

* :class:`ApiError` aus Handlern und Dependencies: ``raise ApiError(404, "Knowledge nicht gefunden",
  success=False)`` ergibt ``{"error": ..., "success": false}`` mit Status 404.
* ``HTTPException`` (z. B. 404/405 von Starlette) -> ``{"error": detail}``.
* Validierungsfehler der Pydantic-Modelle -> 422 mit lesbarer Meldung und den Feldern.
* ``AdminError``/``ProfileError`` der Dienste -> ``{"success": false, "error": ...}`` mit ihrem Status.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ApiError(Exception):
    def __init__(self, status: int, error: str, **extra):
        super().__init__(error)
        self.status = status
        self.payload = {**extra, "error": error}


def _field(loc) -> str:
    parts = [str(p) for p in loc if p not in ("body", "query", "path")]
    return ".".join(parts) or "Anfrage"


# Pydantic-Fehlertypen auf Deutsch (Rest: Originalmeldung)
_MESSAGES = {
    "missing": "fehlt",
    "string_type": "muss Text sein",
    "int_type": "muss eine ganze Zahl sein",
    "int_parsing": "muss eine ganze Zahl sein",
    "int_from_float": "muss eine ganze Zahl sein",
    "float_type": "muss eine Zahl sein",
    "float_parsing": "muss eine Zahl sein",
    "bool_type": "muss wahr oder falsch sein",
    "bool_parsing": "muss wahr oder falsch sein",
    "list_type": "muss eine Liste sein",
    "dict_type": "muss ein Objekt sein",
    "model_type": "muss ein Objekt sein",
    "model_attributes_type": "muss ein Objekt sein",
    "json_invalid": "ist kein gültiges JSON",
    "literal_error": "hat einen unzulässigen Wert",
    "enum": "hat einen unzulässigen Wert",
    "string_too_short": "ist zu kurz",
    "string_too_long": "ist zu lang",
    "greater_than": "ist zu klein",
    "greater_than_equal": "ist zu klein",
    "less_than": "ist zu groß",
    "less_than_equal": "ist zu groß",
    "extra_forbidden": "ist nicht erlaubt",
}

_STATUS = {
    400: "Ungültige Anfrage",
    401: "Nicht authentifiziert",
    403: "Keine Berechtigung",
    404: "Nicht gefunden",
    405: "Methode nicht erlaubt",
    413: "Anfrage zu groß",
    415: "Nicht unterstütztes Format",
}


def _message(error: dict) -> str:
    return _MESSAGES.get(error.get("type", ""), error.get("msg", "ungültig"))


def validation_message(errors) -> str:
    first = errors[0] if errors else {}
    return f"Ungültige Eingabe: {_field(first.get('loc', ()))} – {_message(first)}"


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(_request: Request, exc: ApiError):
        return JSONResponse(exc.payload, status_code=exc.status)

    # Fachliche Fehler der Dienste (Meldung + Status)
    from app.auth.user_admin import AdminError
    from app.rag.profiles import ProfileError

    for domain_error in (AdminError, ProfileError):

        @app.exception_handler(domain_error)
        async def domain(_request: Request, exc):
            return JSONResponse({"success": False, "error": exc.message}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        errors = exc.errors()
        return JSONResponse(
            {
                "success": False,
                "error": validation_message(errors),
                "fields": [
                    {"field": _field(e.get("loc", ())), "message": _message(e)}
                    for e in errors
                ],
            },
            status_code=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        from app.api.pages import spa_fallback

        if exc.status_code == 404:
            fallback = spa_fallback(request)
            if fallback is not None:
                return fallback
        detail = _STATUS.get(exc.status_code) or (
            exc.detail if isinstance(exc.detail, str) else "Fehler"
        )
        return JSONResponse(
            {"error": detail},
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )
