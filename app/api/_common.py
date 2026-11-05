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

"""Gemeinsame Helfer der Routen-Module."""

from typing import Annotated

from fastapi import Depends, Request
from starlette.concurrency import run_in_threadpool

from app.api.errors import ApiError
from app.auth.auth import CurrentUser
from app.ingest import reindex


def abort_if_reindexing(knowledge_id: int) -> None:
    """409, solange das Knowledge neu indexiert wird (Schreibzugriffe warten)."""
    if reindex.is_locked(knowledge_id):
        raise ApiError(409, "Das Knowledge wird gerade neu indexiert", success=False)


class Upload:
    """``UploadFile`` mit der Schnittstelle, die die Dienste erwarten (``filename``, ``save(pfad)``)."""

    def __init__(self, upload):
        self._upload = upload
        self.filename = upload.filename or ""

    def save(self, path) -> None:
        import shutil

        self._upload.file.seek(0)
        with open(path, "wb") as target:
            shutil.copyfileobj(self._upload.file, target)


def _as_int(raw) -> int | None:
    return int(raw) if str(raw or "").isdigit() else None


async def _requested_knowledge_id(request) -> int | None:
    """``knowledge_id`` aus Query, Formular oder JSON-Body (in dieser Reihenfolge)."""
    kid = _as_int(request.query_params.get("knowledge_id"))
    content_type = request.headers.get("content-type", "")
    if kid is None and (
        content_type.startswith("multipart/form-data")
        or content_type.startswith("application/x-www-form-urlencoded")
    ):
        kid = _as_int((await request.form()).get("knowledge_id"))
    if kid is None and content_type.startswith("application/json"):
        try:
            body = await request.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            kid = _as_int(body.get("knowledge_id"))
    return kid


def _check_knowledge(kid: int | None, user: dict, method: str, path: str) -> dict:
    from app.auth.access import can_access_knowledge
    from app.knowledge.service import get_knowledge, is_remote

    knowledge = get_knowledge(kid)
    # Lesende Aufrufe brauchen Lese-, aendernde Schreibrechte auf dieses Knowledge.
    level = "read" if method in ("GET", "HEAD") else "write"
    if knowledge is None or not can_access_knowledge(user, knowledge["id"], level):
        raise ApiError(404, "Knowledge nicht gefunden", success=False)
    if is_remote(knowledge) and (level == "write" or path.startswith("/api/vectordb")):
        raise ApiError(409, "Diese Quelle ist schreibgeschützt", success=False)
    if level == "write":
        abort_if_reindexing(knowledge["id"])
    return knowledge


async def scoped_knowledge(request: Request, user: CurrentUser) -> dict:
    """Dependency: Knowledge-Bereich des Requests, ohne Angabe der Standard-Bereich. Prueft den
    Zugriff vor dem Handler (lesen fuer GET, sonst schreiben; fremde Quellen sind schreibgeschuetzt)."""
    kid = await _requested_knowledge_id(request)
    return await run_in_threadpool(
        _check_knowledge, kid, user, request.method, request.url.path
    )


ScopedKnowledge = Annotated[dict, Depends(scoped_knowledge)]
