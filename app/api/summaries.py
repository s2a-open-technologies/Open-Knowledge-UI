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

"""Analysen (Summaries) des Nutzers: auflisten, lesen, speichern, loeschen, herunterladen."""

from datetime import UTC, datetime
from logging import getLogger
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, Response

from app.api.errors import ApiError
from app.api.schemas.common import Success
from app.api.schemas.summaries import SummaryContent, SummaryFile, SummarySave
from app.auth.auth import CurrentUser
from app.auth.users import role_at_least
from app.chat import summaries as summaries_store
from app.core.database_service import get_db_service

router = APIRouter(prefix="/api", tags=["summaries"])
logger = getLogger(__name__)


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


@router.get("/summaries", response_model=list[SummaryFile])
def summaries(user: CurrentUser):
    if role_at_least(user["role"], "admin"):
        summaries_store.adopt_legacy(user["id"])  # Altbestand einmalig dem Admin zuordnen
    summaries_dir = summaries_store.user_dir(user["id"])
    result = []

    if summaries_dir.exists():
        for file_path in sorted(summaries_dir.glob("*.md")):
            stat = file_path.stat()
            result.append(
                {
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "size_human": _format_file_size(stat.st_size),
                    "modified": stat.st_mtime,
                    "created_at": datetime.fromtimestamp(stat.st_ctime, tz=UTC).isoformat(),
                }
            )

    return result


def _safe_summary_path(user: dict, filename: str) -> Path | None:
    """Datei im Analysen-Ordner des angemeldeten Nutzers (nie die anderer Nutzer)."""
    return summaries_store.safe_path(user["id"], filename)


def _existing_summary(user: dict, filename: str) -> Path:
    file_path = _safe_summary_path(user, filename)
    if file_path is None:
        raise ApiError(400, "Ungültige Datei")
    if not file_path.is_file():
        raise ApiError(404, "Datei nicht gefunden")
    return file_path


# Downloads vor den allgemeinen Routen: ``{filename:path}`` wuerde sonst ".../download-md" schlucken.
@router.get("/summaries/{filename:path}/download-md", response_class=FileResponse)
def download_summary_md(filename: str, user: CurrentUser):
    file_path = _existing_summary(user, filename)
    return FileResponse(str(file_path), filename=file_path.name, media_type="text/markdown")


@router.get("/summaries/{filename:path}/download-docx")
def download_summary_docx(filename: str, user: CurrentUser):
    file_path = _existing_summary(user, filename)
    try:
        data = summaries_store.markdown_to_docx(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"DOCX-Export von {filename} fehlgeschlagen: {e}")
        raise ApiError(500, f"Fehler beim Erzeugen der DOCX-Datei: {e!s}")
    name = file_path.with_suffix(".docx").name
    from urllib.parse import quote

    return Response(
        data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"},
    )


@router.get("/summaries/{filename:path}", response_model=SummaryContent)
def read_summary(filename: str, user: CurrentUser):
    file_path = _safe_summary_path(user, filename)
    if file_path is None:
        raise ApiError(400, "Ungültige Datei")
    if not file_path.exists():
        raise ApiError(404, "Datei nicht gefunden")

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise ApiError(500, f"Fehler beim Lesen: {e!s}")

    return {"filename": filename, "content": content}


@router.put("/summaries/{filename:path}", response_model=Success)
def save_summary(filename: str, user: CurrentUser, body: SummarySave | None = None):
    file_path = _safe_summary_path(user, filename)
    if file_path is None:
        raise ApiError(400, "Ungültige Datei", success=False)

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text((body or SummarySave()).content, encoding="utf-8")
    except Exception as e:
        raise ApiError(500, str(e), success=False)
    return Success()


@router.delete("/summaries/{filename:path}", response_model=Success)
def delete_summary(filename: str, user: CurrentUser):
    file_path = _safe_summary_path(user, filename)
    if file_path is None:
        raise ApiError(400, "Ungültige Datei", success=False)

    try:
        if file_path.exists():
            file_path.unlink()
        get_db_service().delete_summary_by_filename(filename, user_id=user["id"])
    except Exception as e:
        raise ApiError(500, str(e), success=False)
    return Success()
