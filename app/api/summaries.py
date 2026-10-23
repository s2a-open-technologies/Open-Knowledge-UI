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

import io
from datetime import UTC, datetime
from logging import getLogger
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from app.auth.auth import auth_required_api, current_user
from app.auth.users import (
    role_at_least,
)
from app.chat import summaries as summaries_store
from app.core.database_service import get_db_service

bp = Blueprint("summaries", __name__, url_prefix="/api")
logger = getLogger(__name__)


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


@bp.route("/summaries", methods=["GET"])
@auth_required_api
def summaries():
    user = current_user()
    if role_at_least(user["role"], "admin"):
        summaries_store.adopt_legacy(
            user["id"]
        )  # Altbestand einmalig dem Admin zuordnen
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
                    "created_at": datetime.fromtimestamp(
                        stat.st_ctime, tz=UTC
                    ).isoformat(),
                }
            )

    return jsonify(result)


def _safe_summary_path(filename: str) -> Path | None:
    """Datei im Analysen-Ordner des angemeldeten Nutzers (nie die anderer Nutzer)."""
    return summaries_store.safe_path(current_user()["id"], filename)


@bp.route("/summaries/<path:filename>", methods=["GET"])
@auth_required_api
def read_summary(filename):
    file_path = _safe_summary_path(filename)
    if file_path is None:
        return jsonify({"error": "Ungültige Datei"}), 400
    if not file_path.exists():
        return jsonify({"error": "Datei nicht gefunden"}), 404

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return jsonify({"error": f"Fehler beim Lesen: {e!s}"}), 500

    return jsonify({"filename": filename, "content": content})


@bp.route("/summaries/<path:filename>", methods=["PUT"])
@auth_required_api
def save_summary(filename):
    file_path = _safe_summary_path(filename)
    if file_path is None:
        return jsonify({"success": False, "error": "Ungültige Datei"}), 400

    data = request.get_json() or {}
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(data.get("content", ""), encoding="utf-8")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/summaries/<path:filename>", methods=["DELETE"])
@auth_required_api
def delete_summary(filename):
    file_path = _safe_summary_path(filename)
    if file_path is None:
        return jsonify({"success": False, "error": "Ungültige Datei"}), 400

    try:
        if file_path.exists():
            file_path.unlink()
        get_db_service().delete_summary_by_filename(
            filename, user_id=current_user()["id"]
        )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _summary_file_or_404(filename: str):
    file_path = _safe_summary_path(filename)
    if file_path is None:
        return None, (jsonify({"error": "Ungültige Datei"}), 400)
    if not file_path.is_file():
        return None, (jsonify({"error": "Datei nicht gefunden"}), 404)
    return file_path, None


@bp.route("/summaries/<path:filename>/download-md", methods=["GET"])
@auth_required_api
def download_summary_md(filename):
    file_path, error = _summary_file_or_404(filename)
    if error:
        return error
    return send_file(
        str(file_path),
        as_attachment=True,
        download_name=file_path.name,
        mimetype="text/markdown",
    )


@bp.route("/summaries/<path:filename>/download-docx", methods=["GET"])
@auth_required_api
def download_summary_docx(filename):
    file_path, error = _summary_file_or_404(filename)
    if error:
        return error
    try:
        data = summaries_store.markdown_to_docx(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"DOCX-Export von {filename} fehlgeschlagen: {e}")
        return jsonify({"error": f"Fehler beim Erzeugen der DOCX-Datei: {e!s}"}), 500
    return send_file(
        io.BytesIO(data),
        as_attachment=True,
        download_name=file_path.with_suffix(".docx").name,
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
