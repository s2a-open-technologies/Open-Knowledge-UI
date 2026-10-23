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
Datenexport fuer Auskunft und Datenuebertragbarkeit (DSGVO Art. 15/20).

``build_export`` sammelt alles, was einem Konto persoenlich zugeordnet ist, in ein
maschinenlesbares JSON: Profil, eigene Chats mit Meldungen, gespeicherte Analysen
("Meine Ablagen", vollstaendige Markdown-Inhalte), eigene Dokument- und
Anhang-Referenzen (Metadaten incl. Markdown-Text). Es enthaelt keine Inhalte, die
anderen Konten gehoeren: eigene Chats, eigener Ablagen-Ordner, eigene Uploads.

Immer nur mit dicts/einfachen Werten zurueckgeben — die DB-Sitzung schliesst vor
der Rueckgabe (DetachedInstanceError-Gefahr, siehe AGENTS.md).
"""

import json
from datetime import datetime
from logging import getLogger

from app.chat.summaries import safe_path as _summaries_path
from app.core.database_service import get_db_service
from app.core.models import (
    ChatAttachment,
    ChatSession,
    User,
    UserDocument,
    UserSummaryFile,
)

log = getLogger(__name__)

EXPORT_FORMAT = "open-knowledge-ui-export"
EXPORT_VERSION = 1

# Spalten ohne Passwort-Hash oder Fremdschluessel: nur das, was den Nutzer betrifft.
USER_FIELDS = ("id", "username", "display_name", "role", "auth_source", "created_at")


def _iso(dt):
    return dt.isoformat(timespec="seconds") if dt else None


def _read_markdown(path) -> str | None:
    """Markdown-Inhalt lesen; fehlende Datei ist kein Fehler (Altbestand, aufgeräumt)."""
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except (OSError, ValueError):
        log.info("Datei im Export uebergangen: %r", path)
        return None


def build_export(user_id: int) -> dict:
    """Alle persoenlichen Daten des Kontos ``user_id`` als exportfaehigen dict."""
    data = {
        "format": EXPORT_FORMAT,
        "version": EXPORT_VERSION,
        "exported_at": datetime.utcnow().isoformat(timespec="seconds"),
        "profil": {},
        "chats": [],
        "meine_analysen": [],
        "meine_dokumente": [],
        "chat_anhaenge": [],
    }

    db_service = get_db_service()
    session = db_service.get_session()
    try:
        user = session.query(User).filter_by(id=int(user_id)).first()
        if user is None:
            raise LookupError(f"Nutzer {user_id} nicht gefunden")
        data["profil"] = {field: getattr(user, field) for field in USER_FIELDS}

        sessions = (
            session.query(ChatSession)
            .filter_by(user_id=user.id)
            .order_by(ChatSession.created_at)
            .all()
        )
        for chat in sessions:
            data["chats"].append(
                {
                    "id": chat.id,
                    "titel": chat.title,
                    "erstellt": _iso(chat.created_at),
                    "zusammenfassung": chat.summary,
                    "nachrichten": [
                        {
                            "id": message.id,
                            "rolle": message.role,
                            "inhalt": message.content,
                            "erstellt": _iso(message.created_at),
                        }
                        for message in sorted(
                            chat.messages,
                            key=lambda m: (m.created_at is None, m.created_at, m.id),
                        )
                    ],
                }
            )

        for summary in session.query(UserSummaryFile).filter_by(user_id=user.id).all():
            data["meine_analysen"].append(
                {
                    "dateiname": summary.filename,
                    "erstellt": _iso(summary.created_at),
                    "inhalt": _read_markdown(
                        _summaries_path(user.id, summary.filename)
                    ),
                }
            )

        for document in (
            session.query(UserDocument).filter_by(uploaded_by=user.id).all()
        ):
            data["meine_dokumente"].append(
                {
                    "id": document.document_id,
                    "dateiname": document.filename,
                    "knowledge_id": document.knowledge_id,
                    "hochgeladen": _iso(document.uploaded_at),
                    "markdown": _read_markdown(document.markdown_path),
                }
            )

        for attachment in (
            session.query(ChatAttachment).filter_by(user_id=user.id).all()
        ):
            data["chat_anhaenge"].append(
                {
                    "id": attachment.id,
                    "chat": attachment.session_id,
                    "dateiname": attachment.filename,
                    "status": attachment.status,
                    "hochgeladen": _iso(attachment.created_at),
                    "markdown": _read_markdown(attachment.markdown_path),
                }
            )
    finally:
        session.close()
    return data


def export_filename(username: str) -> str:
    date = datetime.utcnow().strftime("%Y-%m-%d")
    return f"open-knowledge-ui-export-{username}-{date}.json"


def to_response_bytes(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, indent=1, default=str).encode("utf-8")
