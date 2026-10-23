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
"Meine Uploads": die eigenen Chat-Anhaenge eines Nutzers, jeweils als Markdown. Loeschen entfernt die
Datei und alle zugehoerigen Eintraege in Qdrant. Dokumente in einem Knowledge gehoeren zum Knowledge und
stehen dort (Dokumentliste); wer sie hochgeladen hat, darf sie dort auch ohne Schreibrecht loeschen.
"""

from logging import getLogger
from pathlib import Path

from qdrant_client import models as qdrant_models

from app.chat import attachments
from app.core.database_service import get_db_service
from app.core.models import ChatAttachment, ChatSession, UserDocument
from app.ingest.tables import delete_tables
from app.knowledge.files import delete_associated_file
from app.knowledge.index_doc import refresh_index
from app.knowledge.service import knowledge_paths

logger = getLogger(__name__)

MARKDOWN_ROOT = Path("./data/markdown")
ATTACHMENTS_ROOT = Path("./data/attachments")


class UploadError(Exception):
    """Fachlicher Fehler mit HTTP-Status fuer die API."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def safe_markdown(path_str: str | None, root: Path) -> Path | None:
    """Markdown-Datei, wenn sie existiert und wirklich unter ``root`` liegt."""
    if not path_str:
        return None
    path = Path(path_str).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None  # nie Dateien ausserhalb der Ablageordner ausliefern
    return path if path.is_file() else None


def markdown_path(user_doc) -> Path | None:
    return safe_markdown(user_doc.markdown_path, MARKDOWN_ROOT)


def delete_knowledge_document(user_doc, knowledge: dict, client) -> None:
    """Loescht ein Dokument eines Knowledge samt Vektoren, Markdown, Original und Tabellen-CSVs.

    Raises:
        UploadError: wenn Qdrant das Loeschen verweigert (dann bleibt alles andere unberuehrt).
    """
    collection = knowledge["collection_name"]
    doc_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="file_group_id",
                match=qdrant_models.MatchValue(value=user_doc.document_id),
            )
        ]
    )
    payload = None
    if client:
        try:
            points, _ = client.scroll(
                collection_name=collection,
                scroll_filter=doc_filter,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if points:
                payload = points[0].payload
        except Exception as e:
            logger.warning(f"Konnte Payload nicht laden: {e}")
        try:
            client.delete(
                collection_name=collection,
                points_selector=qdrant_models.FilterSelector(filter=doc_filter),
            )
        except Exception as e:
            raise UploadError(f"Qdrant Fehler: {e!s}", 500) from e

    if payload:
        delete_associated_file(payload)
    md = markdown_path(user_doc)
    if md is not None:
        md.unlink(missing_ok=True)
    # Aus dem Dokument extrahierte Tabellen (CSV + Verweis-Chunk) gehoeren mit weg.
    delete_tables(knowledge_paths(knowledge), collection, user_doc.document_id)

    db = get_db_service().get_session()
    try:
        db.query(UserDocument).filter_by(id=user_doc.id).delete()
        db.commit()
    finally:
        db.close()
    refresh_index(knowledge["id"])


# --- Liste und Zugriff -------------------------------------------------------


def list_uploads(user_id: int) -> list[dict]:
    """Eigene Chat-Anhaenge, neueste zuerst. Nur Eintraege mit Eigentuemer = Nutzer."""
    db = get_db_service().get_session()
    try:
        items = []
        for att in (
            db.query(ChatAttachment).filter_by(user_id=user_id, status="ready").all()
        ):
            session = db.query(ChatSession).filter_by(id=att.session_id).first()
            md = safe_markdown(att.markdown_path, ATTACHMENTS_ROOT)
            items.append(
                {
                    "kind": "chat",
                    "id": att.id,
                    "filename": _as_md(att.filename),
                    "source": (session.title or "Neuer Chat") if session else "Chat",
                    "created_at": att.created_at.isoformat()
                    if att.created_at
                    else None,
                    "size": md.stat().st_size if md else None,
                    "has_markdown": md is not None,
                }
            )
        items.sort(key=lambda i: i["created_at"] or "", reverse=True)
        return items
    finally:
        db.close()


def _as_md(filename: str) -> str:
    return f"{Path(filename).stem}.md"


def get_owned(kind: str, item_id: str, user_id: int):
    """Eigener Eintrag oder ``UploadError(404)`` (fremde sind nicht auffindbar)."""
    db = get_db_service().get_session()
    try:
        if kind == "chat":
            row = (
                db.query(ChatAttachment).filter_by(id=item_id, user_id=user_id).first()
            )
        else:
            row = None
        if row is None:
            raise UploadError("Datei nicht gefunden", 404)
        db.expunge(row)
        return row
    finally:
        db.close()


def read_markdown(kind: str, row) -> str | None:
    path = (
        safe_markdown(row.markdown_path, ATTACHMENTS_ROOT) if kind == "chat" else None
    )
    return path.read_text(encoding="utf-8") if path else None


def delete_upload(kind: str, row, client) -> None:
    if kind != "chat":
        raise UploadError("Datei nicht gefunden", 404)
    attachments.delete_attachment(row, client)
