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
Chat-Anhaenge: Dateien, die direkt im Chat hochgeladen werden.

Sie landen mit fester Ingestion (kein LLM-Kontext, keine Archivierung) in der
Collection ``user_uploads`` und tragen die ``session_id`` im Payload. Die Suche
filtert darauf (siehe ``app/retrieval.py``), sodass ein Anhang nur in seinem
Chat auffindbar ist.
"""

import base64
import shutil
import threading
from logging import getLogger
from pathlib import Path

from qdrant_client import models as qdrant_models
from werkzeug.utils import secure_filename

from app.core.database_service import get_db_service
from app.knowledge.files import DOCUMENT_EXTS, IMAGE_EXTS
from app.rag.collections import USER_UPLOADS_COLLECTION

logger = getLogger(__name__)

ATTACHMENTS_ROOT = Path("./data/attachments")

# Docling/Embedding sind speicherhungrig: Anhaenge werden nacheinander verarbeitet.
_ingest_lock = threading.Lock()


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in DOCUMENT_EXTS | IMAGE_EXTS


def is_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTS


IMAGE_COPY = "bild.jpg"  # verkleinerte Fassung eines Bild-Anhangs (geht direkt an Modelle, die Bilder verstehen)


def image_copy_path(session_id: str, attachment_id: str) -> Path:
    return session_dir(session_id) / attachment_id / IMAGE_COPY


def image_data_url(session_id: str, attachment_id: str) -> str | None:
    """Das Bild als ``data:``-URL oder ``None``, wenn der Anhang kein (lesbares) Bild ist."""
    path = image_copy_path(session_id, attachment_id)
    try:
        return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode(
            "ascii"
        )
    except OSError:
        return None


def session_dir(session_id: str) -> Path:
    return ATTACHMENTS_ROOT / session_id


def start_attachment_ingestion(
    file_storage, session_id: str, sparse, user_id: int | None = None
) -> dict:
    """Speichert die Datei, legt den DB-Eintrag an und startet die Ingestion
    im Hintergrund. Liefert den Anhang (Status ``processing``)."""
    filename = secure_filename(file_storage.filename or "") or "datei"
    db = get_db_service()
    attachment = db.create_attachment(session_id, filename, user_id)

    folder = session_dir(session_id) / attachment.id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    file_storage.save(path)
    if is_image(filename):
        # Schon hier, damit die Frage nicht auf das Lesen des Bildes warten muss
        try:
            from app.ingest.image_text import encode_image

            (folder / IMAGE_COPY).write_bytes(base64.b64decode(encode_image(path)))
        except Exception as e:  # die Ingestion meldet das unlesbare Bild dem Nutzer
            logger.warning(f"Bildkopie fuer {filename} nicht angelegt: {e}")

    threading.Thread(
        target=_ingest,
        args=(attachment.id, session_id, str(path), folder, sparse, user_id),
        name=f"attachment-{attachment.id[:8]}",
        daemon=True,
    ).start()

    return {"id": attachment.id, "filename": filename, "status": "processing"}


def _ingest(attachment_id, session_id, path, folder, sparse, user_id=None) -> None:
    from app import wait_for_resources
    from app.ingest.pipeline import DocumentIngestion
    from app.rag import model_settings, models_runtime

    db = get_db_service()
    try:
        with _ingest_lock:
            wait_for_resources(timeout=180)
            # Anhaenge nutzen immer das aktuelle globale Modell (Collection je Modell)
            spec = model_settings.embedding_spec()
            with DocumentIngestion(
                embedding_spec=spec,
                sparse_embeddings=sparse,
                collection_name=models_runtime.uploads_collection(spec),
                extra_metadata={
                    "session_id": session_id,
                    "attachment_id": attachment_id,
                    **({"uploaded_by": user_id} if user_id is not None else {}),
                },
                enrich_context=False,
                track_documents=False,
                files_dir=str(folder),
                attachment_markdown_dir=str(folder / "md"),
            ) as ingest:
                ok = ingest._process_file(path, sparse)
                markdown_path = ingest.last_markdown_path
                reason = next((e["error"] for e in ingest.errors), None)
        if ok:
            if markdown_path:
                db.set_attachment_markdown(attachment_id, markdown_path)
                Path(path).unlink(missing_ok=True)  # nur das Markdown bleibt
            db.set_attachment_status(attachment_id, "ready")
        else:
            db.set_attachment_status(
                attachment_id, "error", reason or "Datei enthält keinen lesbaren Inhalt"
            )
    except Exception as e:
        logger.exception(f"Anhang-Ingestion fehlgeschlagen: {attachment_id}")
        db.set_attachment_status(attachment_id, "error", str(e)[:300])


def cleanup_session_attachments(session_id: str, client) -> None:
    """Entfernt Vektoren, Dateien und DB-Eintraege der Anhaenge einer Session."""
    if client is not None:
        try:
            names = [c.name for c in client.get_collections().collections]
            for name in names:
                if not name.startswith(USER_UPLOADS_COLLECTION):
                    continue  # "user_uploads" und "user_uploads_<hash>" je Modell
                client.delete(
                    collection_name=name,
                    points_selector=qdrant_models.FilterSelector(
                        filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="session_id",
                                    match=qdrant_models.MatchValue(value=session_id),
                                )
                            ]
                        )
                    ),
                )
        except Exception as e:
            logger.warning(f"Anhang-Vektoren konnten nicht gelöscht werden: {e}")
    shutil.rmtree(session_dir(session_id), ignore_errors=True)
    try:
        get_db_service().delete_session_attachments(session_id)
    except Exception as e:
        logger.warning(f"Anhang-Einträge konnten nicht gelöscht werden: {e}")


def delete_attachment(attachment, client) -> None:
    """Entfernt einen einzelnen Anhang: Vektoren (alle Anhang-Collections), Dateien, DB-Eintrag."""
    if client is not None:
        try:
            for name in [c.name for c in client.get_collections().collections]:
                if not name.startswith(USER_UPLOADS_COLLECTION):
                    continue
                client.delete(
                    collection_name=name,
                    points_selector=qdrant_models.FilterSelector(
                        filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="attachment_id",
                                    match=qdrant_models.MatchValue(value=attachment.id),
                                )
                            ]
                        )
                    ),
                )
        except Exception as e:
            logger.warning(f"Anhang-Vektoren konnten nicht gelöscht werden: {e}")
    shutil.rmtree(
        session_dir(attachment.session_id) / attachment.id, ignore_errors=True
    )
    get_db_service().delete_attachment_row(attachment.id)
