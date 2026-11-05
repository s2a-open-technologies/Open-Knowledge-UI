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

"""Dokumente eines Knowledge: Upload, Einlesen, Markdown, Datendateien, Vektordatenbank, eigene Uploads."""

import os
from logging import getLogger
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse, Response
from qdrant_client import models

from app import resources_ready, wait_for_resources
from app.api._common import ScopedKnowledge, Upload
from app.api.errors import ApiError
from app.api.schemas.common import Success
from app.api.schemas.documents import (
    BatchDelete,
    ClearAll,
    Cleared,
    DataFile,
    DataPreview,
    DocumentRow,
    IngestionStatus,
    IngestResult,
    KnowledgeBody,
    MarkdownView,
    MyUpload,
    UploadResult,
    VectorDetail,
    VectorList,
)
from app.auth.access import can_access_knowledge
from app.auth.auth import CurrentUser
from app.chat import my_uploads
from app.core import runtime
from app.core.database_service import get_db_service
from app.ingest import reindex
from app.knowledge.files import check_data_file_exists, delete_associated_file
from app.knowledge.index_doc import refresh_index
from app.knowledge.service import get_knowledge, knowledge_paths
from app.rag import model_settings

router = APIRouter(prefix="/api", tags=["documents"])
logger = getLogger(__name__)

_NO_CLIENT = "Qdrant Client nicht initialisiert"


@router.get("/ingestion-status", response_model=IngestionStatus)
def ingestion_status(_user: CurrentUser):
    """Geteilter Ingestion-Status fuer den Navbar-Indikator per Polling."""
    return runtime.state.ingestion_status


def _index_data_references_best_effort(knowledge: dict) -> None:
    """Indexiert Verweis-Chunks für alle strukturierten Dateien. Fehler werden
    geloggt, aber nie an den Upload-Request weitergereicht."""
    try:
        sparse = runtime.state.sparse_embeddings
        if not resources_ready() or not sparse:
            return
        dense = model_settings.embedder_for(knowledge)  # Modell dieses Knowledge
        from app.ingest.pipeline import index_data_references
        from app.rag import sparse as sparse_lang

        paths = knowledge_paths(knowledge)
        index_data_references(
            dense,
            sparse_lang.resolve(sparse, sparse_lang.knowledge_language(knowledge)),
            knowledge["collection_name"],
            str(paths.data_dir),
            name_prefix=paths.name_prefix,
        )
    except Exception as e:
        logger.warning(f"Verweis-Indexierung nach Upload fehlgeschlagen: {e}")


@router.get("/documents", response_model=list[DocumentRow])
def documents(knowledge: ScopedKnowledge):
    db = get_db_service()
    user_docs = db.get_all_documents(knowledge_id=knowledge["id"])

    client = runtime.state.client
    result = []

    for user_doc in user_docs:
        entry = {
            "id": user_doc.id,
            "document_id": user_doc.document_id,
            "filename": user_doc.filename,
            "uploaded_at": user_doc.uploaded_at.isoformat() if user_doc.uploaded_at else None,
            "content_preview": "[Nicht mehr in Qdrant]",
            "source": "N/A",
            "has_data_file": False,
            "has_markdown": my_uploads.markdown_path(user_doc) is not None,
        }

        if client:
            try:
                points, _ = client.scroll(
                    collection_name=knowledge["collection_name"],
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_group_id",
                                match=models.MatchValue(value=user_doc.document_id),
                            )
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
                if points:
                    payload = points[0].payload or {}
                    entry["content_preview"] = (
                        payload.get("page_content", "") or payload.get("original_content", "")
                    )[:100] + "..."
                    entry["source"] = payload.get("source", "N/A")
                    entry["has_data_file"] = check_data_file_exists(user_doc.filename)
            except Exception as e:
                entry["content_preview"] = f"[Fehler: {e!s}]"

        result.append(entry)

    return result


@router.delete("/documents/{doc_id:int}", response_model=Success)
def delete_document(doc_id: int, user: CurrentUser):
    from app.core.models import UserDocument

    db_session = get_db_service().get_session()
    try:
        user_doc = db_session.query(UserDocument).filter_by(id=doc_id).first()
        if user_doc:
            db_session.expunge(user_doc)
    finally:
        db_session.close()
    if not user_doc:
        raise ApiError(404, "Dokument nicht gefunden", success=False)

    # Collection des Bereichs, dem das Dokument gehoert (nicht des Requests).
    doc_knowledge = get_knowledge(user_doc.knowledge_id) or get_knowledge(None)
    if reindex.is_locked(doc_knowledge["id"]):
        raise ApiError(409, "Das Knowledge wird gerade neu indexiert", success=False)
    # Schreibrecht, oder: wer das Dokument hochgeladen hat, darf es auch loeschen (und sieht das Knowledge)
    uploader = user_doc.uploaded_by is not None and user_doc.uploaded_by == user["id"]
    if not (
        can_access_knowledge(user, doc_knowledge["id"], "write")
        or (uploader and can_access_knowledge(user, doc_knowledge["id"], "read"))
    ):
        raise ApiError(404, "Dokument nicht gefunden", success=False)
    try:
        my_uploads.delete_knowledge_document(user_doc, doc_knowledge, runtime.state.client)
    except my_uploads.UploadError as e:
        raise ApiError(e.status, e.message, success=False)
    return Success()


def _markdown_download(text: str, stem: str) -> Response:
    return Response(
        text,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(stem)}.md"},
    )


@router.get("/my-uploads", response_model=list[MyUpload])
def my_uploads_list(user: CurrentUser):
    return my_uploads.list_uploads(user["id"])


@router.get("/my-uploads/{kind}/{item_id}/markdown", response_model=MarkdownView)
def my_uploads_markdown(kind: str, item_id: str, user: CurrentUser, download: str | None = None):
    try:
        row = my_uploads.get_owned(kind, item_id, user["id"])
    except my_uploads.UploadError as e:
        raise ApiError(e.status, e.message, success=False)
    text = my_uploads.read_markdown(kind, row)
    if text is None:
        raise ApiError(404, "Kein Markdown gespeichert", success=False)
    name = Path(row.filename).stem
    if download == "1":
        return _markdown_download(text, name)
    return {"filename": f"{name}.md", "markdown": text}


@router.delete("/my-uploads/{kind}/{item_id}", response_model=Success)
def my_uploads_delete(kind: str, item_id: str, user: CurrentUser):
    try:
        row = my_uploads.get_owned(kind, item_id, user["id"])
        my_uploads.delete_upload(kind, row, runtime.state.client)
    except my_uploads.UploadError as e:
        raise ApiError(e.status, e.message, success=False)
    return Success()


@router.get("/documents/{doc_id:int}/markdown", response_model=MarkdownView)
def document_markdown(doc_id: int, user: CurrentUser, download: str | None = None):
    """Das beim Einlesen erzeugte Markdown (so sieht der Splitter das Dokument).
    ``?download=1`` liefert eine Datei."""
    from app.core.models import UserDocument

    db_session = get_db_service().get_session()
    try:
        user_doc = db_session.query(UserDocument).filter_by(id=doc_id).first()
        knowledge = get_knowledge(user_doc.knowledge_id) if user_doc else None
        if not user_doc or not knowledge or not can_access_knowledge(user, knowledge["id"], "read"):
            raise ApiError(404, "Dokument nicht gefunden", success=False)
        path = my_uploads.markdown_path(user_doc)
        filename = user_doc.filename
        uploaded_by = user_doc.uploaded_by
    finally:
        db_session.close()

    if uploaded_by != user["id"]:
        # Nachweis, wer fremde Dokumente im Volltext ansieht (eigene Uploads nicht)
        from app.auth import audit

        audit.record(
            audit.DOCUMENT_VIEW,
            actor_user_id=user["id"],
            object_type="document",
            object_id=doc_id,
            detail={"knowledge_id": knowledge["id"], "download": download == "1"},
        )

    if path is None:
        raise ApiError(
            404,
            "Für dieses Dokument ist kein Markdown gespeichert "
            "(früher eingelesen). Es entsteht beim Neu-Indexieren.",
            success=False,
        )

    text = path.read_text(encoding="utf-8")
    if download == "1":
        return _markdown_download(text, Path(filename).stem)
    return {"filename": filename, "markdown": text}


@router.post("/documents/upload", response_model=UploadResult)
def upload_document(
    knowledge: ScopedKnowledge,
    user: CurrentUser,
    files: list[UploadFile] = File(default=[]),
    knowledge_id: str | None = Form(default=None),
):
    from app.knowledge.files import store_uploaded_files

    result = store_uploaded_files(
        [Upload(f) for f in files],
        paths=knowledge_paths(knowledge),
        uploader_id=user["id"],
    )

    if result["duplicate"]:
        raise ApiError(
            409,
            f'Datei "{result["duplicate"]}" wurde bereits hochgeladen und '
            f"verarbeitet. Bitte zuerst das bestehende Dokument löschen, "
            f"um es zu ersetzen.",
            success=False,
        )

    documents = result["documents"]
    data_files = result["data_files"]

    if not documents and not data_files:
        raise ApiError(400, "Keine gültigen Dateien", success=False)

    if data_files and not documents:
        message = (
            f"{len(data_files)} strukturierte Datei(en) direkt verfügbar "
            f"(Excel/CSV, keine Ingestion nötig)."
        )
    elif documents and not data_files:
        message = f"{len(documents)} Datei(en) hochgeladen. Bitte Ingestion starten."
    else:
        message = (
            f"{len(documents)} Datei(en) hochgeladen (Ingestion starten), "
            f"{len(data_files)} strukturierte Datei(en) sofort verfügbar."
        )

    # Referenz-Chunks für neu hochgeladene strukturierte Daten best-effort
    # indexieren, damit sie ohne zusätzliche Ingestion semantisch auffindbar
    # sind. Falls die Embedding-Ressourcen noch nicht geladen sind, werden sie
    # beim nächsten "Ingestion starten" nachgezogen.
    if data_files:
        _index_data_references_best_effort(knowledge)
        refresh_index(knowledge["id"])

    return {
        "success": True,
        "documents": documents,
        "data_files": data_files,
        "skipped": result["skipped"],
        "files": documents + data_files,
        "message": message,
    }


@router.post("/documents/upload/clear", response_model=Cleared)
def clear_uploaded_files(knowledge: ScopedKnowledge, _body: KnowledgeBody | None = None):
    target_dir = knowledge_paths(knowledge).files_dir
    deleted = 0
    errors = []

    if target_dir.exists():
        for file_path in target_dir.iterdir():
            if not file_path.is_file():
                continue
            try:
                file_path.unlink()
                deleted += 1
            except Exception as e:
                errors.append(str(e))

    return {"success": True, "deleted": deleted, "errors": errors}


@router.get("/data-files", response_model=list[DataFile])
def data_files(knowledge: ScopedKnowledge):
    from app.knowledge.files import list_data_files

    return list_data_files(knowledge_paths(knowledge).data_dir)


DATA_PREVIEW_ROWS = 200


def _data_file_path(knowledge: dict, filename: str) -> Path | None:
    """Strukturierte Datei im Datenordner des Knowledge (nie ausserhalb)."""
    from app.knowledge.files import STRUCTURED_EXTS

    folder = knowledge_paths(knowledge).data_dir.resolve()
    path = (folder / os.path.basename(filename)).resolve()
    if path.parent != folder or path.suffix.lower() not in STRUCTURED_EXTS or not path.is_file():
        return None
    return path


@router.get("/data-files/{filename:path}", response_model=DataPreview)
def data_file_view(filename: str, knowledge: ScopedKnowledge, download: str | None = None):
    """Vorschau (erste Zeilen) einer CSV/Excel-Datei; ``?download=1`` liefert die Datei selbst."""
    path = _data_file_path(knowledge, filename)
    if path is None:
        raise ApiError(404, "Datei nicht gefunden", success=False)
    if download == "1":
        return FileResponse(path, filename=path.name)

    from app.ingest.pipeline import _load_dataframe

    try:
        df = _load_dataframe(str(path))
    except Exception as e:
        raise ApiError(422, f"Datei nicht lesbar: {e}", success=False)
    head = df.head(DATA_PREVIEW_ROWS)
    return {
        "filename": path.name,
        "columns": [str(c) for c in df.columns],
        "rows": [["" if v != v else str(v) for v in row] for row in head.itertuples(index=False)],
        "total_rows": len(df),
        "truncated": len(df) > DATA_PREVIEW_ROWS,
    }


@router.delete("/data-files/{filename:path}", response_model=Success)
def delete_data_file_route(filename: str, knowledge: ScopedKnowledge):
    from app.knowledge.files import delete_data_file

    paths = knowledge_paths(knowledge)
    if not delete_data_file(filename, data_dir=paths.data_dir):
        raise ApiError(404, "Datei nicht gefunden", success=False)

    # Zugehörigen Verweis-Chunk aus der Vektordatenbank entfernen (best-effort).
    try:
        from app.ingest.pipeline import delete_data_reference

        delete_data_reference(knowledge["collection_name"], f"{paths.name_prefix}{filename}")
    except Exception as e:
        logger.warning(f"Verweis-Chunk für {filename} konnte nicht gelöscht werden: {e}")

    refresh_index(knowledge["id"])
    return Success()


@router.post("/documents/ingest", response_model=IngestResult)
def trigger_ingestion(knowledge: ScopedKnowledge, _body: KnowledgeBody | None = None):
    from app.ingest.pipeline import DocumentIngestion

    try:
        if not wait_for_resources(timeout=180):
            raise ApiError(
                503,
                "RAG-System wird noch initialisiert (Embedding-Modell "
                "lädt). Bitte in wenigen Sekunden erneut versuchen.",
                success=False,
            )
    except RuntimeError as e:
        raise ApiError(503, str(e), success=False)

    status = runtime.state.ingestion_status
    if status["is_ingesting"]:
        raise ApiError(409, "Ingestion läuft bereits", success=False)

    status["is_ingesting"] = True
    status["knowledge_id"] = knowledge["id"]
    try:
        ingest = DocumentIngestion(
            sparse_embeddings=runtime.state.sparse_embeddings,
            knowledge=knowledge,
        )
        ingest.ingest_documents()
        # Fehler und Warnungen einzelner Dateien gehen an die Oberflaeche (nichts schlaegt stumm fehl).
        return {
            "success": not ingest.errors or ingest.processed > 0,
            "message": f"{ingest.processed} Datei(en) eingelesen"
            + (f", {len(ingest.errors)} fehlgeschlagen" if ingest.errors else ""),
            "processed": ingest.processed,
            "errors": ingest.errors,
            "warnings": ingest.warnings,
        }
    except Exception as e:
        raise ApiError(500, str(e), success=False)
    finally:
        status["is_ingesting"] = False
        status["knowledge_id"] = None


def _client(success_flag: bool = True):
    client = runtime.state.client
    if not client:
        if success_flag:
            raise ApiError(503, _NO_CLIENT, success=False)
        raise ApiError(503, _NO_CLIENT)
    return client


@router.get("/vectordb", response_model=VectorList)
def vectordb(
    knowledge: ScopedKnowledge,
    limit: int = 20,
    offset: str | None = None,
    filename: str = "",
    file_status: str = "",
):
    client = _client(success_flag=False)
    filename_filter = filename.strip()
    file_status_filter = file_status.strip()
    collection = knowledge["collection_name"]

    try:
        scroll_filter = None
        if filename_filter:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename", match=models.MatchValue(value=filename_filter)
                    )
                ]
            )

        points, next_offset = client.scroll(
            collection_name=collection,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=scroll_filter,
        )

        documents = []
        for point in points:
            payload = point.payload or {}
            name = payload.get("filename", "N/A")
            has_data_file = check_data_file_exists(name) if name != "N/A" else False

            if file_status_filter == "has_file" and not has_data_file:
                continue
            if file_status_filter == "no_file" and has_data_file:
                continue

            documents.append(
                {
                    "id": str(point.id),
                    "filename": name,
                    "source": payload.get("source", "N/A"),
                    "content_preview": (
                        payload.get("page_content", "") or payload.get("original_content", "")
                    )[:50]
                    + "...",
                    "has_data_file": has_data_file,
                }
            )

        all_points, _ = client.scroll(
            collection_name=collection,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        filenames = sorted({p.payload.get("filename", "N/A") for p in all_points if p.payload})
    except Exception as e:
        raise ApiError(500, f"Fehler beim Laden der Dokumente: {e!s}")

    return {
        "documents": documents,
        "next_offset": next_offset,
        "filenames": filenames,
        "limit": limit,
    }


@router.get("/vectordb/{doc_id}", response_model=VectorDetail)
def vectordb_detail(doc_id: str, knowledge: ScopedKnowledge):
    client = _client(success_flag=False)
    try:
        result = client.retrieve(
            collection_name=knowledge["collection_name"],
            ids=[doc_id],
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        raise ApiError(500, f"Fehler beim Laden des Dokuments: {e!s}")
    if not result:
        raise ApiError(404, "Dokument nicht gefunden")

    point = result[0]
    payload = point.payload or {}
    return {
        "id": str(point.id),
        "page_content": payload.get("page_content", ""),
        "original_content": payload.get("original_content", ""),
        "metadata": {
            k: v for k, v in payload.items() if k not in ["page_content", "original_content"]
        },
    }


@router.delete("/vectordb/{doc_id}", response_model=Success)
def delete_vectordb_document(doc_id: str, knowledge: ScopedKnowledge):
    client = _client()
    collection = knowledge["collection_name"]
    try:
        result = client.retrieve(
            collection_name=collection, ids=[doc_id], with_payload=True, with_vectors=False
        )
        payload = result[0].payload if result else None

        client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=[doc_id]),
        )

        if payload:
            delete_associated_file(payload)
    except Exception as e:
        raise ApiError(500, str(e), success=False)
    return Success()


@router.post("/vectordb/batch-delete", response_model=Success)
def vectordb_batch_delete(knowledge: ScopedKnowledge, body: BatchDelete | None = None):
    client = _client()
    filename = body.filename if body else None
    if not filename:
        raise ApiError(400, "Filename erforderlich", success=False)
    collection = knowledge["collection_name"]

    try:
        points, _ = client.scroll(
            collection_name=collection,
            filter=models.Filter(
                must=[models.FieldCondition(key="filename", match=models.MatchValue(value=filename))]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        if points:
            if points[0].payload:
                delete_associated_file(points[0].payload)

            point_ids = [str(p.id) for p in points]
            client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=point_ids),
            )
    except Exception as e:
        raise ApiError(500, str(e), success=False)
    return Success()


@router.post("/vectordb/clear", response_model=Success)
def vectordb_clear_all(knowledge: ScopedKnowledge, body: ClearAll | None = None):
    client = _client()
    if (body.confirmation if body else "") != "DELETE_ALL":
        raise ApiError(400, "Bestätigung erforderlich", success=False)

    try:
        from app.rag.collections import ensure_collection

        client.delete_collection(knowledge["collection_name"])
        ensure_collection(client, knowledge["collection_name"], knowledge["embedding_dimension"])
    except Exception as e:
        raise ApiError(500, str(e), success=False)
    return Success()
