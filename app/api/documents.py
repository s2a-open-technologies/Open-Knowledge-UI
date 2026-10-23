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
from functools import wraps
from logging import getLogger
from pathlib import Path
from urllib.parse import quote

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    g,
    jsonify,
    make_response,
    request,
    send_file,
)
from qdrant_client import models

from app import resources_ready, wait_for_resources
from app.api._common import _abort_if_reindexing
from app.auth.access import (
    can_access_knowledge,
)
from app.auth.auth import auth_required_api, current_user, require_role
from app.chat import my_uploads
from app.core.database_service import get_db_service
from app.ingest import reindex
from app.knowledge.files import check_data_file_exists, delete_associated_file
from app.knowledge.index_doc import refresh_index
from app.knowledge.service import (
    get_knowledge,
    is_remote,
    knowledge_paths,
)
from app.rag import model_settings

bp = Blueprint("documents", __name__, url_prefix="/api")
logger = getLogger(__name__)


@bp.route("/ingestion-status", methods=["GET"])
@auth_required_api
def ingestion_status():
    """Geteilter Ingestion-Status fuer den Navbar-Indikator per Polling."""
    return jsonify(current_app.extensions["ingestion_status"])


def _current_knowledge() -> dict:
    """Knowledge-Bereich des Requests (``knowledge_id`` als Query-/Form-/JSON-Feld);
    ohne Angabe gilt der Standard-Bereich. Pro Request gecacht."""
    cached = getattr(g, "_knowledge", None)
    if cached is not None:
        return cached
    kid = request.args.get("knowledge_id", type=int)
    if kid is None:
        kid = request.form.get("knowledge_id", type=int)
    if kid is None and request.is_json:
        raw = (request.get_json(silent=True) or {}).get("knowledge_id")
        kid = int(raw) if str(raw or "").isdigit() else None
    knowledge = get_knowledge(kid)
    # Lesende Aufrufe brauchen Lese-, aendernde Schreibrechte auf dieses Knowledge.
    level = "read" if request.method in ("GET", "HEAD") else "write"
    if knowledge is None or not can_access_knowledge(
        current_user(), knowledge["id"], level
    ):
        abort(
            make_response(
                jsonify({"success": False, "error": "Knowledge nicht gefunden"}), 404
            )
        )
    if is_remote(knowledge) and (
        level == "write" or request.path.startswith("/api/vectordb")
    ):
        abort(
            make_response(
                jsonify(
                    {"success": False, "error": "Diese Quelle ist schreibgeschützt"}
                ),
                409,
            )
        )
    if level == "write":
        _abort_if_reindexing(knowledge["id"])
    g._knowledge = knowledge
    return knowledge


def knowledge_scoped(f):
    """Prueft den Zugriff auf das Knowledge des Requests *vor* dem eigentlichen
    Handler. Die Pruefung allein in ``_kb_col()`` wuerde teils erst in einem
    ``try/except Exception`` greifen (dort wird der Abbruch zu einem 500)."""

    @wraps(f)
    def decorated(*args, **kwargs):
        _current_knowledge()
        return f(*args, **kwargs)

    return decorated


def _kb_col() -> str:
    return _current_knowledge()["collection_name"]


def _get_collection_and_embeddings():
    """Liefert (dense_embeddings, sparse, collection) oder None, wenn die
    Embedding-Ressourcen noch nicht geladen sind."""

    if not resources_ready():
        return None

    sparse = current_app.extensions.get("sparse_embeddings")
    if not sparse:
        return None
    # Modell des aktuellen Knowledge (nicht blind das globale)
    return (
        model_settings.embedder_for(_current_knowledge()),
        sparse,
        "open_knowledge_ui",
    )


def _index_data_references_best_effort() -> None:
    """Indexiert Verweis-Chunks für alle strukturierten Dateien. Fehler werden
    geloggt, aber nie an den Upload-Request weitergereicht."""
    try:
        resources = _get_collection_and_embeddings()
        if resources is None:
            return
        dense, sparse, _ = resources
        from app.ingest.pipeline import index_data_references
        from app.rag import sparse as sparse_lang

        knowledge = _current_knowledge()
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


@bp.route("/documents", methods=["GET"])
@require_role("user")
@knowledge_scoped
def documents():
    db = get_db_service()
    knowledge = _current_knowledge()
    user_docs = db.get_all_documents(knowledge_id=knowledge["id"])

    client = current_app.extensions.get("client")
    result = []

    for user_doc in user_docs:
        entry = {
            "id": user_doc.id,
            "document_id": user_doc.document_id,
            "filename": user_doc.filename,
            "uploaded_at": user_doc.uploaded_at.isoformat()
            if user_doc.uploaded_at
            else None,
            "content_preview": "[Nicht mehr in Qdrant]",
            "source": "N/A",
            "has_data_file": False,
            "has_markdown": _markdown_path(user_doc) is not None,
        }

        if client:
            try:
                points, _ = client.scroll(
                    collection_name=_kb_col(),
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
                        payload.get("page_content", "")
                        or payload.get("original_content", "")
                    )[:100] + "..."
                    entry["source"] = payload.get("source", "N/A")
                    entry["has_data_file"] = check_data_file_exists(user_doc.filename)
            except Exception as e:
                entry["content_preview"] = f"[Fehler: {e!s}]"

        result.append(entry)

    return jsonify(result)


@bp.route("/documents/<int:doc_id>", methods=["DELETE"])
@require_role("user")
def delete_document(doc_id):
    from app.core.models import UserDocument

    db_session = get_db_service().get_session()
    try:
        user_doc = db_session.query(UserDocument).filter_by(id=doc_id).first()
        if user_doc:
            db_session.expunge(user_doc)
    finally:
        db_session.close()
    if not user_doc:
        return jsonify({"success": False, "error": "Dokument nicht gefunden"}), 404

    # Collection des Bereichs, dem das Dokument gehoert (nicht des Requests).
    doc_knowledge = get_knowledge(user_doc.knowledge_id) or _current_knowledge()
    if reindex.is_locked(doc_knowledge["id"]):
        return jsonify(
            {"success": False, "error": "Das Knowledge wird gerade neu indexiert"}
        ), 409
    user = current_user()
    # Schreibrecht, oder: wer das Dokument hochgeladen hat, darf es auch loeschen (und sieht das Knowledge)
    uploader = user_doc.uploaded_by is not None and user_doc.uploaded_by == user["id"]
    if not (
        can_access_knowledge(user, doc_knowledge["id"], "write")
        or (uploader and can_access_knowledge(user, doc_knowledge["id"], "read"))
    ):
        return jsonify({"success": False, "error": "Dokument nicht gefunden"}), 404
    try:
        my_uploads.delete_knowledge_document(
            user_doc, doc_knowledge, current_app.extensions.get("client")
        )
    except my_uploads.UploadError as e:
        return jsonify({"success": False, "error": e.message}), e.status
    return jsonify({"success": True})


MARKDOWN_ROOT = Path("./data/markdown")


def _markdown_path(user_doc) -> Path | None:
    return my_uploads.markdown_path(user_doc)


def _upload_error(e):
    return jsonify({"success": False, "error": e.message}), e.status


@bp.route("/my-uploads", methods=["GET"])
@auth_required_api
def my_uploads_list():
    return jsonify(my_uploads.list_uploads(current_user()["id"]))


@bp.route("/my-uploads/<kind>/<item_id>/markdown", methods=["GET"])
@auth_required_api
def my_uploads_markdown(kind, item_id):
    try:
        row = my_uploads.get_owned(kind, item_id, current_user()["id"])
    except my_uploads.UploadError as e:
        return _upload_error(e)
    text = my_uploads.read_markdown(kind, row)
    if text is None:
        return jsonify({"success": False, "error": "Kein Markdown gespeichert"}), 404
    name = Path(row.filename).stem
    if request.args.get("download") == "1":
        return Response(
            text,
            mimetype="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}.md"
            },
        )
    return jsonify({"filename": f"{name}.md", "markdown": text})


@bp.route("/my-uploads/<kind>/<item_id>", methods=["DELETE"])
@auth_required_api
def my_uploads_delete(kind, item_id):
    try:
        row = my_uploads.get_owned(kind, item_id, current_user()["id"])
        my_uploads.delete_upload(kind, row, current_app.extensions.get("client"))
    except my_uploads.UploadError as e:
        return _upload_error(e)
    return jsonify({"success": True})


@bp.route("/documents/<int:doc_id>/markdown", methods=["GET"])
@require_role("user")
def document_markdown(doc_id):
    """Das beim Einlesen erzeugte Markdown (so sieht der Splitter das Dokument).
    ``?download=1`` liefert eine Datei."""
    from app.core.models import UserDocument

    db_session = get_db_service().get_session()
    try:
        user_doc = db_session.query(UserDocument).filter_by(id=doc_id).first()
        knowledge = get_knowledge(user_doc.knowledge_id) if user_doc else None
        if (
            not user_doc
            or not knowledge
            or not can_access_knowledge(current_user(), knowledge["id"], "read")
        ):
            return jsonify({"success": False, "error": "Dokument nicht gefunden"}), 404
        path = _markdown_path(user_doc)
        filename = user_doc.filename
    finally:
        db_session.close()

    if path is None:
        return jsonify(
            {
                "success": False,
                "error": "Für dieses Dokument ist kein Markdown gespeichert "
                "(früher eingelesen). Es entsteht beim Neu-Indexieren.",
            }
        ), 404

    text = path.read_text(encoding="utf-8")
    if request.args.get("download") == "1":
        return Response(
            text,
            mimetype="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(Path(filename).stem)}.md"
            },
        )
    return jsonify({"filename": filename, "markdown": text})


@bp.route("/documents/upload", methods=["POST"])
@require_role("user")
@knowledge_scoped
def upload_document():
    from app.knowledge.files import store_uploaded_files

    result = store_uploaded_files(
        request.files.getlist("files"),
        paths=knowledge_paths(_current_knowledge()),
        uploader_id=current_user()["id"],
    )

    if result["duplicate"]:
        return jsonify(
            {
                "success": False,
                "error": (
                    f'Datei "{result["duplicate"]}" wurde bereits hochgeladen und '
                    f"verarbeitet. Bitte zuerst das bestehende Dokument löschen, "
                    f"um es zu ersetzen."
                ),
            }
        ), 409

    documents = result["documents"]
    data_files = result["data_files"]

    if not documents and not data_files:
        return jsonify({"success": False, "error": "Keine gültigen Dateien"}), 400

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
        _index_data_references_best_effort()
        refresh_index(_current_knowledge()["id"])

    return jsonify(
        {
            "success": True,
            "documents": documents,
            "data_files": data_files,
            "skipped": result["skipped"],
            "files": documents + data_files,
            "message": message,
        }
    )


@bp.route("/documents/upload/clear", methods=["POST"])
@require_role("user")
@knowledge_scoped
def clear_uploaded_files():
    target_dir = knowledge_paths(_current_knowledge()).files_dir
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

    return jsonify({"success": True, "deleted": deleted, "errors": errors})


@bp.route("/data-files", methods=["GET"])
@require_role("user")
@knowledge_scoped
def data_files():
    from app.knowledge.files import list_data_files

    return jsonify(list_data_files(knowledge_paths(_current_knowledge()).data_dir))


DATA_PREVIEW_ROWS = 200


def _data_file_path(filename: str) -> Path | None:
    """Strukturierte Datei im Datenordner des aktuellen Knowledge (nie ausserhalb)."""
    from app.knowledge.files import STRUCTURED_EXTS

    folder = knowledge_paths(_current_knowledge()).data_dir.resolve()
    path = (folder / os.path.basename(filename)).resolve()
    if (
        path.parent != folder
        or path.suffix.lower() not in STRUCTURED_EXTS
        or not path.is_file()
    ):
        return None
    return path


@bp.route("/data-files/<path:filename>", methods=["GET"])
@require_role("user")
@knowledge_scoped
def data_file_view(filename):
    """Vorschau (erste Zeilen) einer CSV/Excel-Datei; ``?download=1`` liefert die Datei selbst."""
    path = _data_file_path(filename)
    if path is None:
        return jsonify({"success": False, "error": "Datei nicht gefunden"}), 404
    if request.args.get("download") == "1":
        return send_file(path, as_attachment=True, download_name=path.name)

    from app.ingest.pipeline import _load_dataframe

    try:
        df = _load_dataframe(str(path))
    except Exception as e:
        return jsonify({"success": False, "error": f"Datei nicht lesbar: {e}"}), 422
    head = df.head(DATA_PREVIEW_ROWS)
    return jsonify(
        {
            "filename": path.name,
            "columns": [str(c) for c in df.columns],
            "rows": [
                ["" if v != v else str(v) for v in row]
                for row in head.itertuples(index=False)
            ],
            "total_rows": len(df),
            "truncated": len(df) > DATA_PREVIEW_ROWS,
        }
    )


@bp.route("/data-files/<path:filename>", methods=["DELETE"])
@require_role("user")
@knowledge_scoped
def delete_data_file_route(filename):
    from app.knowledge.files import delete_data_file

    paths = knowledge_paths(_current_knowledge())
    if not delete_data_file(filename, data_dir=paths.data_dir):
        return jsonify({"success": False, "error": "Datei nicht gefunden"}), 404

    # Zugehörigen Verweis-Chunk aus der Vektordatenbank entfernen (best-effort).
    try:
        from app.ingest.pipeline import delete_data_reference

        delete_data_reference(_kb_col(), f"{paths.name_prefix}{filename}")
    except Exception as e:
        logger.warning(
            f"Verweis-Chunk für {filename} konnte nicht gelöscht werden: {e}"
        )

    refresh_index(_current_knowledge()["id"])
    return jsonify({"success": True})


@bp.route("/documents/ingest", methods=["POST"])
@require_role("user")
@knowledge_scoped
def trigger_ingestion():
    from app.ingest.pipeline import DocumentIngestion

    try:
        if not wait_for_resources(timeout=180):
            return jsonify(
                {
                    "success": False,
                    "error": "RAG-System wird noch initialisiert (Embedding-Modell "
                    "lädt). Bitte in wenigen Sekunden erneut versuchen.",
                }
            ), 503
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 503

    knowledge = _current_knowledge()
    status = current_app.extensions["ingestion_status"]
    if status["is_ingesting"]:
        return jsonify({"success": False, "error": "Ingestion läuft bereits"}), 409

    status["is_ingesting"] = True
    status["knowledge_id"] = knowledge["id"]
    try:
        ingest = DocumentIngestion(
            sparse_embeddings=current_app.extensions.get("sparse_embeddings"),
            knowledge=knowledge,
        )
        ingest.ingest_documents()
        # Fehler und Warnungen einzelner Dateien gehen an die Oberflaeche (nichts schlaegt stumm fehl).
        return jsonify(
            {
                "success": not ingest.errors or ingest.processed > 0,
                "message": f"{ingest.processed} Datei(en) eingelesen"
                + (f", {len(ingest.errors)} fehlgeschlagen" if ingest.errors else ""),
                "processed": ingest.processed,
                "errors": ingest.errors,
                "warnings": ingest.warnings,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        status["is_ingesting"] = False
        status["knowledge_id"] = None


@bp.route("/vectordb", methods=["GET"])
@require_role("user")
@knowledge_scoped
def vectordb():
    client = current_app.extensions.get("client")
    if not client:
        return jsonify({"error": "Qdrant Client nicht initialisiert"}), 503

    limit = request.args.get("limit", 20, type=int)
    offset = request.args.get("offset", None)
    filename_filter = request.args.get("filename", "").strip()
    file_status_filter = request.args.get("file_status", "").strip()

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
            collection_name=_kb_col(),
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=scroll_filter,
        )

        documents = []
        for point in points:
            payload = point.payload or {}
            filename = payload.get("filename", "N/A")
            has_data_file = (
                check_data_file_exists(filename) if filename != "N/A" else False
            )

            if file_status_filter == "has_file" and not has_data_file:
                continue
            if file_status_filter == "no_file" and has_data_file:
                continue

            documents.append(
                {
                    "id": str(point.id),
                    "filename": filename,
                    "source": payload.get("source", "N/A"),
                    "content_preview": (
                        payload.get("page_content", "")
                        or payload.get("original_content", "")
                    )[:50]
                    + "...",
                    "has_data_file": has_data_file,
                }
            )

        all_points, _ = client.scroll(
            collection_name=_kb_col(),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        filenames = sorted(
            {p.payload.get("filename", "N/A") for p in all_points if p.payload}
        )
    except Exception as e:
        return jsonify({"error": f"Fehler beim Laden der Dokumente: {e!s}"}), 500

    return jsonify(
        {
            "documents": documents,
            "next_offset": next_offset,
            "filenames": filenames,
            "limit": limit,
        }
    )


@bp.route("/vectordb/<doc_id>", methods=["GET"])
@require_role("user")
@knowledge_scoped
def vectordb_detail(doc_id):
    client = current_app.extensions.get("client")
    if not client:
        return jsonify({"error": "Qdrant Client nicht initialisiert"}), 503

    try:
        result = client.retrieve(
            collection_name=_kb_col(),
            ids=[doc_id],
            with_payload=True,
            with_vectors=False,
        )
        if not result:
            return jsonify({"error": "Dokument nicht gefunden"}), 404

        point = result[0]
        payload = point.payload or {}
        document = {
            "id": str(point.id),
            "page_content": payload.get("page_content", ""),
            "original_content": payload.get("original_content", ""),
            "metadata": {
                k: v
                for k, v in payload.items()
                if k not in ["page_content", "original_content"]
            },
        }
    except Exception as e:
        return jsonify({"error": f"Fehler beim Laden des Dokuments: {e!s}"}), 500

    return jsonify(document)


@bp.route("/vectordb/<doc_id>", methods=["DELETE"])
@require_role("user")
@knowledge_scoped
def delete_vectordb_document(doc_id):
    client = current_app.extensions.get("client")
    if not client:
        return jsonify(
            {"success": False, "error": "Qdrant Client nicht initialisiert"}
        ), 503

    try:
        result = client.retrieve(
            collection_name=_kb_col(),
            ids=[doc_id],
            with_payload=True,
            with_vectors=False,
        )
        payload = result[0].payload if result else None

        client.delete(
            collection_name=_kb_col(),
            points_selector=models.PointIdsList(points=[doc_id]),
        )

        if payload:
            delete_associated_file(payload)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/vectordb/batch-delete", methods=["POST"])
@require_role("user")
@knowledge_scoped
def vectordb_batch_delete():
    client = current_app.extensions.get("client")
    if not client:
        return jsonify(
            {"success": False, "error": "Qdrant Client nicht initialisiert"}
        ), 503

    data = request.get_json() or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"success": False, "error": "Filename erforderlich"}), 400

    try:
        points, _ = client.scroll(
            collection_name=_kb_col(),
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename", match=models.MatchValue(value=filename)
                    )
                ]
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
                collection_name=_kb_col(),
                points_selector=models.PointIdsList(points=point_ids),
            )

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/vectordb/clear", methods=["POST"])
@require_role("user")
@knowledge_scoped
def vectordb_clear_all():
    client = current_app.extensions.get("client")
    if not client:
        return jsonify(
            {"success": False, "error": "Qdrant Client nicht initialisiert"}
        ), 503

    data = request.get_json() or {}
    if data.get("confirmation", "") != "DELETE_ALL":
        return jsonify({"success": False, "error": "Bestätigung erforderlich"}), 400

    try:
        from app.rag.collections import ensure_collection

        client.delete_collection(_kb_col())
        ensure_collection(
            client, _kb_col(), _current_knowledge()["embedding_dimension"]
        )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
