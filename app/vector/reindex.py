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
Neu indexieren: baut den Index eines Knowledge aus dem gespeicherten Markdown neu auf.

Noetig nach einem Wechsel des Embedding-Modells oder der Zerlege-Einstellungen. Ablauf ohne
Ausfall (Blue/Green): eine neue Collection ``<name>_g<n>`` entsteht neben der alten, die Suche
laeuft solange auf der alten weiter. Erst wenn alles fertig ist, stellt eine DB-Transaktion
das Knowledge um und die alte Collection wird geloescht. Bei einem Fehler wird die neue
Collection verworfen; die alte bleibt aktiv.

Dokumente ohne gespeichertes Markdown (Altbestand) werden aus dem archivierten Original neu
geparst; fehlt es, wird das Markdown notduerftig aus den alten Chunks zusammengesetzt (ohne
Ueberschriften).
"""

import re
import threading
import time
from logging import getLogger
from pathlib import Path

from qdrant_client import models as qdrant_models

from app.core.database_service import get_db_service
from app.core.models import UserDocument
from app.ingest.index_doc import refresh_index
from app.ingest.indexer import PARSER_VERSION, save_markdown
from app.ingest.tables import make_sink
from app.rag import knowledge as kb_store
from app.rag import model_settings, models_runtime
from app.vector.collections import ensure_collection

logger = getLogger(__name__)


class ReindexBusy(Exception):
    """Es laeuft bereits ein Neu-Indexieren."""


_lock = threading.Lock()
_state: dict = {
    "running": False,
    "queue": [],  # Knowledge-IDs, die noch folgen
    "current": None,  # {"id", "name", "done", "total"}
    "results": [],  # {"id", "name", "ok", "error"}
    "started_at": None,
}


def status() -> dict:
    with _lock:
        return {
            "running": _state["running"],
            "queue": list(_state["queue"]),
            "current": dict(_state["current"]) if _state["current"] else None,
            "results": list(_state["results"]),
            "started_at": _state["started_at"],
        }


def is_locked(knowledge_id: int | None) -> bool:
    """Waehrend des Neu-Indexierens sind Einlesen/Loeschen/Umbenennen dieses Knowledge gesperrt."""
    with _lock:
        if not _state["running"]:
            return False
        current = _state["current"]
        return knowledge_id in _state["queue"] or bool(
            current and current["id"] == knowledge_id
        )


def start(knowledge_ids: list[int], client, sparse, wait: bool = False) -> None:
    """Startet den Hintergrund-Job fuer die Knowledge-Bereiche (nacheinander).

    Raises:
        ReindexBusy: wenn schon ein Lauf aktiv ist.
    """
    with _lock:
        if _state["running"]:
            raise ReindexBusy("Es läuft bereits ein Neu-Indexieren")
        _state.update(
            running=True,
            queue=list(knowledge_ids),
            current=None,
            results=[],
            started_at=time.time(),
        )
    thread = threading.Thread(
        target=_run, args=(client, sparse), name="reindex", daemon=True
    )
    thread.start()
    if wait:
        thread.join()


def _run(client, sparse) -> None:
    try:
        while True:
            with _lock:
                if not _state["queue"]:
                    break
                kid = _state["queue"][0]
            k = kb_store.get_knowledge(kid)
            if not k:
                _finish(kid, "?", False, "Knowledge nicht gefunden")
                continue
            with _lock:
                _state["current"] = {
                    "id": kid,
                    "name": k["name"],
                    "done": 0,
                    "total": 0,
                }
            try:
                reindex_knowledge(k, client, sparse, _progress)
                _finish(kid, k["name"], True, None)
            except Exception as e:
                logger.exception(f"Neu-Indexieren von '{k['name']}' fehlgeschlagen")
                message = f"{type(e).__name__}: {e}"[:300]
                kb_store.set_reindex_error(kid, message)
                _finish(kid, k["name"], False, message)
    finally:
        with _lock:
            _state.update(running=False, queue=[], current=None)


def _progress(done: int, total: int) -> None:
    with _lock:
        if _state["current"]:
            _state["current"].update(done=done, total=total)


def _finish(kid: int, name: str, ok: bool, error: str | None) -> None:
    with _lock:
        _state["results"].append({"id": kid, "name": name, "ok": ok, "error": error})
        _state["queue"] = [i for i in _state["queue"] if i != kid]


def next_collection_name(client, current: str) -> str:
    base = re.sub(r"_g\d+$", "", current)
    n = 1
    while client.collection_exists(f"{base}_g{n}"):
        n += 1
    return f"{base}_g{n}"


def reindex_knowledge(
    k: dict, client, sparse, progress=lambda done, total: None
) -> None:
    """Baut ein Knowledge mit dem aktuellen globalen Modell und den aktuellen Einstellungen neu auf."""
    from app.rag import sparse as sparse_lang
    from app.vector.ingestion import DocumentIngestion, index_data_references

    spec = model_settings.embedding_spec()
    language = (
        sparse_lang.default_language()
    )  # der neue Index nutzt die aktuelle BM25-Sprache
    k = {**k, "bm25_language": language}
    embedder = models_runtime.get_embedder(
        spec
    )  # scheitert frueh, bevor etwas angelegt wird
    dimension = embedder.dimension
    paths = kb_store.knowledge_paths(k)
    old_collection = k["collection_name"]
    new_collection = next_collection_name(client, old_collection)
    documents = get_db_service().get_all_documents(k["id"])

    ensure_collection(client, new_collection, dimension)
    try:
        ingest = DocumentIngestion(
            knowledge=k,
            collection_name=new_collection,
            embedding_spec=spec,
            sparse_embeddings=sparse,
            track_documents=False,
        )
        progress(0, len(documents))
        for i, doc in enumerate(documents, 1):
            markdown = _markdown_for(doc, k, paths, ingest, client, old_collection)
            if markdown and markdown.strip():
                ingest.index_markdown(
                    markdown,
                    doc.filename,
                    doc.document_id,
                    sparse,
                    uploaded_by=doc.uploaded_by,
                    uploaded_at=doc.uploaded_at,
                )
            else:
                logger.warning(f"Kein Inhalt für '{doc.filename}', wird übersprungen")
            progress(i, len(documents))

        # Verweis-Chunks der CSV/Excel-Dateien (inkl. aus Dokumenten extrahierter Tabellen)
        index_data_references(
            embeddings=embedder,
            sparse_model=sparse_lang.resolve(sparse, language),
            collection_name=new_collection,
            data_dir=str(paths.data_dir),
            name_prefix=paths.name_prefix,
            knowledge_id=k["id"],
        )
        old = kb_store.swap_collection(k["id"], new_collection, spec, dimension)
    except Exception:
        _drop(client, new_collection)  # alte Collection bleibt aktiv
        raise

    if old:
        _drop(client, old)
    refresh_index(k["id"])
    logger.info(
        f"'{k['name']}' neu indexiert: {len(documents)} Dokumente -> {new_collection}"
    )


def _drop(client, collection: str) -> None:
    try:
        client.delete_collection(collection)
    except Exception as e:
        logger.warning(f"Collection '{collection}' konnte nicht gelöscht werden: {e}")


def _markdown_for(
    doc: UserDocument, k, paths, ingest, client, old_collection
) -> str | None:
    """Markdown des Dokuments: gespeichert, sonst aus dem Original, sonst aus den alten Chunks."""
    if doc.markdown_path and Path(doc.markdown_path).is_file():
        return Path(doc.markdown_path).read_text(encoding="utf-8")

    markdown = _reparse_original(doc, paths, ingest)
    if not markdown:
        markdown = _markdown_from_chunks(client, old_collection, doc.document_id)
    if markdown and markdown.strip():
        path = save_markdown(paths.markdown_dir, doc.document_id, markdown)
        _remember_markdown(doc.id, str(path))
    return markdown


def _reparse_original(doc: UserDocument, paths, ingest) -> str | None:
    matches = [p for p in Path(paths.processed_dir).rglob(doc.filename) if p.is_file()]
    if not matches:
        return None
    original = matches[0]
    try:
        sink = make_sink(paths, Path(doc.filename).stem, doc.document_id)
        return ingest._load_markdown(str(original), original.suffix.lower(), sink)
    except Exception as e:
        logger.warning(f"Original von '{doc.filename}' nicht neu lesbar: {e}")
        return None


def _markdown_from_chunks(client, collection: str, file_group_id: str) -> str | None:
    """Notloesung: Text der alten Chunks in Dateireihenfolge (Ueberschriften gehen verloren)."""
    flt = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="file_group_id", match=qdrant_models.MatchValue(value=file_group_id)
            )
        ]
    )
    rows, offset = [], None
    while True:
        points, offset = client.scroll(
            collection,
            scroll_filter=flt,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        rows.extend(points)
        if offset is None:
            break
    if not rows:
        return None
    rows.sort(key=lambda p: (p.payload or {}).get("chunk_index", 0))
    return "\n\n".join((p.payload or {}).get("original_content") or "" for p in rows)


def _remember_markdown(document_row_id: int, path: str) -> None:
    db = get_db_service().get_session()
    try:
        doc = db.query(UserDocument).filter_by(id=document_row_id).first()
        if doc:
            doc.markdown_path = path
            doc.parser_version = PARSER_VERSION
            db.commit()
    finally:
        db.close()
