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
Knowledge-Bereiche: Verwaltung und Pfad-Aufloesung.

Jeder Bereich besitzt eine eigene Qdrant-Collection. Der Standard-Bereich
(``is_default``) bleibt bei der urspruenglichen Collection ``open_knowledge_ui`` und den
urspruenglichen Ordnern, damit Bestandsdaten unveraendert weiterlaufen.
"""

import json
import os
import shutil
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

from app.core.database_service import get_db_service
from app.core.models import Knowledge, SqlValue, SqlValueIndex, UserDocument
from app.core.settings_service import SettingsService
from app.rag.collections import DEFAULT_COLLECTION

logger = getLogger(__name__)

FILES_ROOT = Path("./data")


@dataclass(frozen=True)
class KnowledgePaths:
    """Ordner eines Knowledge-Bereichs.

    ``name_prefix`` wird vor Dateinamen strukturierter Daten gesetzt, damit
    die Pandas-Tools sie relativ zu DATA_DIR finden (``knowledge_3/x.csv``).
    """

    files_dir: Path
    processed_dir: Path
    data_dir: Path
    name_prefix: str
    # gespeichertes Markdown der Dokumente (data/markdown/<knowledge_id>/)
    markdown_dir: Path


def _to_dict(k: Knowledge) -> dict:
    return {
        "id": k.id,
        "name": k.name,
        "description": k.description or "",
        "collection_name": k.collection_name,
        "is_default": bool(k.is_default),
        "embedding_model": k.embedding_model,
        "embedding_dimension": k.embedding_dimension,
        "embedding_spec": k.embedding_spec,
        "kind": k.kind or "local",
        "source_config": k.source_config,
        "index_signature": k.index_signature,
        "bm25_language": k.bm25_language,
        "last_reindex_error": k.last_reindex_error,
        "is_public": bool(k.is_public),
        "retrieval_profile_id": k.retrieval_profile_id,
        "created_at": k.created_at.isoformat() if k.created_at else None,
    }


def _global_data_dir() -> Path:
    with get_db_service().session_scope(commit=False) as db:
        return Path(SettingsService(db).get("DATA_DIR", "./data") or "./data")


def knowledge_paths(knowledge: dict) -> KnowledgePaths:
    """Loest die Ordner eines Bereichs auf (siehe ``KnowledgePaths``)."""
    data_dir = _global_data_dir()
    if knowledge["is_default"]:
        return KnowledgePaths(
            files_dir=FILES_ROOT / "files",
            processed_dir=FILES_ROOT / "processed_files",
            data_dir=data_dir,
            name_prefix="",
            markdown_dir=FILES_ROOT / "markdown" / str(knowledge["id"]),
        )
    base = FILES_ROOT / "knowledge" / str(knowledge["id"])
    sub = f"knowledge_{knowledge['id']}"
    return KnowledgePaths(
        files_dir=base / "files",
        processed_dir=base / "processed_files",
        data_dir=data_dir / sub,
        name_prefix=f"{sub}/",
        markdown_dir=FILES_ROOT / "markdown" / str(knowledge["id"]),
    )


def ensure_default_knowledge(db_session) -> None:
    """Legt den Standard-Bereich an, falls er fehlt (idempotent)."""
    # Nach dem Neu-Indexieren kann die Collection anders heissen als "open_knowledge_ui".
    existing = (
        db_session.query(Knowledge).filter_by(is_default=True).first()
        or db_session.query(Knowledge)
        .filter_by(collection_name=DEFAULT_COLLECTION)
        .first()
    )
    if existing:
        if not existing.is_default:
            existing.is_default = True
        return
    from app.rag import model_settings
    from app.rag import sparse as sparse_lang

    spec = model_settings.embedding_spec()
    db_session.add(
        Knowledge(
            name="Standard",
            description="Bisheriger Wissensbestand",
            collection_name=DEFAULT_COLLECTION,
            is_default=True,
            is_public=True,
            embedding_model=spec.model,
            embedding_spec=json.dumps(spec.to_dict()),
            index_signature=model_settings.current_signature(spec),
            bm25_language=sparse_lang.default_language(),
        )
    )


def list_knowledge() -> list[dict]:
    with get_db_service().session_scope(commit=False) as db:
        rows = db.query(Knowledge).order_by(Knowledge.is_default.desc(), Knowledge.id)
        return [_to_dict(k) for k in rows.all()]


def get_knowledge(knowledge_id: int | None) -> dict | None:
    """Liefert den Bereich zur ID; ``None`` als ID steht fuer den Standard."""
    with get_db_service().session_scope(commit=False) as db:
        q = db.query(Knowledge)
        k = (
            q.filter_by(is_default=True).first()
            if knowledge_id is None
            else q.filter_by(id=knowledge_id).first()
        )
        return _to_dict(k) if k else None


_COLLECTION_PREFIX = {"local": "kb", "external": "ext", "sql": "sql"}


def _embedding_fields(spec, dimension: int | None) -> dict:
    """Spalten, die das Embedding-Modell eines Bereichs beschreiben (Anlegen und Aendern)."""
    return {
        "embedding_model": spec.model,
        "embedding_spec": json.dumps(spec.to_dict()),
        "embedding_dimension": dimension,
    }


def _create(kind: str, name: str, description: str | None, **fields) -> dict:
    """Legt einen Bereich an. Die Collection heisst erst nach Vergabe der ID endgueltig."""
    prefix = _COLLECTION_PREFIX[kind]
    with get_db_service().session_scope() as db:
        k = Knowledge(
            name=name.strip(),
            description=(description or "").strip(),
            collection_name=f"{prefix}_tmp_{os.urandom(6).hex()}",
            is_default=False,
            is_public=False,
            kind=kind,
            **fields,
        )
        db.add(k)
        db.flush()
        k.collection_name = f"{prefix}_{k.id}"
        db.commit()
        return _to_dict(k)


def _update(knowledge_id: int, kind: str | None, **fields) -> dict | None:
    """Setzt Spalten eines Bereichs (``kind``: nur diese Art; ``None``: jede)."""
    with get_db_service().session_scope() as db:
        q = db.query(Knowledge).filter_by(id=knowledge_id)
        k = (q.filter_by(kind=kind) if kind else q).first()
        if not k:
            return None
        for column, value in fields.items():
            setattr(k, column, value)
        db.commit()
        return _to_dict(k)


def create_external_knowledge(
    name: str, description: str, spec, dimension: int, source_config: dict
) -> dict:
    """Legt ein Knowledge fuer einen fremden Vektorspeicher an (kein eigener Speicherplatz)."""
    from app.rag import model_settings

    return _create(
        "external",
        name,
        description,
        source_config=json.dumps(source_config),
        index_signature=model_settings.current_signature(spec),
        **_embedding_fields(spec, dimension),
    )


def update_external_source(
    knowledge_id: int, spec, dimension: int, source_config: dict
) -> dict | None:
    return _update(
        knowledge_id,
        "external",
        source_config=json.dumps(source_config),
        **_embedding_fields(spec, dimension),
    )


def is_external(knowledge: dict | None) -> bool:
    return bool(knowledge) and knowledge.get("kind") == "external"


def create_knowledge(
    name: str, description: str = "", dimension: int | None = None
) -> dict:
    """Legt einen Bereich an und uebernimmt die globale Embedding-Konfiguration.

    ``dimension``: Vektorgroesse des Modells (vom Aufrufer per Probe ermittelt).
    """
    from app.rag import model_settings
    from app.rag import sparse as sparse_lang

    spec = model_settings.embedding_spec()
    return _create(
        "local",
        name,
        description,
        index_signature=model_settings.current_signature(spec),
        bm25_language=sparse_lang.default_language(),
        **_embedding_fields(spec, dimension),
    )


def update_knowledge(
    knowledge_id: int, name: str | None, description: str | None
) -> dict | None:
    fields = {}
    if name is not None and name.strip():
        fields["name"] = name.strip()
    if description is not None:
        fields["description"] = description.strip()
    return _update(knowledge_id, None, **fields)


def delete_knowledge(knowledge_id: int) -> dict | None:
    """Loescht Zuordnungen und Dateien; liefert den geloeschten Bereich zurueck
    (damit der Aufrufer die Qdrant-Collection entfernen kann). Der Standard-Bereich
    ist nicht loeschbar (``None``)."""
    with get_db_service().session_scope() as db:
        k = db.query(Knowledge).filter_by(id=knowledge_id, is_default=False).first()
        if not k:
            return None
        snapshot = _to_dict(k)
        knowledge_uuid = k.uuid
        db.query(UserDocument).filter_by(knowledge_id=k.id).delete()
        db.query(SqlValue).filter_by(knowledge_id=k.id).delete()
        db.query(SqlValueIndex).filter_by(knowledge_id=k.id).delete()
        db.delete(k)
        db.commit()

    from app.auth.access import delete_grants_for_resource

    delete_grants_for_resource("knowledge", snapshot["id"])

    paths = knowledge_paths(snapshot)
    for folder in (paths.files_dir.parent, paths.data_dir, paths.markdown_dir):
        shutil.rmtree(folder, ignore_errors=True)
    # Crypto-Shredding: ohne Schluessel sind auch Kopien in Backups unlesbar
    from app.core import keys

    keys.shred(keys.KNOWLEDGE, knowledge_uuid)
    keys.forget(keys.KNOWLEDGE, snapshot["id"])
    return snapshot


def swap_collection(
    knowledge_id: int, collection_name: str, spec, dimension: int
) -> str | None:
    """Stellt ein Knowledge in EINER Transaktion auf die neue Collection um (Neu-Indexieren).

    Liefert den Namen der bisherigen Collection (zum Loeschen) oder ``None``, wenn das Knowledge
    fehlt.
    """
    from app.rag import model_settings
    from app.rag import sparse as sparse_lang

    with get_db_service().session_scope() as db:
        k = db.query(Knowledge).filter_by(id=knowledge_id).first()
        if not k:
            return None
        old = k.collection_name
        k.collection_name = collection_name
        for column, value in _embedding_fields(spec, dimension).items():
            setattr(k, column, value)
        k.index_signature = model_settings.current_signature(spec)
        k.bm25_language = sparse_lang.default_language()
        k.last_reindex_error = None
        db.commit()
        return old


def set_reindex_error(knowledge_id: int, message: str | None) -> None:
    with get_db_service().session_scope(commit=False) as db:
        k = db.query(Knowledge).filter_by(id=knowledge_id).first()
        if k:
            k.last_reindex_error = message
            db.commit()


def create_sql_knowledge(name: str, description: str, source_config: dict) -> dict:
    """Legt ein Knowledge fuer eine SQL-Datenbank an (keine Collection, keine Embeddings)."""
    return _create("sql", name, description, source_config=json.dumps(source_config))


def update_sql_source(knowledge_id: int, source_config: dict) -> dict | None:
    return _update(knowledge_id, "sql", source_config=json.dumps(source_config))


def is_sql(knowledge: dict | None) -> bool:
    return bool(knowledge) and knowledge.get("kind") == "sql"


def is_remote(knowledge: dict | None) -> bool:
    """Quelle ausserhalb: kein eigener Speicher, keine lokale Collection, nur lesend."""
    return bool(knowledge) and knowledge.get("kind") in ("external", "sql")
