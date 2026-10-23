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

logger = getLogger(__name__)

DEFAULT_COLLECTION = "open_knowledge_ui"
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
    db = get_db_service().get_session()
    try:
        return Path(SettingsService(db).get("DATA_DIR", "./data") or "./data")
    finally:
        db.close()


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
    db = get_db_service().get_session()
    try:
        rows = db.query(Knowledge).order_by(Knowledge.is_default.desc(), Knowledge.id)
        return [_to_dict(k) for k in rows.all()]
    finally:
        db.close()


def get_knowledge(knowledge_id: int | None) -> dict | None:
    """Liefert den Bereich zur ID; ``None`` als ID steht fuer den Standard."""
    db = get_db_service().get_session()
    try:
        q = db.query(Knowledge)
        k = (
            q.filter_by(is_default=True).first()
            if knowledge_id is None
            else q.filter_by(id=knowledge_id).first()
        )
        return _to_dict(k) if k else None
    finally:
        db.close()


def create_external_knowledge(
    name: str, description: str, spec, dimension: int, source_config: dict
) -> dict:
    """Legt ein Knowledge fuer einen fremden Vektorspeicher an (kein eigener Speicherplatz)."""
    from app.rag import model_settings

    db = get_db_service().get_session()
    try:
        k = Knowledge(
            name=name.strip(),
            description=(description or "").strip(),
            collection_name=f"ext_tmp_{os.urandom(6).hex()}",
            is_default=False,
            is_public=False,
            kind="external",
            source_config=json.dumps(source_config),
            embedding_model=spec.model,
            embedding_spec=json.dumps(spec.to_dict()),
            embedding_dimension=dimension,
            index_signature=model_settings.current_signature(spec),
        )
        db.add(k)
        db.flush()
        k.collection_name = f"ext_{k.id}"
        db.commit()
        return _to_dict(k)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def update_external_source(
    knowledge_id: int, spec, dimension: int, source_config: dict
) -> dict | None:
    db = get_db_service().get_session()
    try:
        k = db.query(Knowledge).filter_by(id=knowledge_id, kind="external").first()
        if not k:
            return None
        k.source_config = json.dumps(source_config)
        k.embedding_model = spec.model
        k.embedding_spec = json.dumps(spec.to_dict())
        k.embedding_dimension = dimension
        db.commit()
        return _to_dict(k)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def is_external(knowledge: dict | None) -> bool:
    return bool(knowledge) and knowledge.get("kind") == "external"


def create_knowledge(
    name: str, description: str = "", dimension: int | None = None
) -> dict:
    """Legt einen Bereich an und uebernimmt die globale Embedding-Konfiguration.

    ``dimension``: Vektorgroesse des Modells (vom Aufrufer per Probe ermittelt).
    """
    db = get_db_service().get_session()
    try:
        from app.rag import model_settings
        from app.rag import sparse as sparse_lang

        spec = model_settings.embedding_spec()
        k = Knowledge(
            name=name.strip(),
            description=(description or "").strip(),
            # vorlaeufig eindeutig; echter Name nach Vergabe der ID
            collection_name=f"kb_tmp_{os.urandom(6).hex()}",
            is_default=False,
            is_public=False,
            embedding_model=spec.model,
            embedding_spec=json.dumps(spec.to_dict()),
            embedding_dimension=dimension,
            index_signature=model_settings.current_signature(spec),
            bm25_language=sparse_lang.default_language(),
        )
        db.add(k)
        db.flush()
        k.collection_name = f"kb_{k.id}"
        db.commit()
        return _to_dict(k)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def update_knowledge(
    knowledge_id: int, name: str | None, description: str | None
) -> dict | None:
    db = get_db_service().get_session()
    try:
        k = db.query(Knowledge).filter_by(id=knowledge_id).first()
        if not k:
            return None
        if name is not None and name.strip():
            k.name = name.strip()
        if description is not None:
            k.description = description.strip()
        db.commit()
        return _to_dict(k)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_knowledge(knowledge_id: int) -> dict | None:
    """Loescht Zuordnungen und Dateien; liefert den geloeschten Bereich zurueck
    (damit der Aufrufer die Qdrant-Collection entfernen kann). Der Standard-Bereich
    ist nicht loeschbar (``None``)."""
    db = get_db_service().get_session()
    try:
        k = db.query(Knowledge).filter_by(id=knowledge_id, is_default=False).first()
        if not k:
            return None
        snapshot = _to_dict(k)
        db.query(UserDocument).filter_by(knowledge_id=k.id).delete()
        db.query(SqlValue).filter_by(knowledge_id=k.id).delete()
        db.query(SqlValueIndex).filter_by(knowledge_id=k.id).delete()
        db.delete(k)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    from app.auth.access import delete_grants_for_resource

    delete_grants_for_resource("knowledge", snapshot["id"])

    paths = knowledge_paths(snapshot)
    for folder in (paths.files_dir.parent, paths.data_dir, paths.markdown_dir):
        shutil.rmtree(folder, ignore_errors=True)
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

    db = get_db_service().get_session()
    try:
        k = db.query(Knowledge).filter_by(id=knowledge_id).first()
        if not k:
            return None
        old = k.collection_name
        k.collection_name = collection_name
        k.embedding_spec = json.dumps(spec.to_dict())
        k.embedding_model = spec.model
        k.embedding_dimension = dimension
        k.index_signature = model_settings.current_signature(spec)
        k.bm25_language = sparse_lang.default_language()
        k.last_reindex_error = None
        db.commit()
        return old
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def set_reindex_error(knowledge_id: int, message: str | None) -> None:
    db = get_db_service().get_session()
    try:
        k = db.query(Knowledge).filter_by(id=knowledge_id).first()
        if k:
            k.last_reindex_error = message
            db.commit()
    finally:
        db.close()


def create_sql_knowledge(name: str, description: str, source_config: dict) -> dict:
    """Legt ein Knowledge fuer eine SQL-Datenbank an (keine Collection, keine Embeddings)."""
    db = get_db_service().get_session()
    try:
        k = Knowledge(
            name=name.strip(),
            description=(description or "").strip(),
            collection_name=f"sql_tmp_{os.urandom(6).hex()}",
            is_default=False,
            is_public=False,
            kind="sql",
            source_config=json.dumps(source_config),
        )
        db.add(k)
        db.flush()
        k.collection_name = f"sql_{k.id}"
        db.commit()
        return _to_dict(k)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def update_sql_source(knowledge_id: int, source_config: dict) -> dict | None:
    db = get_db_service().get_session()
    try:
        k = db.query(Knowledge).filter_by(id=knowledge_id, kind="sql").first()
        if not k:
            return None
        k.source_config = json.dumps(source_config)
        db.commit()
        return _to_dict(k)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def is_sql(knowledge: dict | None) -> bool:
    return bool(knowledge) and knowledge.get("kind") == "sql"


def is_remote(knowledge: dict | None) -> bool:
    """Quelle ausserhalb: kein eigener Speicher, keine lokale Collection, nur lesend."""
    return bool(knowledge) and knowledge.get("kind") in ("external", "sql")
