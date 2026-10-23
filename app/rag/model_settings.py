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
Aktive Modelle: welches Embedding- und Reranking-Modell gilt gerade (Settings) und mit
welchem Modell ein Knowledge indexiert wurde.

- ``EMBEDDING_SPEC`` / ``RERANKER_SPEC`` (JSON, siehe ``ModelSpec``) sind global.
- Jedes Knowledge merkt sich ``embedding_spec`` (womit seine Vektoren entstanden) und
  ``index_signature``. Abfragen und neue Dokumente nutzen immer das Modell des Knowledge,
  nie blind das globale; so bleibt die Suche waehrend einer Umstellung korrekt.
"""

import json
from logging import getLogger

from app.core.database_service import get_db_service
from app.core.models import Knowledge, Setting
from app.core.settings_service import SettingsService
from app.rag import models_runtime as mr

logger = getLogger(__name__)

LEGACY_FASTEMBED = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Aenderungen am Zerlegen/Parsen machen bestehende Indizes "veraltet"
SPLITTER_VERSION_KEY = "splitter"


def _setting(key: str, default: str | None = None) -> str | None:
    db = get_db_service().get_session()
    try:
        return SettingsService(db).get(key, default)
    finally:
        db.close()


def _store(key: str, value: str) -> None:
    db = get_db_service().get_session()
    try:
        SettingsService(db).set(key, value)
    finally:
        db.close()


def ensure_model_settings(db, fresh: bool) -> None:
    """Legt EMBEDDING_SPEC/RERANKER_SPEC an, falls sie fehlen (idempotent).

    Frische Installation: HuggingFace-Modelle. Bestehende Installation (Vektoren stammen von
    fastembed): dieselben Modelle ueber fastembed, damit nichts neu indexiert werden muss.
    """
    settings = SettingsService(db)
    existing = {
        s.key
        for s in db.query(Setting).filter(
            Setting.key.in_(["EMBEDDING_SPEC", "RERANKER_SPEC"])
        )
    }
    if "EMBEDDING_SPEC" not in existing:
        if fresh:
            spec = mr.DEFAULT_EMBEDDING
        else:
            spec = {
                "source": "fastembed",
                "model": settings.get("EMBEDDING_MODEL") or LEGACY_FASTEMBED,
            }
        settings.set("EMBEDDING_SPEC", json.dumps(spec))
    if "RERANKER_SPEC" not in existing:
        spec = (
            mr.DEFAULT_RERANKER
            if fresh
            else {"source": "fastembed", "model": mr.DEFAULT_RERANKER["model"]}
        )
        settings.set("RERANKER_SPEC", json.dumps(spec))


def _spec(key: str, fallback: dict) -> mr.ModelSpec:
    raw = _setting(key)
    try:
        return mr.ModelSpec.from_dict(raw) if raw else mr.ModelSpec.from_dict(fallback)
    except (ValueError, TypeError):
        logger.warning(f"{key} ist ungueltig, Standard wird verwendet")
        return mr.ModelSpec.from_dict(fallback)


def embedding_spec() -> mr.ModelSpec:
    return _spec("EMBEDDING_SPEC", {"source": "fastembed", "model": LEGACY_FASTEMBED})


def reranker_spec() -> mr.ModelSpec:
    return _spec("RERANKER_SPEC", mr.DEFAULT_RERANKER)


def save_specs(
    embedding: mr.ModelSpec | None = None, reranker: mr.ModelSpec | None = None
) -> None:
    if embedding:
        _store("EMBEDDING_SPEC", json.dumps(embedding.to_dict()))
    if reranker:
        _store("RERANKER_SPEC", json.dumps(reranker.to_dict()))


def ingest_params() -> tuple[int, int]:
    db = get_db_service().get_session()
    try:
        s = SettingsService(db)
        return s.get_int("INGEST_CHUNK_SIZE", 1000), s.get_int(
            "INGEST_CHUNK_OVERLAP", 150
        )
    finally:
        db.close()


def current_signature(spec: mr.ModelSpec | dict | None = None) -> str:
    from app.ingest.indexer import PARSER_VERSION
    from app.ingest.splitter import SPLITTER_VERSION

    size, overlap = ingest_params()
    return mr.index_signature(
        spec or embedding_spec(), size, overlap, SPLITTER_VERSION, PARSER_VERSION
    )


def knowledge_spec(knowledge: dict) -> mr.ModelSpec:
    """Modell, mit dem die Vektoren dieses Knowledge entstanden sind."""
    raw = knowledge.get("embedding_spec")
    if raw:
        try:
            return mr.ModelSpec.from_dict(raw)
        except (ValueError, TypeError):
            pass
    model = knowledge.get("embedding_model") or LEGACY_FASTEMBED
    return mr.ModelSpec("fastembed", model)


def is_stale(knowledge: dict) -> bool:
    """Index weicht von der aktuellen Konfiguration ab (Neu indexieren empfohlen).

    Ohne gespeicherte Signatur (Altbestand) zaehlt nur ein anderes Modell.
    """
    if knowledge.get("kind") in ("external", "sql"):
        return False  # fremde Quelle: ihr Modell ist unabhaengig vom globalen
    current = embedding_spec()
    if knowledge_spec(knowledge).key != current.key:
        return True
    from app.rag import sparse as sparse_lang

    if sparse_lang.knowledge_language(knowledge) != sparse_lang.default_language():
        return True  # Stichwortsuche in anderer Sprache indexiert: Neu indexieren verbessert die Treffer
    sig = knowledge.get("index_signature")
    return bool(sig) and sig != current_signature(current)


def embedder_for(knowledge: dict):
    return mr.get_embedder(knowledge_spec(knowledge))


def rerank_score(_model: str, query: str, texts: list[str]) -> list[float]:
    """``rerank_fn`` der Pipeline: das Profil kennt kein eigenes Modell mehr (global)."""
    return mr.get_reranker(reranker_spec()).score(query, texts)


def stamp_knowledge(knowledge_id: int, spec: mr.ModelSpec, dimension: int) -> None:
    """Setzt Modell, Dimension und Signatur eines Knowledge (nach Anlage bzw. Neu-Indexieren)."""
    db = get_db_service().get_session()
    try:
        k = db.query(Knowledge).filter_by(id=knowledge_id).first()
        if k:
            k.embedding_spec = json.dumps(spec.to_dict())
            k.embedding_model = spec.model
            k.embedding_dimension = dimension
            k.index_signature = current_signature(spec)
            k.last_reindex_error = None
            db.commit()
    finally:
        db.close()


def set_dimension(knowledge_id: int, dimension: int) -> None:
    db = get_db_service().get_session()
    try:
        k = db.query(Knowledge).filter_by(id=knowledge_id).first()
        if k and not k.embedding_dimension:
            k.embedding_dimension = dimension
            db.commit()
    finally:
        db.close()


def prepare(client) -> None:
    """Laedt die aktiven Modelle und legt fehlende Collections an (Hintergrund beim Start).

    Fehler (z. B. kein Netz fuer den Download) werden im Modell-Status festgehalten und
    verhindern den Start nicht; sie erscheinen auf der Admin-Seite "Einlesen & Modelle".
    """
    from app.rag.knowledge import list_knowledge
    from app.vector.collections import ensure_collection

    for kb in list_knowledge():
        if kb["embedding_dimension"] or kb.get("kind") in ("external", "sql"):
            continue
        try:
            dim = mr.get_embedder(knowledge_spec(kb)).dimension
            set_dimension(kb["id"], dim)
            ensure_collection(client, kb["collection_name"], dim)
        except mr.ModelLoadError as e:
            logger.warning(f"Collection von '{kb['name']}' noch nicht angelegt: {e}")
    try:
        spec = embedding_spec()
        ensure_collection(
            client, mr.uploads_collection(spec), mr.get_embedder(spec).dimension
        )
    except mr.ModelLoadError as e:
        logger.warning(f"Embedding-Modell nicht verfuegbar: {e}")
    try:
        mr.get_reranker(reranker_spec())
    except mr.ModelLoadError as e:
        logger.warning(f"Reranking-Modell nicht verfuegbar: {e}")
