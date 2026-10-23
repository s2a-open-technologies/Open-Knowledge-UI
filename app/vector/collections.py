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

"""Anlage von Qdrant-Collections (Dense + BM25-Sparse, Hybrid-Suche)."""

from logging import getLogger

from qdrant_client import models as qdrant_models

logger = getLogger(__name__)

USER_UPLOADS_COLLECTION = "user_uploads"

PAYLOAD_INDEXES = {
    "file_group_id": qdrant_models.PayloadSchemaType.KEYWORD,
    "session_id": qdrant_models.PayloadSchemaType.KEYWORD,
    "chunk_index": qdrant_models.PayloadSchemaType.INTEGER,
}


# HNSW-Index der Dense-Vektoren (schnelle Naeherungssuche statt Vollscan).
HNSW_CONFIG = {"m": 16, "ef_construct": 100, "full_scan_threshold": 10000}

# Name des BM25-Vektors (so heisst er bei langchain-qdrant/FastEmbedSparse).
SPARSE_VECTOR = "langchain-sparse"


def _sparse_params() -> qdrant_models.SparseVectorParams:
    # fastembed liefert fuer BM25 auf der Query-Seite nur Einsen; die IDF-Gewichtung
    # (seltene Woerter zaehlen mehr) muss Qdrant uebernehmen. Ohne den Modifier ist die
    # Stichwortsuche deutlich schwaecher (Messung: Treffer@1 6/13 statt 10/13).
    return qdrant_models.SparseVectorParams(modifier=qdrant_models.Modifier.IDF)


def ensure_collection(client, collection_name: str, dimensions: int) -> None:
    """Legt die Collection samt Payload-Indizes an, falls sie fehlt (idempotent).

    Jede Collection hat einen Dense-Vektor mit HNSW-Index und einen BM25-Sparse-Vektor
    mit IDF-Modifier. Bestehenden Collections ohne Modifier wird er nachgetragen.
    """
    if not client.collection_exists(collection_name):
        logger.info(f"Erstelle Qdrant Collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=int(dimensions), distance=qdrant_models.Distance.COSINE
            ),
            sparse_vectors_config={SPARSE_VECTOR: _sparse_params()},
            hnsw_config=qdrant_models.HnswConfigDiff(**HNSW_CONFIG),
        )
    else:
        _ensure_idf(client, collection_name)

    # Payload-Indizes: file_group_id fuer Liste/Loeschen pro Datei, session_id fuer
    # die Chat-Anhaenge, chunk_index fuer die Kontext-Erweiterung. Idempotent.
    for field, schema in PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=schema,
            )
        except Exception as e:
            logger.debug(f"Payload-Index {field} bereits vorhanden oder Fehler: {e}")


def _ensure_idf(client, collection_name: str) -> None:
    """Traegt den IDF-Modifier an Collections nach, die vor dieser Aenderung entstanden."""
    try:
        sparse = (
            client.get_collection(collection_name).config.params.sparse_vectors or {}
        )
        current = sparse.get(SPARSE_VECTOR)
        if current is not None and current.modifier != qdrant_models.Modifier.IDF:
            client.update_collection(
                collection_name=collection_name,
                sparse_vectors_config={SPARSE_VECTOR: _sparse_params()},
            )
            logger.info(f"IDF-Modifier fuer BM25 in '{collection_name}' nachgetragen")
    except Exception as e:
        logger.warning(f"IDF-Modifier fuer '{collection_name}' nicht setzbar: {e}")
