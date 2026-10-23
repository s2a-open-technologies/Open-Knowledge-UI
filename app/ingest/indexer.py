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
Vom Markdown zu Qdrant-Punkten.

Gemeinsamer Weg fuer das Einlesen neuer Dateien (``DocumentIngestion``) und das
Neu-Indexieren aus gespeichertem Markdown (``app/reindex.py``):

    Markdown -> Chunks (Ueberschriften-Pfad) -> [Kontext-Anreicherung] -> Embedding
    (Dense + BM25) -> Punkte

Aenderungen, die vorhandene Indizes veralten lassen (Splitter, Payload-Aufbau),
erhoehen ``SPLITTER_VERSION`` bzw. ``PARSER_VERSION``.
"""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from logging import getLogger
from pathlib import Path

from qdrant_client import models as qdrant_models

from app.ingest.splitter import embedding_text, split_markdown

logger = getLogger(__name__)

# Version der Markdown-Erzeugung (Docling-Export + Bildbeschreibungen).
PARSER_VERSION = 1

DENSE_VECTOR = ""
SPARSE_VECTOR = "langchain-sparse"
UPSERT_BATCH = 100


def document_metadata(
    *,
    filename: str,
    knowledge_id: int | None = None,
    uploaded_by: int | None = None,
    uploaded_at: datetime | None = None,
) -> dict:
    """Metadaten, die jeder Chunk eines Dokuments zusaetzlich traegt (Nachvollziehbarkeit, Filter)."""
    from app.ingest.splitter import SPLITTER_VERSION

    meta = {
        "file_type": Path(filename).suffix.lower().lstrip("."),
        "parser_version": PARSER_VERSION,
        "splitter_version": SPLITTER_VERSION,
        "uploaded_at": (uploaded_at or datetime.now(UTC)).isoformat(),
    }
    if knowledge_id is not None:
        meta["knowledge_id"] = knowledge_id
    if uploaded_by is not None:
        meta["uploaded_by"] = uploaded_by
    return meta


def markdown_file(markdown_dir: Path | str, file_group_id: str) -> Path:
    return Path(markdown_dir) / f"{file_group_id}.md"


def save_markdown(markdown_dir: Path | str, file_group_id: str, markdown: str) -> Path:
    """Speichert das Markdown eines Dokuments (UTF-8) und liefert den Pfad."""
    path = markdown_file(markdown_dir, file_group_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def build_points(
    markdown: str,
    *,
    filename: str,
    source: str,
    file_group_id: str,
    dense,
    sparse,
    chunk_size: int,
    chunk_overlap: int,
    extra_metadata: dict | None = None,
    enrich: Callable[[str, str], str] | None = None,
    point_ids: list[str] | None = None,
) -> list[qdrant_models.PointStruct]:
    """Teilt das Markdown, berechnet Dense- und BM25-Vektoren und baut die Punkte.

    Args:
        dense / sparse: Objekte mit ``embed_documents(texts)``.
        enrich: optional ``enrich(dokument_auszug, chunk_text) -> angereicherter Text``
            (LLM-Kontextanreicherung); der Rueckgabewert wird eingebettet.
        point_ids: feste IDs (sonst zufaellig), z. B. fuer reproduzierbare Tests.

    Returns:
        Leere Liste, wenn das Markdown keinen Inhalt hat.
    """
    chunks = split_markdown(
        markdown, filename=filename, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    if not chunks:
        return []

    document_excerpt = " ".join(c.text for c in chunks)[:2500]
    texts = []
    for chunk in chunks:
        text = embedding_text(chunk)
        texts.append(enrich(document_excerpt, text) if enrich else text)

    dense_vectors = dense.embed_documents(texts)
    sparse_vectors = sparse.embed_documents(texts)

    points = []
    for i, chunk in enumerate(chunks):
        sv = sparse_vectors[i]
        points.append(
            qdrant_models.PointStruct(
                id=point_ids[i] if point_ids else str(uuid.uuid4()),
                vector={
                    DENSE_VECTOR: list(dense_vectors[i]),
                    SPARSE_VECTOR: qdrant_models.SparseVector(
                        indices=sv.indices if hasattr(sv, "indices") else sv["indices"],
                        values=sv.values if hasattr(sv, "values") else sv["values"],
                    ),
                },
                payload={
                    "page_content": texts[i],
                    "original_content": chunk.text,
                    "source": source,
                    "filename": filename,
                    "title": chunk.title,
                    "headers": chunk.headers,
                    "type": "document_chunk",
                    "file_group_id": file_group_id,
                    # Reihenfolge in der Datei: Grundlage der Nachbar-Chunks im Retrieval.
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    # Seiten der Fundstelle, nur wenn der Parser sie lieferte (PDF)
                    **(
                        {"page_start": chunk.page_start, "page_end": chunk.page_end}
                        if chunk.page_start is not None
                        else {}
                    ),
                    **(extra_metadata or {}),
                },
            )
        )
    return points


def upsert_points(
    client, collection_name: str, points: list[qdrant_models.PointStruct]
) -> None:
    for i in range(0, len(points), UPSERT_BATCH):
        client.upsert(
            collection_name=collection_name, points=points[i : i + UPSERT_BATCH]
        )
