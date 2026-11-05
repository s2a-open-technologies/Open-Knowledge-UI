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

"""Modelle fuer ``app/api/knowledge.py``.

Quell-Konfigurationen (``source``, ``embedding``, ``identity``) bleiben offene Objekte: Sie werden
von ``app.knowledge.external.parse_config`` bzw. ``app.knowledge.sql.config.parse_config`` geprueft,
die auch die Meldungen liefern.
"""

from typing import Any

from pydantic import BaseModel

from app.api.schemas.common import Loose


class KnowledgeOut(Loose):
    """Ein Knowledge-Bereich mit Zaehlungen; je nach Art zusaetzlich ``source_config`` (ohne Geheimnisse)."""

    id: int
    name: str
    description: str = ""
    collection_name: str
    kind: str = "local"
    is_default: bool = False
    is_public: bool = False
    created_at: str | None = None
    embedding_model: str | None = None
    embedding_spec: str | None = None
    embedding_dimension: int | None = None
    bm25_language: str | None = None
    retrieval_profile_id: int | None = None
    index_signature: str | None = None
    last_reindex_error: str | None = None
    document_count: int
    chunk_count: int | None = None
    can_write: bool
    stale: bool


class KnowledgeCreate(BaseModel):
    name: str | None = None
    description: str = ""
    kind: str | None = None
    source: dict[str, Any] | None = None
    embedding: dict[str, Any] | None = None


class KnowledgeUpdate(BaseModel):
    """Nur gesetzte Felder zaehlen (``source``/``embedding`` aendern die Quelle, nur Admins)."""

    name: str | None = None
    description: str | None = None
    source: dict[str, Any] | None = None
    embedding: dict[str, Any] | None = None


class IndexPayload(BaseModel):
    available: bool
    enabled: bool
    markdown: str | None = None
    updated_at: str | None = None


class SourceTest(BaseModel):
    knowledge_id: int | str | None = None
    source: dict[str, Any] | None = None


class FilterTest(SourceTest):
    identity: dict[str, Any] | None = None


class FilterTestResult(BaseModel):
    success: bool
    blocked: bool
    count: int
    samples: list[str]


class ValuesOverview(Loose):
    """Stand des Wertverzeichnisses einer SQL-Quelle (``valueindex.status`` plus Auswahl und Grenzen)."""

    columns: dict[str, list[str]]
    max_distinct: int
    max_results: int


class ValuesColumns(BaseModel):
    columns: dict[str, list[str]] | None = None


class ValuesSearch(BaseModel):
    term: str | int | float | None = None


class ValueHit(BaseModel):
    table: str
    column: str
    value: str
    count: int
    score: float


class ValuesSearchResult(BaseModel):
    results: list[ValueHit]


class ReindexStarted(BaseModel):
    success: bool = True
    started: bool
    knowledge_ids: list[int] | None = None
    message: str | None = None
