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

"""Modelle fuer ``app/api/documents.py``. ``knowledge_id`` waehlt den Bereich (sonst der Standard)."""

from typing import Any

from pydantic import BaseModel


class IngestionStatus(BaseModel):
    is_ingesting: bool
    knowledge_id: int | None = None


class KnowledgeBody(BaseModel):
    knowledge_id: int | str | None = None


class DocumentRow(BaseModel):
    id: int
    document_id: str
    filename: str
    uploaded_at: str | None = None
    content_preview: str
    source: Any = None
    has_data_file: bool
    has_markdown: bool


class MarkdownView(BaseModel):
    filename: str
    markdown: str


class MyUpload(BaseModel):
    kind: str
    id: int | str
    filename: str
    source: str
    created_at: str | None = None
    size: int | None = None
    has_markdown: bool


class UploadResult(BaseModel):
    success: bool
    documents: list[str]
    data_files: list[str]
    skipped: list[Any]
    files: list[str]
    message: str


class Cleared(BaseModel):
    success: bool
    deleted: int
    errors: list[str]


class DataFile(BaseModel):
    filename: str
    size: int
    size_human: str
    modified: float


class DataPreview(BaseModel):
    filename: str
    columns: list[str]
    rows: list[list[str]]
    total_rows: int
    truncated: bool


class IngestResult(BaseModel):
    success: bool
    message: str
    processed: int
    errors: list[Any]
    warnings: list[Any]


class VectorRow(BaseModel):
    id: str
    filename: Any = None
    source: Any = None
    content_preview: str
    has_data_file: bool


class VectorList(BaseModel):
    documents: list[VectorRow]
    next_offset: Any = None
    filenames: list[Any]
    limit: int


class VectorDetail(BaseModel):
    id: str
    page_content: Any = None
    original_content: Any = None
    metadata: dict[str, Any]


class BatchDelete(KnowledgeBody):
    filename: str | None = None


class ClearAll(KnowledgeBody):
    confirmation: str = ""
