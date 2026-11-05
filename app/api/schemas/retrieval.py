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

"""Modelle fuer ``app/api/retrieval.py``.

Die Profil-Konfiguration (``config``) prueft ``app.rag.profiles.validate_config`` (mit eigenen
Meldungen); hier bleibt sie ein offenes Objekt.
"""

from typing import Any

from pydantic import BaseModel

from app.api.schemas.common import Loose


class Profile(Loose):
    id: int
    name: str
    description: str = ""
    config: dict[str, Any]


class ProfileCreate(BaseModel):
    name: str = ""
    description: str = ""
    config: dict[str, Any] | None = None


class ProfileUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    config: dict[str, Any] | None = None


class Defaults(BaseModel):
    config: dict[str, Any]
    prompt: str


class ProfileAssign(BaseModel):
    profile_id: Any = None  # int oder null; geprueft im Handler (Meldung "Ungültige Profil-ID")


class ProfileAssigned(BaseModel):
    success: bool
    profile_id: int | None


class RetrievalTestRequest(BaseModel):
    question: str | None = None
    knowledge_ids: list[int] | None = None
    profile_id: int | None = None
    config: dict[str, Any] | None = None
    model: str | None = None


class TestDocument(BaseModel):
    filename: str | None = None
    score: float
    chunk_index: int | None = None
    chunks: Any = None
    text: str


class TestResult(BaseModel):
    knowledge_id: int
    knowledge: str
    profile_id: int | None = None
    documents: list[TestDocument]
    trace: dict[str, Any]


class RetrievalTestResponse(BaseModel):
    results: list[TestResult]
    helper_model: str | None = None
