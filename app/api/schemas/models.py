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

"""Modelle fuer ``app/api/models.py`` (Connections, Modell-Konfigurationen)."""

from typing import Any

from pydantic import BaseModel

from app.api.schemas.common import Loose


class Connection(BaseModel):
    id: int
    name: str
    base_url: str
    has_key: bool
    enabled: bool
    kind: str


class ConnectionCreate(BaseModel):
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    enabled: bool = True
    kind: str | None = None


class ConnectionUpdate(BaseModel):
    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    enabled: bool | None = None
    kind: str | None = None


class ConnectionTest(BaseModel):
    id: int | str | None = None
    base_url: str | None = None
    api_key: str | None = None
    kind: str | None = None


class ModelConfig(Loose):
    id: int
    ref: str
    name: str | None = None
    connection_id: int | None = None
    base_model: str | None = None
    is_clone: bool
    visible: bool
    is_public: bool
    system_prompt: str
    web_search: bool
    vision: bool
    params: dict[str, Any]
    knowledge_ids: list[int] | None = None
    allow_extra_knowledge: bool


class ModelConfigSave(BaseModel):
    """Nur gesetzte Felder werden uebernommen (``app.llm.connections._apply``)."""

    ref: str = ""
    name: str | None = None
    visible: bool | None = None
    is_public: bool | None = None
    web_search: bool | None = None
    vision: bool | None = None
    system_prompt: str | None = None
    params: dict[str, Any] | None = None
    allow_extra_knowledge: bool | None = None
    knowledge_ids: list[int] | None = None


class ModelList(Loose):
    """Modelle der aktiven Connections samt Standardmodell und Compaction-Werten."""

    models: list[dict[str, Any]]
    default: str
    errors: list[Any] = []


class ConfigDeleted(BaseModel):
    success: bool
    deleted: bool


class CloneRequest(BaseModel):
    ref: str = ""
    name: str | None = None


class DefaultModel(BaseModel):
    ref: str = ""


class DefaultSet(BaseModel):
    success: bool
    default: str
