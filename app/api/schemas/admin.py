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

"""Modelle fuer ``app/api/admin.py``.

Die Einstellungs-Bloecke prueft der Handler bzw. der zustaendige Dienst mit eigenen Meldungen
(Grenzen, Sprachen, Prompts); hier sind sie offene Objekte, damit diese Meldungen gleich bleiben.
"""

from typing import Any

from pydantic import BaseModel

from app.api.schemas.common import Loose


class AuditEvents(BaseModel):
    events: list[dict[str, Any]]


class EmbeddingSave(BaseModel):
    """Nur gesetzte Bloecke werden gespeichert."""

    embedding: dict[str, Any] | None = None
    reranker: dict[str, Any] | None = None
    tasks: dict[str, dict[str, Any]] | None = None
    ingest: dict[str, Any] | None = None


class ModelTest(BaseModel):
    kind: str | None = None
    spec: dict[str, Any] | None = None


class ToolsSave(BaseModel):
    stt: dict[str, Any] | None = None
    search: dict[str, Any] | None = None
    pii: dict[str, Any] | None = None


class ToolsTest(BaseModel):
    kind: str | None = None
    values: dict[str, Any] | None = None


class AuthTest(BaseModel):
    kind: str | None = None
    values: dict[str, Any] | None = None
    test_user: str | None = None
    test_password: str | None = None


class SettingsPatch(Loose):
    """Einstellungen eines Bereichs (nur genannte Felder; der Dienst prueft sie)."""
