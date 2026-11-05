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

"""Laufzeit-Zustand der App (frueher ``app.extensions``): ein Objekt pro Prozess.

``create_app`` und das Laden der Ressourcen (``app/agent/resources.py``) fuellen es; Router und
Dienste lesen es, ohne einen Request zu brauchen. Tests setzen die Felder direkt.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeState:
    client: Any = None  # Qdrant-Client
    settings: Any = None  # LiveSettingsService
    db: Any = None  # DatabaseService
    sparse_embeddings: Any = None
    llm: Any = None
    agent_tools: list = field(default_factory=list)
    # Geteilter Ingestion-Status fuer den Navbar-Indikator (/api/ingestion-status)
    ingestion_status: dict = field(
        default_factory=lambda: {"is_ingesting": False, "knowledge_id": None}
    )


state = RuntimeState()


def reset() -> None:
    """Zustand leeren (Tests, neuer App-Aufbau)."""
    global state
    fresh = RuntimeState()
    for name in fresh.__dataclass_fields__:
        setattr(state, name, getattr(fresh, name))
