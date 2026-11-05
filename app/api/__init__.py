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

"""HTTP-Routen: ein ``APIRouter`` je Fachbereich."""

import importlib

ROUTERS = (
    "pages",
    "chat",
    "auth",
    "users",
    "knowledge",
    "documents",
    "summaries",
    "models",
    "admin",
    "retrieval",
)


def include_routers(app, only: tuple[str, ...] | None = None) -> None:
    """Alle Router einbinden (Import erst hier, damit ``import app.api`` nichts nachlaedt)."""
    for name in only or ROUTERS:
        module = importlib.import_module(f"app.api.{name}")
        app.include_router(module.router)
