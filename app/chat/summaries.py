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
Analysen (Summaries) gehoeren dem Nutzer, der sie erstellt: jeder hat einen eigenen Ordner
``<SUMMARIES_DIR>/u<id>/``. Weder die Oberflaeche noch die Werkzeuge des Modells sehen oder
aendern Dateien anderer Nutzer.
"""

import os
import shutil
from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)


def base_dir() -> Path:
    return Path(os.getenv("SUMMARIES_DIR") or "./data/summaries")


def user_dir(user_id: int | None, create: bool = False) -> Path:
    """Ordner eines Nutzers; ohne Nutzer (Skripte) der gemeinsame Basisordner."""
    path = base_dir() if user_id is None else base_dir() / f"u{int(user_id)}"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def safe_path(user_id: int | None, filename: str) -> Path | None:
    """Pfad einer .md-Datei im Ordner des Nutzers oder ``None`` (Ausbruchsversuch, falsche Endung)."""
    folder = user_dir(user_id).resolve()
    path = (folder / filename).resolve()
    if path.parent != folder or path.suffix != ".md":
        return None
    return path


def adopt_legacy(user_id: int) -> int:
    """Uebergibt Dateien aus der Zeit vor den Nutzerordnern (direkt im Basisordner) einem Admin."""
    base = base_dir()
    if not base.is_dir():
        return 0
    moved = 0
    for file in base.glob("*.md"):
        target = user_dir(user_id, create=True) / file.name
        if not target.exists():
            shutil.move(str(file), str(target))
            moved += 1
    if moved:
        logger.info(
            f"{moved} bestehende Analysen in den Ordner von Nutzer {user_id} verschoben"
        )
    return moved
