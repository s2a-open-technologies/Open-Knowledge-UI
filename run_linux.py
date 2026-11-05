#!/usr/bin/env python3

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
Linux Entry Point
- Gunicorn auf 0.0.0.0:5000 (Docker)
- Qdrant separat (docker-compose)
- Alembic Migration automatisch

Usage:
    python run_linux.py              # Normaler Start (mit Alembic)
"""

import os
import subprocess


def main():
    print("=" * 50)
    print("🐧 Open Knowledge UI - Linux Mode")
    print("=" * 50)
    print()
    print("   Gunicorn: http://0.0.0.0:5000")
    print("   Qdrant: Separat (docker-compose)")
    print()

    # Alembic Migration (immer ausführen)
    print("📊 Alembic Migration...")
    result = subprocess.run(
        ["alembic", "upgrade", "head"], capture_output=True, text=True
    )
    if result.returncode == 0:
        print("✅ Migration erfolgreich")
    else:
        print("⚠️  Migration Fehler:")
        print(result.stderr)
    print()

    print("=" * 50)
    print("🚀 Starte Gunicorn...")
    print("=" * 50)
    print()

    # Uvicorn starten (ASGI). Ein Worker-Prozess, damit In-Memory-State (Qdrant-Client,
    # Ressourcen-Init-Status, Ingestion-Status) konsistent bleibt; Chat-Streams laufen als
    # asyncio-Tasks, blockierende Handler im Threadpool. APP_PRELOAD laedt die Modelle vor dem Start.
    import uvicorn

    os.environ.setdefault("APP_PRELOAD", "1")
    uvicorn.run("app:create_app", factory=True, host="0.0.0.0", port=5000, workers=1)


if __name__ == "__main__":
    main()
