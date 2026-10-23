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
Windows Entry Point
- Waitress auf 127.0.0.1:5000 (keine Firewall-Abfrage)
- Qdrant.exe auf 127.0.0.1:6333 (keine Firewall-Abfrage)

Usage:
    python run_windows.py              # Auto-Start Qdrant + Flask
    python run_windows.py --no-qdrant # Nur Flask (Qdrant manuell/docker)
"""

import atexit
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_qdrant_process = None

QDRANT_DOWNLOAD_URL = "https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-pc-windows-msvc.exe"


def download_qdrant():
    """Download qdrant.exe wenn nicht vorhanden"""
    qdrant_exe = Path("qdrant.exe")

    if qdrant_exe.exists():
        return True

    print("📥 qdrant.exe nicht gefunden. Download wird gestartet...")
    print(f"   URL: {QDRANT_DOWNLOAD_URL}")
    print("   (Dies kann einen Moment dauern...)")

    try:
        urllib.request.urlretrieve(QDRANT_DOWNLOAD_URL, str(qdrant_exe))
        print(f"✅ Download complete: {qdrant_exe.absolute()}")
        return True
    except Exception as e:
        print(f"❌ Download fehlgeschlagen: {e}")
        print("   Bitte manuell downloaden von:")
        print(f"   {QDRANT_DOWNLOAD_URL}")
        return False


def start_qdrant():
    """Startet qdrant.exe auf localhost (keine Firewall-Abfrage)"""
    qdrant_exe = Path("qdrant.exe")

    if not qdrant_exe.exists() and not download_qdrant():
        return None

    try:
        # WICHTIG: --uri für localhost-only (keine Firewall-Abfrage)
        process = subprocess.Popen(
            [str(qdrant_exe), "--uri", "http://127.0.0.1:6333"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        print(f"✅ Qdrant gestartet (localhost:6333, PID: {process.pid})")
        time.sleep(3)  # Warte auf Start
        return process
    except Exception as e:
        print(f"❌ Fehler beim Starten von Qdrant: {e}")
        return None


def cleanup():
    """Beendet Qdrant beim Shutdown"""
    global _qdrant_process
    if _qdrant_process:
        print("\n🛑 Beende Qdrant...")
        _qdrant_process.terminate()
        _qdrant_process = None


def main():
    global _qdrant_process

    args = sys.argv[1:]
    no_qdrant = "--no-qdrant" in args

    print("=" * 50)
    print("Open Knowledge UI - Windows Mode")
    print("=" * 50)
    print()

    if no_qdrant:
        print("   Flask: http://127.0.0.1:5000")
        print("   Qdrant: Manuell/Docker (nicht automatisch)")
    else:
        print("   Flask: http://127.0.0.1:5000")
        print("   Qdrant: http://127.0.0.1:6333")
        print("   (Nur localhost - keine Firewall-Abfrage!)")
    print()

    # Qdrant starten (außer --no-qdrant)
    if not no_qdrant:
        _qdrant_process = start_qdrant()
        if _qdrant_process:
            atexit.register(cleanup)
            signal.signal(signal.SIGTERM, lambda s, f: cleanup())
            signal.signal(signal.SIGINT, lambda s, f: cleanup())
        else:
            print("WARNUNG: Qdrant konnte nicht gestartet werden.")
            print("   Starte trotzdem Flask...")
            print()

    # Migration
    print("Alembic Migration...")
    result = subprocess.run(
        ["alembic", "upgrade", "head"], capture_output=True, text=True
    )
    if result.returncode == 0:
        print("Migration erfolgreich")
    else:
        print("Migration Fehler (kann ignoriert werden wenn bereits aktuell):")
        print(result.stderr)
    print()

    # Waitress starten (nur localhost!)
    from waitress import serve

    # Import app nach Alembic (wegen DB-Initialisierung)
    from app import create_app

    app = create_app()

    print("=" * 50)
    print("Server läuft!")
    print("   http://127.0.0.1:5000")
    print("=" * 50)
    print("   (Nur auf diesem PC erreichbar)")
    print("   Drücken Sie Ctrl+C zum Beenden")
    print()

    serve(app, host="127.0.0.1", port=5000, threads=4)


if __name__ == "__main__":
    main()
