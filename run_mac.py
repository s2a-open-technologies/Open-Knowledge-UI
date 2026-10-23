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

"""
macOS Entry Point (App nativ, Hilfsdienste in Docker)
- Gunicorn auf 127.0.0.1:5000
- Hilfsdienste (Postgres, Qdrant, SearXNG, entityguard, Parakeet, Tool-Worker) in Docker, siehe docker-compose.mac.yml
- Embedding, Reranking und Docling rechnen auf Apple Silicon ueber Metal (MPS);
  ONNX-Modelle nutzen CoreML. Mit RAG_DEVICE=cpu abschaltbar.

Docker auf dem Mac laeuft in einer Linux-VM ohne Zugriff auf die GPU: nur die App laeuft
deshalb nativ, alles andere bleibt in Docker.

Usage:
    python run_mac.py              # Docker-Dienste starten + App
    python run_mac.py --no-docker  # Nur die App (Dienste laufen schon bzw. woanders)
"""

import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

COMPOSE_FILE = "docker-compose.mac.yml"

# Die Hilfsdienste laufen in Docker (Ports auf localhost); die App erreicht sie darueber.
# Gilt nur fuer eine frische Datenbank, danach stehen die Werte in den Einstellungen.
SERVICE_DEFAULTS = {
    "DATABASE_URL": "postgresql+psycopg://rag:rag@localhost:5432/rag",
    "QDRANT_HOST": "localhost",
    "SEARCH_SEARXNG_URL": "http://localhost:8081",
    "PII_FILTER_URL": "http://localhost:8082/api/v1/sanitize",
    "STT_BASEURL": "http://localhost:5001/v1",
}


def ensure_worker_token() -> None:
    """``TOOL_WORKER_TOKEN`` aus der ``.env`` (oder neu erzeugen und dort ablegen). Compose und App brauchen dasselbe Token."""
    env_file = Path(".env")
    text = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    for line in text.splitlines():
        if line.startswith("TOOL_WORKER_TOKEN=") and line.split("=", 1)[1].strip():
            os.environ.setdefault(
                "TOOL_WORKER_TOKEN", line.split("=", 1)[1].strip().strip("\"'")
            )
            return
    token = os.environ.get("TOOL_WORKER_TOKEN") or secrets.token_hex(32)
    with env_file.open("a", encoding="utf-8") as fh:
        fh.write(
            ("" if text.endswith("\n") or not text else "\n")
            + f"TOOL_WORKER_TOKEN={token}\n"
        )
    os.environ["TOOL_WORKER_TOKEN"] = token
    print("🔑 TOOL_WORKER_TOKEN erzeugt und in .env gespeichert")


def point_workers_to_localhost() -> None:
    """Die Worker laufen in Docker mit Port auf localhost; die Standardadressen (Docker-Namen) passen fuer die
    native App nicht. Nur Standardwerte werden ersetzt, Angepasstes bleibt."""
    code = (
        "import os\n"
        "from app.core.database_service import init_db_service\n"
        "from app.core.settings_service import SettingsService\n"
        "svc = init_db_service(os.environ['DATABASE_URL'])\n"
        "s = svc.get_session()\n"
        "st = SettingsService(s)\n"
        "for key, docker, local in (('TOOL_WORKER_URL', 'http://tool-worker:8090', 'http://127.0.0.1:8090'),\n"
        "                           ('SQL_WORKER_URL', 'http://tool-worker-sql:8090', 'http://127.0.0.1:8091')):\n"
        "    if st.get(key) == docker:\n"
        "        st.set(key, local)\n"
        "s.commit()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(
            f"⚠️  Worker-Adressen konnten nicht gesetzt werden:\n{result.stderr[-400:]}"
        )


def wait_for_postgres(timeout: int = 60) -> bool:
    """Wartet, bis die Datenbank Verbindungen annimmt (sonst scheitert die Migration)."""
    for _ in range(timeout):
        ready = subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                COMPOSE_FILE,
                "exec",
                "-T",
                "postgres",
                "pg_isready",
                "-q",
            ],
            capture_output=True,
        )
        if ready.returncode == 0:
            time.sleep(1)  # Qdrant braucht ebenfalls einen Moment
            return True
        time.sleep(1)
    print("❌ Postgres wurde nicht rechtzeitig bereit.")
    return False


def start_services() -> bool:
    print(
        "🐳 Starte Hilfsdienste (Postgres, Qdrant, SearXNG, entityguard, Parakeet, Tool-Worker) in Docker..."
    )
    try:
        result = subprocess.run(["docker", "compose", "-f", COMPOSE_FILE, "up", "-d"])
    except FileNotFoundError:
        print(
            "❌ Docker nicht gefunden. Docker Desktop installieren oder --no-docker nutzen."
        )
        return False
    if result.returncode != 0:
        print("❌ docker compose ist fehlgeschlagen (siehe oben).")
        return False
    return wait_for_postgres()


def main():
    print("=" * 50)
    print(" Open Knowledge UI - macOS Mode")
    print("=" * 50)
    for key, value in SERVICE_DEFAULTS.items():
        os.environ.setdefault(key, value)
    ensure_worker_token()
    if "--no-docker" not in sys.argv[1:] and not start_services():
        sys.exit(1)

    print("📊 Alembic Migration...")
    result = subprocess.run(
        ["alembic", "upgrade", "head"], capture_output=True, text=True
    )
    print(
        "✅ Migration erfolgreich"
        if result.returncode == 0
        else f"⚠️  Migration Fehler:\n{result.stderr}"
    )
    point_workers_to_localhost()

    print("🚀 Starte Gunicorn auf http://127.0.0.1:5000")
    # Ein Worker (In-Memory-State), Threads fuer parallele Requests, siehe run_linux.py.
    # Auf dem Mac erlaubt PyTorch/MPS kein fork() nach dem Laden der Modelle: kein --preload.
    os.system(
        "gunicorn -w 1 --threads 4 -k gthread -b 127.0.0.1:5000 --timeout 240 'app:create_app(preload=True)'"
    )


if __name__ == "__main__":
    main()
