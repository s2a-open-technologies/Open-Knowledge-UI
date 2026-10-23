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
Qdrant Client Factory - Server-Modus (remote)
Verbindet sich zu einem laufenden Qdrant-Server (exe, docker, remote).
"""

import os

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


def get_qdrant_client(
    collection_name: str = "open_knowledge_ui",
    custom_host: str | None = None,
    custom_port: int | None = None,
    api_key: str | None = None,
) -> "QdrantClient":
    """
    Erstellt einen Qdrant-Client fuer einen laufenden Qdrant-Server.

    Args:
        collection_name: Name der Collection (default: "open_knowledge_ui")
        custom_host: Optionaler custom Host (ueberschreibt Settings)
        custom_port: Optionaler custom Port (ueberschreibt Settings)
        api_key: Optionaler API-Key fuer gesicherte Qdrant-Instanz

    Returns:
        QdrantClient: Konfigurierter Client
    """
    # Versuche Settings aus DB zu laden (fuer App-Context)
    try:
        from app.core.database_service import get_db_service
        from app.core.settings_service import SettingsService

        db = get_db_service()
        db_session = db.get_session()
        settings = SettingsService(db_session)

        host = custom_host or settings.get("QDRANT_HOST", "localhost")
        port = custom_port or settings.get_int("QDRANT_PORT", 6333)
        key = api_key or settings.get("QDRANT_API_KEY")

        db_session.close()

        if key:
            return QdrantClient(host=host, port=port, api_key=key)
        return QdrantClient(host=host, port=port)

    except Exception:
        # Fallback auf Environment
        pass

    # Environment/Defaults
    host = custom_host or os.getenv("QDRANT_HOST", "localhost")
    port = custom_port or int(os.getenv("QDRANT_PORT", 6333))
    key = api_key or os.getenv("QDRANT_API_KEY")

    if key:
        return QdrantClient(host=host, port=port, api_key=key)
    return QdrantClient(host=host, port=port)


def test_connection(client: QdrantClient, timeout: int = 5) -> tuple[bool, str]:
    """
    Testet die Qdrant-Verbindung.

    Args:
        client: QdrantClient Instanz
        timeout: Timeout in Sekunden

    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        collections = client.get_collections()
        return (
            True,
            f"Verbunden. {len(collections.collections)} Collection(s) gefunden.",
        )
    except Exception as e:
        return False, f"Verbindungsfehler: {e!s}"


def get_collection_info(
    client: QdrantClient, collection_name: str = "open_knowledge_ui"
) -> dict | None:
    """
    Holt Informationen ueber eine Collection.

    Args:
        client: QdrantClient Instanz
        collection_name: Name der Collection

    Returns:
        dict mit Collection-Info oder None
    """
    try:
        collection = client.get_collection(collection_name)

        try:
            count = client.count(collection_name)
            points_count = count.count
        except Exception:
            points_count = collection.points_count

        return {
            "name": collection_name,
            "points_count": points_count,
            "vectors_count": getattr(
                collection, "indexed_vectors_count", collection.points_count
            ),
            "status": str(collection.status),
            "vector_size": collection.config.params.vectors.size
            if hasattr(collection.config.params, "vectors")
            else None,
            "distance": str(collection.config.params.vectors.distance)
            if hasattr(collection.config.params, "vectors")
            else None,
        }
    except UnexpectedResponse:
        return None
    except Exception as e:
        return {"error": str(e)}
