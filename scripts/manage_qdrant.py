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
Qdrant Management Script

Verwaltung und Migration zwischen Qdrant-Server-Instanzen.

Usage:
    # Status einer Qdrant-Instanz pruefen
    uv run python scripts/manage_qdrant.py status --host localhost --port 6333

    # Migration Server A -> Server B (Dry-Run)
    uv run python scripts/manage_qdrant.py migrate \
        --source-host localhost --source-port 6333 \
        --target-host 192.168.1.100 --target-port 6333 --dry-run

    # Migration wirklich durchfuehren
    uv run python scripts/manage_qdrant.py migrate \
        --source-host localhost --source-port 6333 \
        --target-host 192.168.1.100 --target-port 6333

    # Collection-Info anzeigen
    uv run python scripts/manage_qdrant.py info --host localhost --port 6333
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from tqdm import tqdm


def create_remote_client(
    host: str, port: int, api_key: str | None = None
) -> QdrantClient:
    """Erstellt Client fuer einen Qdrant Server."""
    if api_key:
        return QdrantClient(host=host, port=port, api_key=api_key)
    return QdrantClient(host=host, port=port)


def test_connection(client: QdrantClient, name: str) -> tuple[bool, str]:
    """Testet die Verbindung zu einem Qdrant Server."""
    try:
        collections = client.get_collections()
        return True, f"Verbunden. {len(collections.collections)} Collection(s)"
    except Exception as e:
        return False, f"Fehler: {e!s}"


def get_collection_info(client: QdrantClient, collection_name: str) -> dict | None:
    """Holt Informationen ueber eine Collection."""
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
    except Exception:
        return None


def migrate_collection(
    source_client: QdrantClient,
    target_client: QdrantClient,
    collection_name: str,
    batch_size: int = 100,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Migriert eine Collection von Source zu Target."""
    source_info = get_collection_info(source_client, collection_name)
    if not source_info:
        return False, f"Collection '{collection_name}' nicht in Source gefunden"

    print(f"\n{'=' * 60}")
    print(f"Migration: {collection_name}")
    print(f"{'=' * 60}")
    print(f"Source Points: {source_info['points_count']:,}")
    print(f"Vector Size: {source_info['vector_size']}")
    print(f"Distance: {source_info['distance']}")

    if dry_run:
        print(f"\n[DRY-RUN] Wuerde {source_info['points_count']:,} Punkte migrieren")
        return True, "Dry-Run erfolgreich"

    try:
        target_client.get_collection(collection_name)
        print("\nWarnung: Collection existiert bereits in Target")

        target_info = get_collection_info(target_client, collection_name)
        if target_info and target_info["vector_size"] != source_info["vector_size"]:
            return (
                False,
                f"Vector Size mismatch: Source={source_info['vector_size']}, Target={target_info['vector_size']}",
            )

        response = input("Collection ueberschreiben? (j/N): ")
        if response.lower() != "j":
            print("Migration abgebrochen")
            return False, "Abgebrochen durch User"

        target_client.delete_collection(collection_name)
        print("Bestehende Collection geloescht")
    except Exception:
        pass

    try:
        vectors_config = qdrant_models.VectorParams(
            size=source_info["vector_size"],
            distance=getattr(qdrant_models.Distance, source_info["distance"].upper()),
        )

        target_client.create_collection(
            collection_name=collection_name, vectors_config=vectors_config
        )
        print("Collection erstellt")
    except Exception as e:
        return False, f"Fehler beim Erstellen der Collection: {e}"

    total_points = source_info["points_count"]
    migrated = 0
    errors = 0

    print("\nMigration startet...")

    offset = None
    with tqdm(total=total_points, desc="Punkte") as pbar:
        while True:
            try:
                records, offset = source_client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )

                if not records:
                    break

                points = [
                    qdrant_models.PointStruct(
                        id=record.id, vector=record.vector, payload=record.payload
                    )
                    for record in records
                ]

                target_client.upsert(collection_name=collection_name, points=points)

                migrated += len(points)
                pbar.update(len(points))

                if offset is None:
                    break

            except Exception as e:
                errors += 1
                print(f"\nFehler bei Batch: {e}")
                if offset is None:
                    break
                continue

    target_info = get_collection_info(target_client, collection_name)
    if target_info:
        if target_info["points_count"] == source_info["points_count"]:
            print("\nMigration erfolgreich!")
            print(f"  Migriert: {migrated:,} Punkte")
            return True, f"Migration erfolgreich: {migrated:,} Punkte"
        else:
            print("\nValidierungsfehler!")
            print(f"  Source: {source_info['points_count']:,}")
            print(f"  Target: {target_info['points_count']:,}")
            return (
                False,
                f"Validierungsfehler: {target_info['points_count']}/{source_info['points_count']}",
            )

    return True, f"Migration abgeschlossen: {migrated:,} Punkte"


def cmd_status(args):
    """Zeigt Status einer Qdrant-Instanz."""
    print(f"Qdrant Status ({args.host}:{args.port})")
    print("=" * 60)

    try:
        client = create_remote_client(args.host, args.port, args.api_key)
        success, msg = test_connection(client, "remote")
        print(f"  {msg}")

        if success:
            try:
                collections = client.get_collections()
                if not collections.collections:
                    print("\n  Keine Collections vorhanden")
                else:
                    print(f"\n{'Collection':<20} {'Punkte':>12} {'Status':<15}")
                    print("-" * 50)
                    for col in collections.collections:
                        info = get_collection_info(client, col.name)
                        if info and "error" not in info:
                            print(
                                f"{col.name:<20} {info['points_count']:>12,} {info['status']:<15}"
                            )
                        else:
                            print(f"{col.name:<20} {'N/A':>12} {'N/A':<15}")
            except Exception as e:
                print(f"  Fehler beim Lesen: {e}")
    except Exception as e:
        print(f"  Nicht verfügbar: {e}")

    print("\n" + "=" * 60)


def cmd_migrate(args):
    """Fuehrt Migration zwischen zwei Qdrant-Servern durch."""
    print("Qdrant Server-Migration")
    print("=" * 60)

    if args.dry_run:
        print("[DRY-RUN] Modus (keine Aenderungen werden vorgenommen)")

    print(f"\nSource: {args.source_host}:{args.source_port}")
    source_client = create_remote_client(
        args.source_host, args.source_port, args.source_api_key
    )

    print(f"Target: {args.target_host}:{args.target_port}")
    target_client = create_remote_client(
        args.target_host, args.target_port, args.target_api_key
    )

    print("\nVerbindungen testen...")
    src_ok, src_msg = test_connection(source_client, "source")
    tgt_ok, tgt_msg = test_connection(target_client, "target")

    print(f"  Source: {src_msg}")
    print(f"  Target: {tgt_msg}")

    if not src_ok:
        print("\nSource nicht verfuegbar. Abbruch.")
        sys.exit(1)
    if not tgt_ok:
        print("\nTarget nicht verfuegbar. Abbruch.")
        sys.exit(1)

    if args.collection:
        collections = [args.collection]
    else:
        try:
            cols = source_client.get_collections()
            collections = [c.name for c in cols.collections]
        except Exception as e:
            print(f"\nFehler beim Lesen der Collections: {e}")
            sys.exit(1)

    if not collections:
        print("\nKeine Collections in Source gefunden.")
        sys.exit(1)

    print(f"\nZu migrierende Collections: {', '.join(collections)}")

    if not args.dry_run:
        response = input("\nMigration starten? (j/N): ")
        if response.lower() != "j":
            print("Abgebrochen")
            sys.exit(0)

    success_count = 0
    error_count = 0

    for collection_name in collections:
        success, msg = migrate_collection(
            source_client,
            target_client,
            collection_name,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )

        if success:
            success_count += 1
        else:
            error_count += 1
            print(f"Fehler: {msg}")

    print(f"\n{'=' * 60}")
    print("Zusammenfassung:")
    print(f"  Erfolgreich: {success_count}")
    print(f"  Fehler: {error_count}")
    if args.dry_run:
        print("\n  (Dry-Run: Keine Aenderungen vorgenommen)")
    print("=" * 60)

    if error_count > 0:
        sys.exit(1)


def cmd_info(args):
    """Zeigt Collection-Info einer Qdrant-Instanz."""
    print(f"Collection Info ({args.host}:{args.port})")
    print("=" * 60)

    client = create_remote_client(args.host, args.port, args.api_key)
    ok, msg = test_connection(client, "remote")
    print(f"  {msg}")

    if not ok:
        sys.exit(1)

    try:
        collections = client.get_collections()

        if not collections.collections:
            print("\n  Keine Collections vorhanden")
            return

        print(f"\n{'Collection':<20} {'Punkte':>12} {'Status':<15}")
        print("-" * 50)

        for col in collections.collections:
            info = get_collection_info(client, col.name)
            if info and "error" not in info:
                print(
                    f"{col.name:<20} {info['points_count']:>12,} {info['status']:<15}"
                )
            else:
                print(f"{col.name:<20} {'N/A':>12} {'N/A':<15}")

    except Exception as e:
        print(f"\nFehler: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Qdrant Management Script (Server-zu-Server)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
    # Status pruefen
    uv run python scripts/manage_qdrant.py status --host localhost --port 6333

    # Migration Server A -> Server B (Dry-Run)
    uv run python scripts/manage_qdrant.py migrate \\
        --source-host localhost --source-port 6333 \\
        --target-host 192.168.1.100 --target-port 6333 --dry-run

    # Migration durchfuehren
    uv run python scripts/manage_qdrant.py migrate \\
        --source-host localhost --source-port 6333 \\
        --target-host 192.168.1.100 --target-port 6333

    # Nur spezifische Collection
    uv run python scripts/manage_qdrant.py migrate \\
        --source-host localhost --source-port 6333 \\
        --target-host 192.168.1.100 --target-port 6333 \\
        --collection open_knowledge_ui

    # Collection-Info
    uv run python scripts/manage_qdrant.py info --host localhost --port 6333
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Befehl")

    # Status Command
    status_parser = subparsers.add_parser("status", help="Zeigt Verbindungsstatus")
    status_parser.add_argument(
        "--host", default="localhost", help="Qdrant Host (default: localhost)"
    )
    status_parser.add_argument(
        "--port", type=int, default=6333, help="Qdrant Port (default: 6333)"
    )
    status_parser.add_argument("--api-key", help="API Key fuer gesicherte Qdrant")

    # Migrate Command
    migrate_parser = subparsers.add_parser(
        "migrate", help="Migriert Collections zwischen Servern"
    )
    migrate_parser.add_argument(
        "--source-host", default="localhost", help="Source Host"
    )
    migrate_parser.add_argument(
        "--source-port", type=int, default=6333, help="Source Port"
    )
    migrate_parser.add_argument("--source-api-key", help="Source API Key")
    migrate_parser.add_argument("--target-host", required=True, help="Target Host")
    migrate_parser.add_argument(
        "--target-port", type=int, default=6333, help="Target Port"
    )
    migrate_parser.add_argument("--target-api-key", help="Target API Key")
    migrate_parser.add_argument("--collection", help="Nur diese Collection migrieren")
    migrate_parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch-Groesse (default: 100)"
    )
    migrate_parser.add_argument(
        "--dry-run", action="store_true", help="Nur simulieren, keine Aenderungen"
    )

    # Info Command
    info_parser = subparsers.add_parser("info", help="Zeigt Collection-Info")
    info_parser.add_argument("--host", default="localhost", help="Qdrant Host")
    info_parser.add_argument("--port", type=int, default=6333, help="Qdrant Port")
    info_parser.add_argument("--api-key", help="API Key")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "status":
        cmd_status(args)
    elif args.command == "migrate":
        cmd_migrate(args)
    elif args.command == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()
