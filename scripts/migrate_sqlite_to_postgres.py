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
Uebernimmt die Daten einer SQLite-Datenbank (alte ``data/app.db``) nach Postgres.

Ablauf:
  1. Postgres anlegen und migrieren:  DATABASE_URL=postgresql+psycopg://... alembic upgrade head
  2. Dieses Skript ausfuehren (liest die SQLite-Datei nur, veraendert sie nie):

       python scripts/migrate_sqlite_to_postgres.py --sqlite data/app.db --yes

Die SQLite-Datei wird zuerst in eine temporaere Kopie gelegt und dort auf den aktuellen
Alembic-Stand gebracht (alte Installationen stehen oft bei 0007). Die Zieltabellen werden
geleert (die Migration hat nur Standarddaten angelegt) und mit den Daten der Kopie gefuellt.
Danach werden die ID-Sequenzen nachgezogen und die Zeilenzahlen verglichen.

Optionen:
  --sqlite PFAD          Quelle (Standard: data/app.db)
  --database-url URL     Ziel (Standard: Umgebungsvariable DATABASE_URL)
  --dry-run              nur zaehlen, nichts schreiben
  --yes                  Rueckfrage ueberspringen
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sqlalchemy as sa

from app.core.models import Base

BATCH = 500


def upgrade_copy(sqlite_path: Path) -> Path:
    """Kopie der SQLite-Datei auf den aktuellen Alembic-Stand bringen."""
    from alembic.config import Config

    from alembic import command

    tmp = Path(tempfile.mkdtemp(prefix="rag-migrate-")) / "app.db"
    shutil.copy2(sqlite_path, tmp)
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{tmp}")
    previous = os.environ.pop("DATABASE_URL", None)  # env.py soll die Kopie nehmen
    try:
        command.upgrade(cfg, "head")
    finally:
        if previous:
            os.environ["DATABASE_URL"] = previous
    return tmp


def reflect_counts(engine, names):
    with engine.connect() as conn:
        return {
            n: conn.execute(sa.text(f'SELECT COUNT(*) FROM "{n}"')).scalar()
            for n in names
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sqlite", default="data/app.db")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    source_file = Path(args.sqlite)
    if not source_file.is_file():
        print(f"❌ SQLite-Datei nicht gefunden: {source_file}")
        return 1
    if not args.database_url or args.database_url.startswith("sqlite"):
        print(
            "❌ Ziel fehlt oder ist SQLite: --database-url postgresql+psycopg://... bzw. DATABASE_URL setzen."
        )
        return 1

    target = sa.create_engine(args.database_url)
    try:
        with target.connect() as conn:
            current = conn.execute(
                sa.text("SELECT version_num FROM alembic_version")
            ).scalar()
    except Exception as e:
        print(
            f"❌ Ziel nicht bereit ({type(e).__name__}). Zuerst 'alembic upgrade head' gegen Postgres ausfuehren."
        )
        return 1

    print(f"📄 Quelle: {source_file} (wird nur gelesen)")
    work = upgrade_copy(source_file)
    source = sa.create_engine(f"sqlite:///{work}")

    with source.connect() as conn:
        source_version = conn.execute(
            sa.text("SELECT version_num FROM alembic_version")
        ).scalar()
    if source_version != current:
        print(f"❌ Alembic-Stand verschieden: Quelle {source_version}, Ziel {current}.")
        return 1

    tables = [t for t in Base.metadata.sorted_tables]
    source_counts = reflect_counts(source, [t.name for t in tables])
    print("Zeilen in der Quelle:")
    for name, n in source_counts.items():
        print(f"  {name:28} {n}")
    if args.dry_run:
        print("Trockenlauf: nichts geschrieben.")
        return 0

    if not args.yes:
        answer = (
            input("Zieltabellen leeren und mit diesen Daten fuellen? [ja/N] ")
            .strip()
            .lower()
        )
        if answer not in ("ja", "j", "yes", "y"):
            print("Abgebrochen.")
            return 1

    with target.begin() as out, source.connect() as src:
        names = ", ".join(f'"{t.name}"' for t in tables)
        out.execute(sa.text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
        for table in tables:
            src_table = sa.Table(table.name, sa.MetaData(), autoload_with=source)
            columns = [c.name for c in table.columns if c.name in src_table.c]
            rows = src.execute(sa.select(*[src_table.c[c] for c in columns])).mappings()
            batch = []
            for row in rows:
                batch.append(dict(row))
                if len(batch) >= BATCH:
                    out.execute(table.insert(), batch)
                    batch = []
            if batch:
                out.execute(table.insert(), batch)

        # Sequenzen der Integer-Primaerschluessel nachziehen
        for table in tables:
            pk = list(table.primary_key.columns)
            if len(pk) == 1 and isinstance(pk[0].type, sa.Integer):
                col = pk[0].name
                out.execute(
                    sa.text(
                        f"SELECT setval(pg_get_serial_sequence('\"{table.name}\"', '{col}'), "
                        f'COALESCE((SELECT MAX("{col}") FROM "{table.name}"), 1), '
                        f'(SELECT MAX("{col}") FROM "{table.name}") IS NOT NULL)'
                    )
                )

    target_counts = reflect_counts(target, [t.name for t in tables])
    bad = {
        n: (source_counts[n], target_counts[n])
        for n in source_counts
        if source_counts[n] != target_counts[n]
    }
    if bad:
        print(f"❌ Zeilenzahlen weichen ab (Quelle, Ziel): {bad}")
        return 1
    print(
        f"✅ {sum(target_counts.values())} Zeilen in {len(tables)} Tabellen uebernommen."
    )
    print("   Die SQLite-Datei wurde nicht veraendert; sie kann als Sicherung bleiben.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
