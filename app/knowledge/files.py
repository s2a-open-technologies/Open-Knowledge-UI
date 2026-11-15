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
File Manager Utilities für Qdrant-zugehörige Dateien und strukturierte
Daten (CSV/Excel als Pandas-Referenz, NICHT in der Vektordatenbank).
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Strukturierte Daten: nur als Referenz nach DATA_DIR für die Pandas-Tools,
# niemals in die Vektordatenbank embedden.
STRUCTURED_EXTS = {".csv", ".xlsx", ".xls"}

# Dokumente: werden gechunkt und in die Vektordatenbank ingested.
DOCUMENT_EXTS = {".pdf", ".docx", ".doc", ".odt", ".md"}

# Bilder gibt es nur als Chat-Anhang: das Bildmodell liest sie, der Text wird durchsuchbar.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def get_data_dir() -> Path:
    """Liefert das konfigurierte DATA_DIR (Settings DB → env → Default)."""
    from app.core import runtime

    settings = runtime.state.settings
    try:
        data_dir = settings.get("DATA_DIR", "./data") if settings else None
    except Exception:
        data_dir = None
    return Path(data_dir or os.getenv("DATA_DIR", "./data"))


def _unique_path(directory: Path, filename: str) -> Path:
    """Kollisionsfreier Zielpfad (Timestamp-Suffix bei existierender Datei)."""
    target = directory / filename
    if not target.exists():
        return target
    base, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
    return directory / f"{base}{timestamp}{ext}"


UPLOADER_SUFFIX = ".uploader"


def read_uploader(file_path: str, consume: bool = False) -> int | None:
    """Nutzer-ID aus der Begleitdatei einer hochgeladenen Datei (``None`` = unbekannt)."""
    sidecar = Path(f"{file_path}{UPLOADER_SUFFIX}")
    try:
        value = int(sidecar.read_text().strip())
    except (OSError, ValueError):
        return None
    if consume:
        sidecar.unlink(missing_ok=True)
    return value


def store_uploaded_files(
    files, paths, uploader_id: int | None = None
) -> dict[str, Any]:
    """Speichert hochgeladene Dateien je nach Typ.

    - Strukturierte Daten (CSV/XLSX/XLS) gehen direkt nach ``paths.data_dir`` und sind
      sofort für die Pandas-Tools verfügbar (keine Ingestion, keine Vektoren).
    - Dokumente (PDF/DOCX/DOC/ODT/MD) gehen nach ``paths.files_dir`` zur Ingestion.

    ``paths`` (``app.knowledge.service.KnowledgePaths``) legt die Zielordner eines
    Knowledge-Bereichs fest.

    Returns:
        dict mit ``documents`` (zur Ingestion), ``data_files`` (sofort
        verfügbar), ``skipped`` (unbekannte Endungen) und ``duplicate``
        (Dateiname, falls ein bereits verarbeitetes Dokument erneut kam).
    """
    files_dir, processed_dir, data_dir = (
        paths.files_dir,
        paths.processed_dir,
        paths.data_dir,
    )
    for d in (files_dir, processed_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    documents: list[str] = []
    data_files: list[str] = []
    skipped: list[str] = []

    for file in files:
        if not file or not file.filename:
            continue
        filename = os.path.basename(file.filename)
        ext = Path(filename).suffix.lower()

        if ext in STRUCTURED_EXTS:
            target = _unique_path(data_dir, filename)
            file.save(target)
            data_files.append(target.name)
        elif ext in DOCUMENT_EXTS:
            if (processed_dir / filename).exists():
                return {
                    "documents": documents,
                    "data_files": data_files,
                    "skipped": skipped,
                    "duplicate": filename,
                }
            file.save(files_dir / filename)
            if uploader_id is not None:
                # Wer hochgeladen hat, merkt sich die Ingestion fuer "Meine Uploads"
                (files_dir / f"{filename}{UPLOADER_SUFFIX}").write_text(
                    str(int(uploader_id))
                )
            documents.append(filename)
        else:
            skipped.append(filename)

    return {
        "documents": documents,
        "data_files": data_files,
        "skipped": skipped,
        "duplicate": None,
    }


def list_data_files(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Listet die strukturierten Daten (CSV/Excel) im DATA_DIR auf."""
    data_dir = Path(data_dir) if data_dir is not None else get_data_dir()
    result: list[dict[str, Any]] = []
    if not data_dir.exists():
        return result

    for file_path in sorted(data_dir.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in STRUCTURED_EXTS:
            continue
        stat = file_path.stat()
        result.append(
            {
                "filename": file_path.name,
                "size": stat.st_size,
                "size_human": _format_size(stat.st_size),
                "modified": stat.st_mtime,
            }
        )
    return result


def delete_data_file(filename: str, data_dir: Path | None = None) -> bool:
    """Löscht eine strukturierte Datei sicher aus dem DATA_DIR."""
    data_dir = (Path(data_dir) if data_dir is not None else get_data_dir()).resolve()
    name = os.path.basename(filename)
    file_path = (data_dir / name).resolve()
    if (
        not file_path.is_relative_to(data_dir)
        or file_path.suffix.lower() not in STRUCTURED_EXTS
    ):
        return False
    if file_path.is_file():
        file_path.unlink()
        return True
    return False


def check_data_file_exists(filename: str) -> bool:
    """Prüft, ob die strukturierte Datei im DATA_DIR liegt.

    ``filename`` ist relativ zum globalen DATA_DIR und enthält bei Knowledge-Bereichen
    schon den Unterordner (``knowledge_3/x.csv``, siehe ``index_data_references``).
    """
    return (get_data_dir() / filename).is_file()


def _archive_candidates(source: str | None, filename: str) -> list[Path]:
    """Moegliche Orte der archivierten Datei eines Dokuments.

    Nach der Ingestion wandert sie von ``files/<rel>`` nach ``processed_files/<rel>``
    (``rel`` enthaelt die Unterordner). ``source`` im Payload zeigt noch auf den Pfad zur
    Ingestion-Zeit: ``data/files/<rel>`` (Standard) oder ``data/knowledge/<id>/files/<rel>``.
    """
    if not source:
        return [Path("./data/processed_files") / filename]
    candidates = []
    try:
        candidates.append(
            Path("./data/processed_files") / os.path.relpath(source, "./data/files")
        )
    except ValueError:
        pass
    src = Path(source)
    if "files" in src.parts:
        parts = list(src.parts)
        parts[len(parts) - 1 - parts[::-1].index("files")] = "processed_files"
        candidates.append(Path(*parts))
    candidates.append(src)
    return candidates


def delete_associated_file(payload: dict[str, Any]) -> bool:
    """Löscht die zu einem Qdrant-Payload gehörende Datei (Archiv bzw. DATA_DIR).

    Returns:
        True, wenn eine Datei gelöscht wurde; False, wenn keine gefunden wurde oder das
        Löschen scheiterte (wird geloggt, blockiert aber nicht).
    """
    source = payload.get("source") if payload else None
    filename = (payload or {}).get("filename") or (Path(source).name if source else None)
    if not filename:
        return False

    ext = Path(filename).suffix.lower()
    if ext in STRUCTURED_EXTS:
        # CSV/Excel liegen nach der Ingestion im DATA_DIR
        candidates = [get_data_dir() / filename]
    elif ext in DOCUMENT_EXTS:
        candidates = _archive_candidates(source, filename)
    else:
        return False

    try:
        for file_path in candidates:
            if file_path.is_file():
                file_path.unlink()
                return True
    except Exception as e:
        logger.error(f"Fehler beim Löschen der Datei {filename}: {e}")
    return False


def _format_size(size_bytes: int) -> str:
    """Formatiert Bytes in menschenlesbare Größe."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
