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
from datetime import datetime
from pathlib import Path
from typing import Any

# Strukturierte Daten: nur als Referenz nach DATA_DIR für die Pandas-Tools,
# niemals in die Vektordatenbank embedden.
STRUCTURED_EXTS = {".csv", ".xlsx", ".xls"}

# Dokumente: werden gechunkt und in die Vektordatenbank ingested.
DOCUMENT_EXTS = {".pdf", ".docx", ".doc", ".odt", ".md"}

# Bilder gibt es nur als Chat-Anhang: das Bildmodell liest sie, der Text wird durchsuchbar.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def get_data_dir() -> Path:
    """Liefert das konfigurierte DATA_DIR (Settings DB → env → Default)."""
    try:
        from flask import current_app

        settings = current_app.extensions.get("settings")
        if settings:
            data_dir = settings.get("DATA_DIR", "./data")
        else:
            data_dir = os.getenv("DATA_DIR", "./data")
    except Exception:
        data_dir = os.getenv("DATA_DIR", "./data")
    return Path(data_dir)


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
    files, paths=None, uploader_id: int | None = None
) -> dict[str, Any]:
    """Speichert hochgeladene Dateien je nach Typ.

    - Strukturierte Daten (CSV/XLSX/XLS) gehen direkt nach DATA_DIR und sind
      sofort für die Pandas-Tools verfügbar (keine Ingestion, keine Vektoren).
    - Dokumente (PDF/DOCX/DOC/ODT/MD) gehen nach data/files zur Ingestion.

    ``paths`` (``app.knowledge.service.KnowledgePaths``) legt die Zielordner eines
    Knowledge-Bereichs fest; ohne Angabe gelten die Standard-Ordner.

    Returns:
        dict mit ``documents`` (zur Ingestion), ``data_files`` (sofort
        verfügbar), ``skipped`` (unbekannte Endungen) und ``duplicate``
        (Dateiname, falls ein bereits verarbeitetes Dokument erneut kam).
    """
    from flask import current_app

    if paths is not None:
        files_dir, processed_dir, data_dir = (
            paths.files_dir,
            paths.processed_dir,
            paths.data_dir,
        )
    else:
        project_root = Path(current_app.root_path).parent
        files_dir = project_root / "data" / "files"
        processed_dir = project_root / "data" / "processed_files"
        data_dir = get_data_dir()
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
        not str(file_path).startswith(str(data_dir))
        or file_path.suffix.lower() not in STRUCTURED_EXTS
    ):
        return False
    if file_path.is_file():
        file_path.unlink()
        return True
    return False


def extract_filename_from_payload(payload: dict[str, Any]) -> str | None:
    """
    Extrahiert den Dateinamen aus Qdrant-Dokument-Payload.
    Prüft 'filename' und 'source' Felder.

    Args:
        payload: Qdrant Dokument Payload (dict)

    Returns:
        Dateiname oder None
    """
    if not payload:
        return None

    # Versuche filename zuerst
    filename = payload.get("filename")
    if filename:
        return filename

    # Fallback auf source
    source = payload.get("source")
    if source:
        # Source ist oft ein Pfad, extrahiere nur den Dateinamen
        return Path(source).name

    return None


def check_data_file_exists(filename: str) -> bool:
    """
    Prüft ob eine Datei im DATA_DIR existiert.

    Args:
        filename: Name der Datei

    Returns:
        True wenn Datei existiert, sonst False
    """
    from flask import current_app

    try:
        settings = current_app.extensions.get("settings")
        if settings:
            data_dir = settings.get("DATA_DIR", "./data")
        else:
            data_dir = os.getenv("DATA_DIR", "./data")
    except Exception:
        data_dir = os.getenv("DATA_DIR", "./data")

    file_path = Path(data_dir) / filename
    return file_path.exists() and file_path.is_file()


def delete_associated_file(payload: dict[str, Any]) -> bool:
    """
    Löscht zugehörige Datei basierend auf Qdrant-Metadaten.

    Args:
        payload: Qdrant Dokument Payload (dict)

    Returns:
        True wenn Datei gelöscht wurde oder nicht existierte,
        False bei Fehler
    """
    filename = extract_filename_from_payload(payload)
    if not filename:
        return False

    ext = Path(filename).suffix.lower()

    if ext in [".csv", ".xlsx", ".xls"]:
        # CSV/Excel werden nach erfolgreicher Ingestion nach DATA_DIR verschoben
        try:
            from flask import current_app

            settings = current_app.extensions.get("settings")
            if settings:
                data_dir = settings.get("DATA_DIR", "./data")
            else:
                data_dir = os.getenv("DATA_DIR", "./data")
        except Exception:
            data_dir = os.getenv("DATA_DIR", "./data")

        candidates = [Path(data_dir) / filename]
    elif ext in [".pdf", ".docx", ".doc", ".odt", ".md"]:
        # PDF/DOCX/DOC werden nach erfolgreicher Ingestion von data/files/<rel>
        # nach data/processed_files/<rel> verschoben (rel enthaelt die
        # Nutzer-Unterordner-Struktur). 'source' im Payload zeigt noch auf den
        # urspruenglichen data/files-Pfad zur Ingestion-Zeit.
        source = payload.get("source")
        candidates = []
        if source:
            try:
                rel_path = os.path.relpath(source, "./data/files")
                candidates.append(Path("./data/processed_files") / rel_path)
            except ValueError:
                pass
            # Knowledge-Bereiche: data/knowledge/<id>/files -> .../processed_files
            src_path = Path(source)
            if "files" in src_path.parts:
                idx = len(src_path.parts) - 1 - src_path.parts[::-1].index("files")
                parts = list(src_path.parts)
                parts[idx] = "processed_files"
                candidates.append(Path(*parts))
            candidates.append(src_path)
        if not candidates:
            candidates = [Path("./data/processed_files") / filename]
    else:
        return False

    try:
        for file_path in candidates:
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                return True
    except Exception as e:
        # Fehler beim Löschen loggen aber nicht blockieren
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Fehler beim Löschen der Datei {filename}: {e}")
        return False

    return False


def get_data_file_info(filename: str) -> dict[str, Any] | None:
    """
    Holt Informationen über eine Datei im DATA_DIR.

    Args:
        filename: Name der Datei

    Returns:
        Dict mit file_info oder None
    """
    from flask import current_app

    try:
        settings = current_app.extensions.get("settings")
        if settings:
            data_dir = settings.get("DATA_DIR", "./data")
        else:
            data_dir = os.getenv("DATA_DIR", "./data")
    except Exception:
        data_dir = os.getenv("DATA_DIR", "./data")

    file_path = Path(data_dir) / filename

    if not file_path.exists() or not file_path.is_file():
        return None

    try:
        stat = file_path.stat()
        return {
            "filename": filename,
            "path": str(file_path),
            "size": stat.st_size,
            "size_human": _format_size(stat.st_size),
            "modified": stat.st_mtime,
        }
    except Exception:
        return None


def _format_size(size_bytes: int) -> str:
    """Formatiert Bytes in menschenlesbare Größe."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
