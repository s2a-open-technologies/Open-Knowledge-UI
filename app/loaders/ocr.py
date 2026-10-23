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
OCR-Auswahl per Umgebung (ENV_ONLY, nie in der Settings-DB).

Gelesen werden die Werte direkt ueber ``os.getenv`` (gleiche Konvention wie
QDRANT_*), damit docker-compose/.env die Maschinenrealitaet beschreiben:
  - ``OCR_ENGINE``: ``tesseract`` (Default, CLI-Binary) | ``rapidocr``
    (ONNX, im Docling-Standardumfang enthalten) | ``ocrmac`` (nur macOS) | ``off``
  - ``OCR_LANGUAGES``: Komma-getrennte Sprachcodes (Default ``deu,eng``). Die
    Bedeutung haengt an der Engine: Tesseract liest ISO-639-2 (``deu``, ``eng``),
    RapidOCR PP-OCR-Codes (``de``; nur der erste Wert wird genutzt), ocrmac
    Locales (``de-DE``).
  - ``OCR_TESSERACT_CMD``: Pfad zur tesseract-Binary, wenn sie nicht im PATH
    liegt (z. B. Windows) — nur fuer die Tesseract-Engine.

Alles, was schiefgehen kann (unbekannter Wert, fehlendes Paket, falsches
Betriebssystem), faellt weich auf Tesseract-CLI zurueck statt ein Dokument
zu verlieren.
"""

# Standard library
import logging
import os
import sys

log = logging.getLogger(__name__)

DEFAULT_LANGUAGES = "deu,eng"


def _tesseract(langs: list) -> tuple[bool, object]:
    from docling.datamodel.pipeline_options import TesseractCliOcrOptions

    cmd = (os.getenv("OCR_TESSERACT_CMD") or "").strip()
    if cmd:
        return True, TesseractCliOcrOptions(lang=langs, tesseract_cmd=cmd)
    return True, TesseractCliOcrOptions(lang=langs)


def build_ocr_options() -> tuple[bool, object | None]:
    """
    Liest OCR_ENGINE/OCR_LANGUAGES/OCR_TESSERACT_CMD und liefert
    ``(do_ocr, ocr_options)``; ``ocr_options`` ist None nur bei ``off``.
    """
    engine = (os.getenv("OCR_ENGINE") or "tesseract").strip().lower()
    raw = (os.getenv("OCR_LANGUAGES") or DEFAULT_LANGUAGES).strip()
    langs = [p.strip() for p in raw.split(",") if p.strip()] or DEFAULT_LANGUAGES.split(
        ","
    )

    try:
        if engine == "off":
            return False, None
        if engine == "rapidocr":
            from docling.datamodel.pipeline_options import RapidOcrOptions

            return True, RapidOcrOptions(lang=langs)
        if engine == "ocrmac":
            from docling.datamodel.pipeline_options import OcrMacOptions

            if sys.platform != "darwin":
                log.warning(
                    "OCR_ENGINE=ocrmac laeuft nur auf macOS — nutze Tesseract-CLI."
                )
                return _tesseract(langs)
            return True, OcrMacOptions(lang=langs)
        if engine == "tesseract":
            return _tesseract(langs)
        log.warning(
            "Unbekannter OCR_ENGINE=%r (gueltig: tesseract, rapidocr, ocrmac, off) "
            "— nutze Tesseract-CLI.",
            engine,
        )
        return _tesseract(langs)
    except ImportError:
        log.warning(
            "OCR_ENGINE=%r ist in dieser Installation nicht enthalten — "
            "nutze Tesseract-CLI.",
            engine,
        )
        return _tesseract(langs)
