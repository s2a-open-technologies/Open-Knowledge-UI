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
Gemeinsame Basisklasse fuer Dokumenten-Preprocessing (PDF, DOCX, ODT).

Verarbeitung erfolgt einheitlich ueber Docling (DoclingDocument). Das Ergebnis
ist **Markdown** (``build_markdown_from_docling``):
  - Ueberschriften mit Ebenen, Listen, Tabellen als Markdown-Tabellen
    (Docling-Export in Lesereihenfolge),
  - Bilder als ``[BILD-BESCHREIBUNG: ...]`` an ihrer Stelle (VisionLLM mit Retry,
    Deduplikation via perceptual Hash).
Das Markdown wird gespeichert und spaeter von ``app/ingest/splitter.py`` geteilt;
so muss das teure Parsen (Layout, OCR, Vision) beim Neu-Indexieren nicht wiederholt
werden.
"""

# Standard library
import base64
import logging
import re
import time
from io import BytesIO

import imagehash
from docling_core.types.doc import PictureItem, TableItem

# Custom imports
from app.llm.local_llm import VisionLLM

logger = logging.getLogger(__name__)


def _item_pages(doc) -> list[int]:
    """Seiten des Dokuments in Lesereihenfolge (ohne Duplikate), aus der Herkunft der Elemente."""
    pages: list[int] = []
    for item, _ in doc.iterate_items():
        prov = getattr(item, "prov", None) or []
        page = getattr(prov[0], "page_no", None) if prov else None
        if page is not None and (not pages or pages[-1] != page) and page not in pages:
            pages.append(page)
    return pages


def apply_page_marks(markdown: str, pages: list[int]) -> str:
    """Ersetzt die Seitenwechsel-Platzhalter durch ``<!--p:N-->``-Marker (siehe Splitter).

    Passt die Zahl der Platzhalter nicht zu den Seiten, entfallen die Marker (lieber keine
    Seitenzahl als eine falsche).
    """
    pattern = re.compile(rf"\s*{re.escape(PAGE_BREAK)}\s*")
    breaks = len(pattern.findall(markdown))
    if len(pages) < 2 or breaks != len(pages) - 1:
        if breaks:
            logger.warning(
                f"Seitenzuordnung verworfen: {breaks} Seitenwechsel, {len(pages)} Seiten"
            )
        return pattern.sub("\n\n", markdown)
    numbers = iter(pages[1:])
    out = pattern.sub(lambda _m: f"\n\n<!--p:{next(numbers)}-->\n\n", markdown)
    return f"<!--p:{pages[0]}-->\n\n{out}"


def _pil_to_b64(pil_image) -> str:
    """Encodiert ein PIL-Bild in Base64 (PNG)."""
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8").replace("\n", "")


def _get_picture_image(picture):
    """Liefert das PIL-Bild eines Docling PictureItem oder None.

    Benoetigt PdfPipelineOptions(generate_picture_images=True), damit
    picture.image gesetzt ist.
    """
    img_ref = getattr(picture, "image", None)
    if img_ref is None:
        return None
    try:
        return img_ref.pil_image
    except Exception:
        return None


# Eindeutige Platzhalter im Docling-Markdown-Export.
IMAGE_MARKER = "@@DOCLING-BILD@@"
PAGE_BREAK = "@@DOCLING-SEITE@@"
TABLE_MARKER = "@@DOCLING-TABELLE-{}@@"


def _picture_block(description: str) -> str:
    """Bildbeschreibung als eigener Absatz.

    Zeilen, die mit ``#`` beginnen, werden entschaerft, sonst wuerde der
    Markdown-Splitter sie als Ueberschriften behandeln.
    """
    text = re.sub(r"(?m)^(\s*)#", r"\1\\#", description.strip())
    return f"\n\n[BILD-BESCHREIBUNG: {text}]\n\n"


def _vision_enabled() -> bool:
    try:
        from app.core.database_service import get_db_service
        from app.core.settings_service import SettingsService

        session = get_db_service().get_session()
        try:
            return SettingsService(session).get_bool("INGEST_VISION_ENABLED", True)
        finally:
            session.close()
    except Exception:
        return True


class PreprocessBase:
    """Basis-Klasse fuer Dokumenten-Preprocessing via Docling.

    Subklassen (PreprocessPDF, PreprocessDOCX) implementieren
    process_document(file_path) und rufen self.build_chunks_from_docling()
    mit dem konvertierten DoclingDocument auf.
    """

    def __init__(
        self,
        model: VisionLLM = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        table_model=None,
    ):
        self.model = model or VisionLLM()
        self._table_model = (
            table_model  # Standard: TaskLLM("tables"), erst bei Bedarf angelegt
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ---- Bild-Hilfsfunktionen (formatunabhaengig) ----

    def process_image_with_retry(
        self,
        b64image: str,
        image_path: str,
        current_image: int,
        unique_image_count: int,
    ) -> dict:
        """Verarbeitet ein Bild mit Retry-Logik fuer LLM-Timeouts."""
        message = self.model.generate_image_message(b64image)

        retry_count = 0
        last_error = None

        while retry_count < self.max_retries:
            try:
                description = self.model.generate(message)
                desc_text = (
                    description.content
                    if hasattr(description, "content")
                    else str(description)
                )
                return {"image": desc_text}
            except Exception as e:
                retry_count += 1
                last_error = e
                logger.warning(
                    f"Bild {current_image}/{unique_image_count} - "
                    f"Versuch {retry_count}/{self.max_retries} fehlgeschlagen: {str(e)[:100]}"
                )
                if retry_count < self.max_retries:
                    time.sleep(self.retry_delay * retry_count)

        logger.error(
            f"Bild {current_image}/{unique_image_count} - "
            f"Alle {self.max_retries} Versuche fehlgeschlagen: {image_path}"
        )
        return {
            "error": f"LLM-Error nach {self.max_retries} Versuchen: {str(last_error)[:100]}"
        }

    def _collect_unique_pictures(self, doc) -> dict:
        """Pre-Pass: sammelt einzigartige Bilder (phash-Dedup).

        Returns:
            {picture_self_ref: PIL.Image} fuer die erste Vorkommen jedes
            einzigartigen Bildes (in Lesereihenfolge).
        """
        unique_pil: dict = {}
        seen_hashes = []
        for pic in doc.pictures:
            pil = _get_picture_image(pic)
            if pil is None:
                continue
            try:
                h = imagehash.phash(pil)
            except Exception:
                continue
            if any(h - sh < 5 for sh in seen_hashes):
                continue
            seen_hashes.append(h)
            unique_pil[pic.self_ref] = pil
        return unique_pil

    # ---- Markdown-Erzeugung (formatunabhaengig, Docling-basiert) ----

    def build_markdown_from_docling(self, doc, file_path: str, table_sink=None) -> str:
        """Baut aus einem DoclingDocument Markdown mit Bildbeschreibungen.

        Der Docling-Export liefert Ueberschriften, Listen und Tabellen in
        Lesereihenfolge; fuer Bilder setzt er einen Marker, der durch die
        VisionLLM-Beschreibung ersetzt wird. Doppelte Bilder (phash) bekommen keine
        zweite Beschreibung.

        Args:
            doc: DoclingDocument (aus DocumentConverter.convert().document).
            file_path: Pfad zur Quelldatei (nur fuer Log-Meldungen).
            table_sink: optional ``sink(dataframe, nummer) -> info | None`` (siehe
                ``app/ingest/tables.py``); Tabellen werden dann als CSV gespeichert und
                im Markdown durch einen Verweis ersetzt.

        Returns:
            Markdown-Text.
        """
        # Bildbeschreibung abschaltbar (Einlesen & Modelle): dann kein Vision-Aufruf,
        # die Bild-Marker verschwinden aus dem Text.
        unique_pil = self._collect_unique_pictures(doc) if _vision_enabled() else {}
        pictures = [
            item for item, _ in doc.iterate_items() if isinstance(item, PictureItem)
        ]

        descriptions: list[str] = []
        current_image = 0
        for item in pictures:
            pil = unique_pil.get(item.self_ref)
            if pil is None:
                descriptions.append("")  # Duplikat oder ohne Bild
                continue
            current_image += 1
            print(f"Verarbeite Bild {current_image}/{len(unique_pil)}", flush=True)
            result = self.process_image_with_retry(
                _pil_to_b64(pil),
                f"<docling_picture:{item.self_ref}>",
                current_image,
                len(unique_pil),
            )
            if "image" in result:
                descriptions.append(_picture_block(result["image"]))
            else:
                logger.error(f"Bild übersprungen: {result.get('error')}")
                descriptions.append("")

        pages = _item_pages(doc)
        markdown = doc.export_to_markdown(
            image_placeholder=IMAGE_MARKER,
            page_break_placeholder=PAGE_BREAK if len(pages) > 1 else None,
        )
        if table_sink is not None:
            markdown = self._extract_tables(doc, markdown, table_sink, file_path)
        parts = markdown.split(IMAGE_MARKER)
        if len(parts) - 1 == len(descriptions):
            out = [parts[0]]
            for description, tail in zip(descriptions, parts[1:], strict=True):
                out.append(description)
                out.append(tail)
            markdown = "".join(out)
        else:
            # Sollte nicht vorkommen (Export und Iteration laufen ueber dieselben Elemente).
            logger.warning(
                f"{file_path}: {len(parts) - 1} Bild-Marker, {len(descriptions)} Bilder; "
                "Beschreibungen werden am Ende angehaengt"
            )
            markdown = "".join(parts) + "".join(f"\n\n{d}" for d in descriptions if d)

        markdown = apply_page_marks(markdown, pages)
        return re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"

    def _extract_tables(self, doc, markdown: str, table_sink, file_path: str) -> str:
        """Speichert Tabellen als CSV und ersetzt sie im Markdown durch Verweise.

        Die Tabellen-Markdown aus ``TableItem.export_to_markdown`` steht woertlich im
        Dokument-Export; in zwei Schritten (Marker, dann Text) ersetzt, damit identische
        Tabellen und Ersatztexte, die die Tabelle selbst enthalten, nichts durcheinanderbringen.
        """
        from app.ingest.table_merge import is_continuation, merge_frames, table_geometry
        from app.ingest.tables import render_reference

        refiner = self._table_refiner(file_path)
        # 1. Tabellen einsammeln (ggf. vom Bildmodell korrigiert). "gap": zwischen zwei Tabellen steht anderer Inhalt.
        parts: list[dict] = []
        gap = True
        try:
            for item, _ in doc.iterate_items():
                if not isinstance(item, TableItem):
                    gap = True
                    continue
                try:
                    table_markdown = item.export_to_markdown(doc=doc)
                    frame = item.export_to_dataframe(doc=doc)
                    if (
                        refiner is not None
                        and frame.shape[0] >= 2
                        and frame.shape[1] >= 2
                    ):
                        frame = refiner(item, doc, frame, table_markdown)
                except Exception as e:
                    logger.warning(
                        f"{file_path}: Tabelle {len(parts) + 1} nicht extrahierbar: {e}"
                    )
                    gap = True
                    continue
                parts.append(
                    {
                        "markdown": table_markdown,
                        "frame": frame,
                        "geo": table_geometry(item, doc),
                        "gap": gap,
                    }
                )
                gap = False
        finally:
            if refiner is not None:
                refiner.close()

        # 2. Ueber Seitengrenzen laufende Tabellen zu einer zusammenfassen.
        groups: list[list[dict]] = []
        for part in parts:
            last = groups[-1][-1] if groups else None
            if (
                last is not None
                and not part["gap"]
                and is_continuation(
                    last["frame"], last["geo"], part["frame"], part["geo"]
                )
            ):
                groups[-1].append(part)
            else:
                groups.append([part])

        replacements: dict[str, str] = {}
        for number, group in enumerate(groups, start=1):
            merged = len(group) > 1
            try:
                frame = (
                    merge_frames([p["frame"] for p in group])
                    if merged
                    else group[0]["frame"]
                )
                table_markdown = (
                    frame.to_markdown(index=False) if merged else group[0]["markdown"]
                )
                info = table_sink(frame, number)
            except Exception as e:
                logger.warning(f"{file_path}: Tabelle {number} nicht extrahierbar: {e}")
                continue
            if info is None or any(p["markdown"] not in markdown for p in group):
                continue
            marker = TABLE_MARKER.format(number)
            markdown = markdown.replace(group[0]["markdown"], marker, 1)
            for extra in group[
                1:
            ]:  # weitere Seiten der Tabelle stehen nur noch in der CSV
                markdown = markdown.replace(extra["markdown"], "", 1)
            pages = (group[0]["geo"].page, group[-1]["geo"].page) if merged else None
            replacements[marker] = render_reference(info, number, table_markdown, pages)
        for marker, text in replacements.items():
            markdown = markdown.replace(marker, text, 1)
        return markdown

    def _table_refiner(self, file_path: str):
        """Aufrufbares Objekt fuer den zweiten Durchgang per Vision-Modell oder ``None`` (nur PDF, nur wenn aktiv)."""
        if not str(file_path).lower().endswith(".pdf") or not _vision_enabled():
            return None
        from app.ingest.table_vision import (
            DEFAULT_MAX_ROW_DEVIATION,
            PageRenderer,
            refine_table,
        )
        from app.ingest.tasks import load_task

        try:
            cfg = load_task("tables")
        except Exception:
            return None
        if not cfg["enabled"]:
            return None
        if self._table_model is None:
            self._table_model = VisionLLM("tables")
        renderer = PageRenderer(file_path)
        deviation = cfg.get("max_row_deviation") or DEFAULT_MAX_ROW_DEVIATION
        model, prompt = self._table_model, cfg["prompt"]

        class _Refiner:
            def __call__(self, item, doc, frame, markdown):
                fixed = refine_table(
                    item, doc, frame, markdown, renderer, model, prompt, deviation
                )
                return frame if fixed is None else fixed

            def close(self):
                renderer.close()

        return _Refiner()
