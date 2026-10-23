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
Zweiter Durchgang fuer Tabellen aus PDFs: das Vision-Modell liest das Bild der Tabelle.

Docling (TableFormer) ordnet Zellen bei verbundenen Kopfzellen, mehrzeiligen Zellen oder dicht
gesetzten Zahlen manchmal falsch zu. Deshalb bekommt das Vision-Modell das Bild der Tabelle plus den
Docling-Entwurf und gibt die geprueft-korrigierte Tabelle als Markdown zurueck. Nur eine plausible
Antwort wird uebernommen (gleiche Spaltenstruktur, Zeilenzahl nahe am Entwurf), sonst bleibt es beim
Docling-Ergebnis. Das Bild wird nur fuer den Tabellenausschnitt erzeugt (pypdfium2), nicht fuer alle Seiten.
"""

import base64
import re
from io import BytesIO
from logging import getLogger

logger = getLogger(__name__)

# Rand um die Tabelle in PDF-Punkten und Render-Massstab
MARGIN_PT = 8
SCALE = 2
DEFAULT_MAX_ROW_DEVIATION = 30  # Prozent


def parse_markdown_table(text: str):
    """Markdown-Tabelle -> DataFrame (Kopfzeile = Spaltennamen) oder ``None``, wenn sie nicht einheitlich ist."""
    import pandas as pd

    text = re.sub(
        r"^```[a-z]*\s*|\s*```$", "", (text or "").strip(), flags=re.IGNORECASE
    )
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return None

    def cells(line: str) -> list[str]:
        inner = line.strip()[1:]
        if inner.endswith("|") and not inner.endswith("\\|"):
            inner = inner[:-1]
        return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", inner)]

    header = cells(lines[0])
    if not all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells(lines[1])):
        return None
    rows = [cells(ln) for ln in lines[2:]]
    width = len(header)
    if width < 2 or len(rows) < 1 or any(len(r) != width for r in rows):
        return None
    return pd.DataFrame(rows, columns=header)


def is_plausible(
    fixed, draft, max_deviation_percent: int = DEFAULT_MAX_ROW_DEVIATION
) -> bool:
    """Weicht die korrigierte Tabelle zu stark vom Entwurf ab, ist eher etwas weggelassen oder erfunden worden."""
    if fixed is None or fixed.shape[0] < 1 or fixed.shape[1] < 2:
        return False
    rows, draft_rows = fixed.shape[0], max(1, draft.shape[0])
    if abs(rows - draft_rows) > max(1, draft_rows * max_deviation_percent / 100):
        return False
    cols, draft_cols = fixed.shape[1], max(1, draft.shape[1])
    return abs(cols - draft_cols) <= max(1, draft_cols * max_deviation_percent / 100)


class PageRenderer:
    """Rendert Tabellenausschnitte aus einer PDF; haelt hoechstens eine Seite im Speicher."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self._pdf = None
        self._page_no = None
        self._image = None

    def _page_image(self, page_no: int):
        if self._pdf is None:
            import pypdfium2 as pdfium

            self._pdf = pdfium.PdfDocument(self.pdf_path)
        if self._page_no != page_no:
            self._image = (
                self._pdf[page_no - 1].render(scale=SCALE).to_pil().convert("RGB")
            )
            self._page_no = page_no
        return self._image

    def crop(self, page_no: int, bbox, page_height: float):
        """``bbox``: Docling-BoundingBox (beliebiger Ursprung); liefert das PIL-Bild des Ausschnitts."""
        box = bbox.to_top_left_origin(page_height=page_height)
        image = self._page_image(page_no)
        left = max(0, int((box.l - MARGIN_PT) * SCALE))
        top = max(0, int((box.t - MARGIN_PT) * SCALE))
        right = min(image.width, int((box.r + MARGIN_PT) * SCALE))
        bottom = min(image.height, int((box.b + MARGIN_PT) * SCALE))
        if right <= left or bottom <= top:
            return None
        return image.crop((left, top, right, bottom))

    def close(self) -> None:
        if self._pdf is not None:
            try:
                self._pdf.close()
            except Exception:
                pass
        self._pdf = self._image = None


def _b64(image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def refine_table(
    item,
    doc,
    draft_df,
    draft_markdown: str,
    renderer: PageRenderer,
    model,
    instruction: str,
    max_deviation_percent: int = DEFAULT_MAX_ROW_DEVIATION,
    retries: int = 2,
):
    """Korrigierte Tabelle (DataFrame) oder ``None`` (dann gilt der Docling-Entwurf)."""
    prov = (getattr(item, "prov", None) or [None])[0]
    if prov is None:
        return None
    page = doc.pages.get(prov.page_no) if hasattr(doc, "pages") else None
    if page is None:
        return None
    image = renderer.crop(prov.page_no, prov.bbox, page.size.height)
    if image is None:
        return None
    text = f"{instruction.strip()}\n\n### Entwurf (kann Fehler enthalten):\n{draft_markdown}"
    message = model.generate_image_message(_b64(image), text)
    for attempt in range(1, retries + 1):
        try:
            answer = model.generate(message)
            answer = answer.content if hasattr(answer, "content") else str(answer)
        except Exception as e:
            logger.warning(
                f"Tabellen-Pruefung: Versuch {attempt}/{retries} fehlgeschlagen: {str(e)[:100]}"
            )
            continue
        fixed = parse_markdown_table(answer)
        if is_plausible(fixed, draft_df, max_deviation_percent):
            return fixed
        logger.warning("Tabellen-Pruefung: Antwort unplausibel, Docling-Tabelle bleibt")
        return None
    return None
