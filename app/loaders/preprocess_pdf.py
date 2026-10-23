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
PDF-spezifisches Preprocessing via Docling.

Nutzt DocumentConverter mit waehlbarer OCR (OCR_ENGINE, siehe
``app/loaders/ocr.py``) und TableFormer (do_table_structure=True) fuer
saubere Tabellenextraktion. Bilder werden ueber generate_picture_images=True
als PIL-Bilder zur Verfuegung gestellt und in Lesereihenfolge via VisionLLM
beschrieben; Ergebnis ist Markdown.
"""

# Third party
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Custom imports
from app.loaders.ocr import build_ocr_options
from app.loaders.preprocess_base import PreprocessBase


class PreprocessPDF(PreprocessBase):
    """PDF-Preprocessing mit Docling (Layout, Tabellen, OCR, Bilder)."""

    def _build_converter(self) -> DocumentConverter:
        do_ocr, ocr_options = build_ocr_options()
        pipeline_kwargs = {
            "do_ocr": do_ocr,
            "do_table_structure": True,
            "generate_picture_images": True,
            "images_scale": 2,
        }
        if ocr_options is not None:
            pipeline_kwargs["ocr_options"] = ocr_options
        pipeline_options = PdfPipelineOptions(**pipeline_kwargs)
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

    def process_document(self, file_path: str, table_sink=None) -> str:
        """Verarbeitet die PDF und gibt Markdown (inkl. Bildbeschreibungen) zurueck."""
        converter = self._build_converter()
        result = converter.convert(file_path)
        return self.build_markdown_from_docling(result.document, file_path, table_sink)
