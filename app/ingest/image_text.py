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
Bilder aus dem Chat als Text: das Bildmodell liest das Bild (Einlese-Aufgabe ``images``), das
Ergebnis wird als Markdown wie jeder andere Anhang gesplittet, embeddet und durchsucht. So bleibt
der PII-Filter bei Text; das Bild selbst geht nie in den Chat-Kontext.
"""

import base64
import time
from io import BytesIO
from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)

MAX_SIDE = 2048  # laengere Kante in Pixeln; groessere Bilder werden verkleinert (Tokens, Dauer)
RETRIES = 3


class ImageReadError(Exception):
    """Das Bild konnte nicht gelesen werden (Meldung ist fuer den Nutzer gedacht)."""


def encode_image(path: str | Path, max_side: int = MAX_SIDE) -> str:
    """Bild als Base64-JPEG, bei Bedarf verkleinert; Transparenz wird auf Weiss gelegt."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)  # Handyfotos stehen sonst auf der Seite
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                background = Image.new("RGB", img.size, "white")
                background.paste(img, mask=img.split()[-1])
                img = background
            else:
                img = img.convert("RGB")
            if max(img.size) > max_side:
                img.thumbnail((max_side, max_side))
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=90)
    except (UnidentifiedImageError, OSError) as e:
        raise ImageReadError("Das Bild lässt sich nicht öffnen") from e
    return base64.b64encode(buf.getvalue()).decode("ascii")


def enabled() -> bool:
    from app.ingest.tasks import load_task

    return bool(load_task("images")["enabled"])


def image_to_markdown(
    path: str | Path,
    filename: str,
    model=None,
    retries: int = RETRIES,
    delay: float = 2.0,
) -> str:
    """Liest das Bild mit dem Bildmodell und liefert Markdown (Ueberschrift = Dateiname).

    Raises:
        ImageReadError: Aufgabe abgeschaltet, Bild nicht lesbar, Modell fehlgeschlagen oder ohne Text.
    """
    from app.ingest.tasks import load_task
    from app.llm.local_llm import TaskLLM

    cfg = load_task("images")
    if not cfg["enabled"]:
        raise ImageReadError(
            "Bilder im Chat sind abgeschaltet (Administration › Einlesen & Modelle)"
        )
    image = encode_image(path)
    model = model or TaskLLM("images")
    message = model.generate_image_message(image, cfg["prompt"])
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            result = model.generate(message)
            text = (
                result.content if hasattr(result, "content") else str(result)
            ).strip()
            if text:
                return f"# {filename}\n\n{text}\n"
            last = ValueError("leere Antwort")
        except (
            Exception
        ) as e:  # Modell nicht erreichbar, Zeitueberschreitung, kein Bildmodell ...
            last = e
        logger.warning(
            f"Bild {filename}: Versuch {attempt}/{retries} fehlgeschlagen: {str(last)[:120]}"
        )
        if attempt < retries:
            time.sleep(delay * attempt)
    raise ImageReadError(
        f"Das Bildmodell konnte das Bild nicht lesen ({str(last)[:120]})"
    )
