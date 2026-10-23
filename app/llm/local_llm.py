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
Local LLM with vision - supports separate chat and vision models
"""

# Standard library
import json
import time

from langchain_core.messages import HumanMessage

from app.llm.factory import build_chat_llm
from app.utils.phase_logger import Phase, phase_logger


class TaskLLM:
    """LLM fuer eine Einlese-Aufgabe: ``vision`` (Bildbeschreibung) oder ``context`` (Kontext-Anreicherung).

    Modell (``<AUFGABE>_MODEL_REF``, sonst Standardmodell), Temperatur, Max-Tokens und Prompt
    kommen aus der Datenbank (Administration > Einlesen & Modelle) und werden bei jedem Aufruf
    frisch aufgeloest, Aenderungen wirken also sofort.
    """

    def __init__(self, task: str = "vision"):
        self.task = task
        self._llm = None
        self._signature = None

    def _config(self) -> dict:
        from app.ingest.tasks import load_task

        return load_task(self.task)

    def _resolve(self, ref: str | None):
        from app.llm import registry

        try:
            return registry.resolve_model(ref or None)
        except registry.ModelUnavailableError:
            if ref:  # gewaehltes Modell weg: auf das Standardmodell zurueckfallen
                return registry.resolve_model(None)
            raise

    @property
    def llm(self):
        """Nicht-streamendes LLM der Aufgabe (nach Aenderungen neu gebaut)."""
        cfg = self._config()
        ref = cfg["model_ref"]
        if not ref and self.task in ("tables", "images"):
            from app.ingest.tasks import load_task

            ref = load_task(
                "vision"
            )[
                "model_ref"
            ]  # Tabellen und Chat-Bilder brauchen ein Bildmodell: wie die Bildbeschreibung
        resolved = self._resolve(ref)
        params = {**resolved.params, "thinking": "none"}
        if cfg["temperature"] is not None:
            params["temperature"] = cfg["temperature"]
        if cfg["max_tokens"] is not None:
            params["max_tokens"] = cfg["max_tokens"]
        signature = (resolved.signature(), json.dumps(params, sort_keys=True))
        if self._llm is None or self._signature != signature:
            self._llm = build_chat_llm(
                resolved.base_url,
                resolved.api_key,
                resolved.model_id,
                params,
                streaming=False,
            )
            self._signature = signature
            self.vision_model = resolved.model_id
        return self._llm

    vision_model = ""

    def generate_image_message(
        self, image: str, instruction: str | None = None
    ) -> HumanMessage:
        """Generiert die Message für den Call ans Visionmodell"""

        img_size_kb = len(image) * 3 / 4 / 1024

        phase_logger.log_phase(
            Phase.VISION_PROCESSING, f"Bild verarbeiten | Größe: {img_size_kb:.1f}KB"
        )

        if not instruction:
            from app.ingest.tasks import load_task

            instruction = load_task("vision")["prompt"]

        message = HumanMessage(
            content=[
                {"type": "text", "text": instruction},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                },
            ]
        )

        return message

    def generate(self, message: HumanMessage) -> str:
        """Generiert den Text mit VISION-Modell (für Bilder)"""
        llm = self.llm  # loest das Modell auf und setzt vision_model
        start_time = time.time()
        phase_logger.log_phase(
            Phase.LLM_CALL, f"LLM-Call gestartet | Model: {self.vision_model}"
        )

        result = llm.invoke([message])

        duration = time.time() - start_time
        phase_logger.log_phase(
            Phase.LLM_CALL,
            f"LLM-Response erhalten | Model: {self.vision_model}",
            duration=duration,
        )

        return result.content


# Rueckwaertskompatibler Name (Bildbeschreibung)
VisionLLM = TaskLLM
