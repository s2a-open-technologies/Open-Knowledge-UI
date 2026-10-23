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
Rechengeraet fuer Embedding und Reranking.

- NVIDIA-GPU (CUDA) wenn vorhanden,
- Apple Silicon (MPS, Metal) auf dem Mac,
- sonst CPU.

Ueberschreibbar mit der Umgebungsvariable ``RAG_DEVICE`` (``cuda``, ``mps``, ``cpu``).
Fuer ONNX-Modelle (fastembed) nutzt ``onnx_providers`` unter macOS zusaetzlich CoreML.
"""

import os
import sys
from functools import lru_cache
from logging import getLogger

logger = getLogger(__name__)


@lru_cache(maxsize=1)
def torch_device() -> str:
    forced = (os.getenv("RAG_DEVICE") or "").strip().lower()
    if forced:
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception as e:  # torch fehlt oder ist defekt: CPU
        logger.debug(f"Geraeteerkennung fehlgeschlagen: {e}")
    return "cpu"


def onnx_providers() -> list[str] | None:
    """ONNX-Runtime-Provider (CoreML auf dem Mac); ``None`` = Standard der Bibliothek."""
    if os.getenv("RAG_DEVICE", "").strip().lower() == "cpu":
        return ["CPUExecutionProvider"]
    if sys.platform != "darwin":
        return None
    try:
        import onnxruntime

        if "CoreMLExecutionProvider" in onnxruntime.get_available_providers():
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    except Exception as e:
        logger.debug(f"ONNX-Provider-Erkennung fehlgeschlagen: {e}")
    return None
