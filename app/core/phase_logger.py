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
Phase Logger for tracking Agent execution phases
"""

import logging
import os
import time
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    USER_INPUT = "USER_INPUT"
    AGENT_START = "AGENT_START"
    RETRIEVAL = "RETRIEVAL"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    LLM_CALL = "LLM_CALL"
    VISION_PROCESSING = "VISION_PROCESSING"
    CONTEXT_ENRICHMENT = "CONTEXT_ENRICHMENT"
    STREAMING = "STREAMING"
    AGENT_RESPONSE = "AGENT_RESPONSE"
    DOCUMENT_PROCESSING = "DOCUMENT_PROCESSING"


class PhaseLogger:
    """Phasen-Log. Ohne ``level`` ist ein Eintrag DEBUG (Ablaufdetail); sichtbar sind
    standardmaessig nur Zusammenfassungen (INFO) sowie Warnungen und Fehler. Der
    Schwellwert kommt aus ``LOG_LEVEL`` (nur env, Default INFO). Nutzerinhalte
    (Fragen, Suchbegriffe, SQL) gehoeren nie in eine Meldung."""

    def __init__(self, name: str = "PhaseLogger"):
        self.name = name
        self.logger = logging.getLogger(name)

        if not self.logger.handlers:
            threshold = getattr(
                logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO
            )
            self.logger.setLevel(threshold)
            handler = logging.StreamHandler()
            handler.setLevel(threshold)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(phase)s] %(message)s | Duration: %(duration)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    @contextmanager
    def phase(self, phase: Phase, message: str, level: int = logging.DEBUG, **kwargs):
        start_time = time.time()
        self._log(phase, message, 0.0, level, **kwargs)
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._log(phase, message, duration, level, **kwargs, end=True)

    def log_phase(
        self,
        phase: Phase,
        message: str,
        duration: float | None = None,
        level: int = logging.DEBUG,
        **kwargs,
    ):
        self._log(phase, message, duration, level, **kwargs)

    def _log(
        self,
        phase: Phase,
        message: str,
        duration: float | None,
        level: int = logging.DEBUG,
        **kwargs,
    ):
        if not self.logger.isEnabledFor(level):
            return
        duration_str = f"{duration:.2f}s" if duration is not None else "N/A"

        record = self.logger.makeRecord(
            self.logger.name, level, None, None, message, (), None
        )
        record.phase = phase.value
        record.duration = duration_str

        for key, value in kwargs.items():
            setattr(record, key, value)

        self.logger.handle(record)


phase_logger = PhaseLogger("PhaseLogger")
