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

"""Query-Enhancement: ein LLM erzeugt aus der Nutzerfrage mehrere Suchfragen."""

import re
from collections.abc import Callable

# helper(prompt, config) -> Antworttext des Helfer-LLMs
Helper = Callable[[str, dict], str]

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_BULLET = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")
MAX_QUESTION_CHARS = 300


def build_prompt(question: str, config: dict) -> str:
    """Setzt {frage} und {anzahl} ein (kein str.format: Prompts duerfen Klammern enthalten)."""
    return (
        config["prompt"]
        .replace("{frage}", question)
        .replace("{anzahl}", str(config["n_queries"]))
    )


def parse_questions(raw: str, original: str, limit: int) -> list[str]:
    """Zerlegt die LLM-Antwort in Fragen: eine je Zeile, ohne Aufzaehlungszeichen,
    ohne Duplikate (auch zur Originalfrage), hoechstens ``limit``."""
    seen = {original.strip().lower()}
    questions: list[str] = []
    for line in _THINK.sub("", raw or "").splitlines():
        text = _BULLET.sub("", line).strip().strip('"„“').strip()
        if not text or len(text) > MAX_QUESTION_CHARS or text.lower() in seen:
            continue
        seen.add(text.lower())
        questions.append(text)
        if len(questions) >= limit:
            break
    return questions


def generate(question: str, config: dict, helper: Helper) -> list[str]:
    """Erzeugt bis zu ``n_queries`` zusaetzliche Suchfragen."""
    raw = helper(build_prompt(question, config), config)
    return parse_questions(raw, question, config["n_queries"])
