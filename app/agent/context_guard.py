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
Schutz des Kontexts innerhalb eines Agent-Laufs (Ebene 2 der Compaction).

Der Agent ruft Werkzeuge in einer Schleife auf; deren Ergebnisse (Dokumentsuche) wachsen im
Kontext. ``SummarizationMiddleware`` fasst den Lauf zusammen, sobald das Eingabe-Budget fast
erreicht ist, ``ToolResultClipMiddleware`` kuerzt ein einzelnes riesiges Ergebnis vorab.
"""

from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware

from app.chat.compaction import Budget, clip_text, estimate_tokens, prompt_for_langchain

CLIP_SHARE = (
    0.4  # ein einzelnes Werkzeug-Ergebnis darf hoechstens so viel vom Budget belegen
)


def count_message_tokens(messages) -> int:
    """Token-Zaehler fuer die Middleware (gleiche vorsichtige Schaetzung wie im Verlauf)."""
    total = 0
    for m in messages:
        content = getattr(m, "content", m if isinstance(m, str) else "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        total += estimate_tokens(str(content)) + 4
        for call in getattr(m, "tool_calls", None) or []:
            total += estimate_tokens(str(call.get("args", "")))
    return total


class ToolResultClipMiddleware(AgentMiddleware):
    """Kuerzt zu grosse Werkzeug-Ergebnisse, bevor sie in den Kontext kommen."""

    def __init__(self, max_tokens: int):
        super().__init__()
        self.max_tokens = max_tokens

    def _clip(self, result):
        content = getattr(result, "content", None)
        if isinstance(content, str) and estimate_tokens(content) > self.max_tokens:
            return result.model_copy(
                update={"content": clip_text(content, self.max_tokens)}
            )
        return result

    def wrap_tool_call(self, request, handler):
        return self._clip(handler(request))

    async def awrap_tool_call(self, request, handler):
        return self._clip(await handler(request))


def build_middleware(
    summary_llm, budget: Budget, prompt: str, overhead_tokens: int
) -> list:
    """Middleware fuer ``create_agent``: Kuerzen einzelner Ergebnisse, Zusammenfassen des Laufs.

    ``overhead_tokens`` (Systemprompt, Werkzeug-Beschreibungen) zaehlt die Middleware nicht mit
    und wird deshalb vom Trigger abgezogen.
    """
    trigger = max(512, budget.trigger - overhead_tokens)
    return [
        ToolResultClipMiddleware(max_tokens=int(budget.usable * CLIP_SHARE)),
        SummarizationMiddleware(
            model=summary_llm,
            trigger=("tokens", trigger),
            keep=("tokens", min(budget.keep, trigger // 2)),
            token_counter=count_message_tokens,
            summary_prompt=prompt_for_langchain(prompt),
        ),
    ]
