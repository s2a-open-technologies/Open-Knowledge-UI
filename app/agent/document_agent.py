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
The Document Agent
"""

from langchain.agents import create_agent
from langchain_core.language_models import BaseLanguageModel

# Werkzeuge, die nur mit ausgewaehltem Wissen (oder Chat-Anhaengen) etwas finden koennen
KNOWLEDGE_TOOLS = frozenset(
    {
        "document_search_tool",
        "read_knowledge_index",
        "list_files",
        "preview_data",
        "run_pandas",
    }
)

CONVERSATION_ONLY_HINT = (
    "\n\n[Hinweis: In diesem Chat ist kein Wissen ausgewählt und es sind keine Dateien angehängt. "
    "Antworte und fasse ausschließlich aus dem Gesprächsverlauf zusammen. "
    "Erfinde keine Dokumentinhalte und nenne keine Quellen aus Dokumenten.]"
)


SQL_TOOLS = frozenset({"sql_schema", "sql_query", "sql_find_values"})


def select_tools(tools: list, has_knowledge: bool, has_sql: bool = False) -> list:
    """Ohne Wissen entfallen die Wissens-Werkzeuge, ohne SQL-Quelle die SQL-Werkzeuge; Websuche und Ablagen bleiben."""
    drop = set()
    if not has_knowledge:
        drop |= KNOWLEDGE_TOOLS
    if not has_sql:
        drop |= SQL_TOOLS
    return [t for t in tools if getattr(t, "name", "") not in drop]


class DocumentAgent:
    """Agent for document data analysis"""

    # Fester Basis-Prompt: nur, was Werkzeugaufrufe und die Quellenanzeige sicher macht. Rolle, Sprache, Stil
    # und das Verhalten bei Uebersichts-/Zusammenfassungsfragen stehen in app/agent/behavior.py (UI: Modelle).
    SYSTEM_PROMPT = """
        # Werkzeuge und Quellen (fest)

        ## Verfügbare Tools
        1. **document_search_tool**: Durchsucht die Vektordatenbank nach Textpassagen
        2. **list_files**: Listet verfügbare Excel/CSV-Dateien auf
        3. **preview_data**: Zeigt Spalten und erste Zeilen einer Datei
        4. **run_pandas**: Führt Pandas-Analysen auf Excel/CSV-Dateien aus
        5. **list_summaries**: Listet die Ablagen des Nutzers ("Meine Ablagen") auf
        6. **read_summary**: Liest eine Ablage
        7. **write_summary**: Legt eine neue Ablage an (nur mit dem Inhalt, den der Nutzer ablegen will)
        8. **edit_summary**: Bearbeitet oder ergänzt eine bestehende Ablage
        9. **read_knowledge_index**: Liefert das Inhaltsverzeichnis des Wissens (Dokumente, Gliederung, Tabellen mit Spalten)

        ## Regeln für Tool-Aufrufe
        - **Dokumentenfrage** (PDF/DOCX): `document_search_tool` → Antwort. **Datenanalyse** (Excel/CSV): `list_files` → `preview_data` → `run_pandas` → Antwort. Gruppiere verwandte Suchen in EINEM Aufruf ("Umsatz 2023, Umsatz 2024") und rufe bei klaren Fragen nicht erst `list_files` oder `preview_data` auf.
        - **Neue oder breitere Frage** → neue Tool-Aufrufe, unabhängig von früheren Suchen; bei einer spezifischen Frage zu bereits gesuchten Daten reicht der Gesprächsverlauf.
        - **Verweis-Chunk** (`[STRUKTURIERTE DATEN]` oder `[TABELLE n: Datei …]`): Die Rohdaten liegen NICHT als Text vor. Wechsle für konkrete Werte, Summen und Filter zu `run_pandas` mit dem genannten Dateinamen.
        - **Inhaltsverzeichnis**: Nur zur Orientierung (welches Dokument oder welche Tabelle ist zuständig). Gib es nie vollständig wieder und zähle seinen Inhalt nicht auf, außer der Nutzer verlangt ausdrücklich eine Übersicht. Ist keins mitgegeben und unklar, wo etwas steht, rufe `read_knowledge_index` auf.
        - **Keine Halluzinationen**: Verwende nur Tool-Ergebnisse.
        - **Quellennachweise**: Belege jede Aussage aus Dokumenten direkt im Satz mit der Nummer der Fundstelle in eckigen Klammern, genau wie in den Suchergebnissen (z. B. "Die Kaution beträgt drei Monatsmieten [2]."). Keine eigene Quellenliste am Ende.
    """

    @classmethod
    def compose_prompt(cls, extra: str = "", behavior: str | None = None) -> str:
        """Basis-Prompt (feste Werkzeug-/Quellenregeln) + Standard-Anweisungen (Einstellung in der
        Datenbank) + modellspezifischer Zusatz. Der Basis-Prompt laesst sich nicht ersetzen."""
        if behavior is None:
            from app.agent import behavior as behavior_settings

            behavior = behavior_settings.current()
        parts = [cls.SYSTEM_PROMPT]
        if (behavior or "").strip():
            parts.append(f"        ## Standard-Anweisungen\n        {behavior.strip()}")
        if (extra or "").strip():
            parts.append(f"        ## Zusätzliche Anweisungen\n        {extra.strip()}")
        return "\n\n".join(parts)

    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: list,
        system_prompt: str | None = None,
        middleware: list | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self._agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt or self.SYSTEM_PROMPT,
            middleware=middleware or (),
        )

    @property
    def agent(self):
        """Return the cached compiled agent graph."""
        return self._agent

    def stream(self, input_data: dict, config: dict | None = None):
        """Stream the agent's response."""
        if config is None:
            config = {}
        config["recursion_limit"] = 500
        return self.agent.astream(input_data, config, stream_mode="messages")

    def astream_events(
        self,
        input_data: dict,
        config: dict | None = None,
        version: str = "v2",
        **kwargs,
    ):
        """Stream agent events."""
        if config is None:
            config = {}
        config["recursion_limit"] = 500
        return self.agent.astream_events(
            input_data, config=config, version=version, **kwargs
        )
