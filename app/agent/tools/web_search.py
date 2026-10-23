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
Web-Suche Tool für SearXNG Metasuchmaschine
"""

import logging
import random
import time
from logging import getLogger

import requests
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel, Field

from app.core.phase_logger import Phase, phase_logger

logger = getLogger(__name__)

# Realistische Browser User-Agent Rotation
_BROWSER_UAS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]


def _get_headers() -> dict:
    """Generiert realistische Browser-Header für SearXNG-Requests."""
    return {
        "User-Agent": random.choice(_BROWSER_UAS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }


class WebSearchInput(BaseModel):
    """Input model für web_search_tool."""

    query: str = Field(
        description="Die Suchanfrage in natürlicher Sprache (z.B. 'Was ist die Hauptstadt von Frankreich?')"
    )
    num_results: int = Field(
        default=10, description="Anzahl der Suchergebnisse (Standard: 10, Maximum: 50)"
    )
    categories: str = Field(
        default="general",
        description="Durchsuchbare Kategorien, Komma-getrennt (z.B. 'general,news,science')",
    )


WEB_SEARCH_DESCRIPTION = """
### Tool Name:
web_search_tool

### Tool Description:
Durchsucht das Web mittels SearXNG Metasuchmaschine nach Informationen, die nicht in der lokalen
Vektordatenbank oder den hochgeladenen Dokumenten vorhanden sind. Dieses Tool dient als Ergänzung
zum RAG-System: Immer wenn eine Nutzerfrage Wissen erfordert, das weder in den Dokumenten noch
im internen Wissen des Modells abgebildet ist, kann dieses Tool externe Quellen abrufen.

### Wann dieses Tool verwenden:
1. Wenn die Nutzerfrage Themen betrifft, die **nicht in den hochgeladenen Dokumenten** enthalten sind.
2. Wenn die lokale Vektordatenbank (`document_search_tool`) **keine passenden Ergebnisse** liefert.
3. Wenn aktuelle, tagesaktuelle oder sehr spezifische Informationen benötigt werden, die das RAG-System nicht abdeckt.
4. Zur Ergänzung und Validierung von Fakten, bei denen die internen Quellen lückenhaft sein könnten.

### Workflow:
1. Prüfe zuerst, ob `document_search_tool` die Frage beantworten kann.
2. Nur wenn die lokalen Dokumente keine ausreichende Antwort liefern, formuliere eine präzise Suchanfrage.
3. Rufe `web_search_tool` mit der Query auf.
4. Analysiere die Suchergebnisse und extrahiere relevante Informationen.
5. Kombiniere die externen Ergebnisse mit dem Kontext aus den lokalen Dokumenten.

### Input:
- query: Die Suchanfrage (präzise, mit Keywords)
- num_results: Anzahl Ergebnisse (Standard: 10, Maximum: 50)
- categories: Kategorien (z.B. 'general,news,science,tech')

### Output:
Ein formatierter String mit:
- Titel der Suchergebnisse
- URL der Quelle
- Kurze Zusammenfassung/Content-Snippet
- Veröffentlichtes Datum (falls verfügbar)

### Beispiel-Output:
```
Ergebnisse für 'Hauptstadt Frankreich Einwohnerzahl':

1. **Frankreich – Wikipedia**
   URL: https://de.wikipedia.org/wiki/Frankreich
   Zusammenfassung: Frankreich ist ein Land in Westeuropa mit der Hauptstadt Paris...

2. **Paris: Bevölkerung, Fläche & Geschichte**
   URL: https://example.com/paris-daten
   Veröffentlicht: 2024-03-10
   Zusammenfassung: Paris hat rund 2,1 Millionen Einwohner im Stadtgebiet...
```

### Bei leeren Ergebnissen (WICHTIG!):
- Wenn die Rückmeldung "Keine Suchergebnisse für ..." lautet, **muss** du die Query ändern und erneut versuchen.
- Versuche bis zu 3 verschiedene Formulierungsansätze:
  1. **Breiter/formaler**: Statt "CEO von Apple reagiert auf Krise" → "Apple CEO Strategie"
  2. **Andere Keywords**: Statt Personennamen → Firmenname/Thema (z.B. "Apple Krisenmanagement")
  3. **Englisch versuchen**: "Apple CEO crisis management" → oft bessere Ergebnisse
- **Nie** eine leere Antwort zurückgeben — wenn alle 3 Versuche leer sind, antworte:
  "Für diese spezifische Anfrage wurden keine öffentlichen Informationen gefunden."

### WICHTIG:
- Nutze dieses Tool **nur dann**, wenn die lokalen Dokumente und das interne Wissen nicht ausreichen.
- SearXNG durchsucht multiple Quellen gleichzeitig (Google, Bing, Wikipedia, etc.).
- Ergebnisse sind bereits nach Relevanz sortiert.
- Bei Zeitangaben immer das Veröffentlichungsdatum der Quelle nennen.
- Formuliere die Query neutral und faktenbasiert, nicht als Nachrichten-Suche.
"""


def create_web_search_tool(
    searxng_url: str, default_categories: str = "general", pii_filter_client=None
) -> callable:
    """
    Erstellt ein web_search_tool für die gegebene SearXNG-Instanz.

    Args:
        searxng_url: URL der SearXNG-Instanz
        default_categories: Standard-Kategorien für die Suche
        pii_filter_client: Optionaler PII-Filter-Client zum Filtern von Queries

    Returns:
        Strukturiertes Tool für LangChain
    """

    def _web_search(
        query: str, num_results: int = 10, categories: str | None = None
    ) -> str:
        """Führt Web-Suche via SearXNG durch."""

        start_time = time.time()

        # Mit PII-Filter wird die Query vor dem Versand maskiert. Schlaegt das fehl, wird NICHT
        # unmaskiert gesucht: die Suche verlaesst das System, die Maskierung ist ihr Zweck.
        search_query = query
        if pii_filter_client:
            import re

            from app.chat.pii_client import mapping_pairs

            try:
                filter_result = pii_filter_client.filter_text(query)
            except Exception as e:
                logger.warning(f"PII-Filter für Web-Suche fehlgeschlagen: {e}")
                filter_result = None
            if filter_result is None or filter_result.error:
                from app.chat.pii_session import unavailable_message

                return (
                    "FEHLER: Die Suche wurde abgebrochen, weil die Suchanfrage nicht maskiert "
                    f"werden konnte ({unavailable_message(filter_result.error if filter_result else 'nicht erreichbar')}). "
                    "Beantworte die Frage ohne Websuche."
                )
            pairs = mapping_pairs(filter_result.mapping)
            if pairs:
                # PII-Werte direkt aus der Query entfernen (nicht durch [NAME_1] ersetzen)
                for _placeholder, value in pairs:
                    search_query = search_query.replace(value, "")
                search_query = re.sub(r"\s+", " ", search_query).strip()
                logger.info(f"Query PII-bereinigt: '{query}' → '{search_query}'")
            elif filter_result.filtered != query:
                # Einseitiger Dienst ohne Mapping: den maskierten Text suchen, nie das Original
                search_query = filter_result.filtered
            if not search_query.strip():
                return "FEHLER: Die Suchanfrage bestand nur aus personenbezogenen Angaben und wurde nicht gesendet."

        phase_logger.log_phase(
            Phase.TOOL_EXECUTION, f"Tool: web_search | {len(search_query)} Zeichen"
        )

        if not searxng_url:
            phase_logger.log_phase(
                Phase.TOOL_EXECUTION,
                "Tool: web_search | Fehler: Keine SearXNG URL konfiguriert",
                duration=0.0,
                level=logging.WARNING,
            )
            return "FEHLER: SearXNG-URL ist nicht konfiguriert. Bitte im Settings-Panel 'SEARCH_SEARXNG_URL' setzen."

        # Num_results auf 50 begrenzen
        num_results = min(max(1, num_results), 50)
        search_categories = categories or default_categories

        # SearXNG API Endpoint
        search_url = f"{searxng_url.rstrip('/')}/search"

        try:
            response = requests.get(
                search_url,
                params={
                    "q": search_query,
                    "format": "json",
                    "categories": search_categories,
                    "pageno": 1,
                },
                headers=_get_headers(),
                timeout=10.0,
            )

            if response.status_code != 200:
                phase_logger.log_phase(
                    Phase.TOOL_EXECUTION,
                    f"Tool: web_search | HTTP {response.status_code}",
                    duration=0.0,
                    level=logging.WARNING,
                )
                return f"FEHLER: SearXNG API returned HTTP {response.status_code}"

            data = response.json()
            results = data.get("results", [])

            if not results:
                duration = time.time() - start_time
                phase_logger.log_phase(
                    Phase.TOOL_EXECUTION,
                    "Tool: web_search | Keine Ergebnisse gefunden",
                    duration=duration,
                )
                return f"Keine Suchergebnisse für '{search_query}' gefunden."

            # Ergebnisse formatieren
            formatted = [f"Ergebnisse für '{search_query}':\n"]

            for i, result in enumerate(results[:num_results], 1):
                title = result.get("title", "Ohne Titel")
                url = result.get("url", "")
                content = result.get("content", "")
                published_date = result.get("publishedDate", "")

                entry = f"{i}. **{title}**\n"
                entry += f"   URL: {url}\n"
                if published_date:
                    entry += f"   Veröffentlicht: {published_date}\n"
                if content:
                    entry += (
                        f"   Zusammenfassung: {content[:200]}...\n"
                        if len(content) > 200
                        else f"   Zusammenfassung: {content}\n"
                    )

                formatted.append(entry)

            duration = time.time() - start_time
            phase_logger.log_phase(
                Phase.TOOL_EXECUTION,
                f"Tool: web_search | {len(results)} Ergebnisse gefunden",
                duration=duration,
            )

            return "\n".join(formatted)

        except requests.exceptions.Timeout:
            phase_logger.log_phase(
                Phase.TOOL_EXECUTION,
                "Tool: web_search | Timeout",
                duration=0.0,
                level=logging.WARNING,
            )
            return "FEHLER: SearXNG-Request timeout nach 10 Sekunden."

        except requests.exceptions.RequestException as e:
            phase_logger.log_phase(
                Phase.TOOL_EXECUTION,
                f"Tool: web_search | Fehler: {e}",
                duration=0.0,
                level=logging.WARNING,
            )
            return f"FEHLER: Verbindung zu SearXNG fehlgeschlagen: {e}"

        except Exception as e:
            phase_logger.log_phase(
                Phase.TOOL_EXECUTION,
                f"Tool: web_search | Unerwarteter Fehler: {e}",
                duration=0.0,
                level=logging.ERROR,
            )
            return f"FEHLER: {type(e).__name__}: {e}"

    return StructuredTool.from_function(
        func=_web_search,
        name="web_search_tool",
        description=WEB_SEARCH_DESCRIPTION,
        args_schema=WebSearchInput,
    )
