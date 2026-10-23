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
Custom tools for the agent
"""

# Standard library
import logging
import os
import time
from logging import getLogger
from pathlib import Path

# Third party
import pandas as pd
from charset_normalizer import from_bytes
from langchain_core.tools.structured import StructuredTool
from pydantic import BaseModel, Field

from app.agent.toolworker import config as worker_config
from app.agent.toolworker.client import ToolWorkerUnavailable, get_client
from app.core.phase_logger import Phase, phase_logger
from app.knowledge.index_doc import read_indexes
from app.rag.format import format_hits, record_sources
from app.rag.retrieval import get_scope_or_none

logger = getLogger(__name__)


# === Pydantic Input Models ===


class DocumentSearchInput(BaseModel):
    query: str = Field(description="Suchanfrage für die Dokumentensuche")


STRUCTURED_SUFFIXES = {".csv", ".xlsx", ".xls"}


class ReadIndexInput(BaseModel):
    """Input model for read_knowledge_index tool (keine Parameter)."""


class ListFilesInput(BaseModel):
    """Input model for list_files tool."""


class PreviewDataInput(BaseModel):
    """Input model for preview_data tool."""

    filename: str = Field(
        description="Name der Datei aus der Dateiliste (z. B. 'sales_2024.xlsx')"
    )
    rows: int = Field(
        default=5, description="Anzahl der Zeilen für die Vorschau (Standard: 5)"
    )


class RunPandasInput(BaseModel):
    """Input model for run_pandas tool."""

    filename: str = Field(
        description="Name der Datei aus der Dateiliste (z. B. 'sales.xlsx')"
    )
    code: str = Field(
        description="""Python/Pandas-Code, der auf dem DataFrame 'df' ausgeführt wird.
        Beispiele:
        - result = df['Umsatz'].sum()
        - result = df.groupby('Kategorie')['Wert'].mean()
        - result = df[df['Datum'] > '2024-01-01']['Umsatz'].sum()
        WICHTIG: Weise das Ergebnis der Variable 'result' zu für die Rückgabe.
        """
    )


class ListSummariesInput(BaseModel):
    """Input model for list_summaries tool."""


class ReadSummaryInput(BaseModel):
    """Input model for read_summary tool."""

    filename: str = Field(
        description="Name der Markdown-Datei aus der Liste (z. B. 'analysis_2024.md')"
    )


class WriteSummaryInput(BaseModel):
    """Input model for write_summary tool."""

    filename: str = Field(
        description="Name für die neue Markdown-Datei (z. B. 'sales_summary.md')"
    )
    content: str = Field(
        description="Der Markdown-Inhalt, der in die Datei geschrieben werden soll"
    )


class EditSummaryInput(BaseModel):
    """Input model for edit_summary tool."""

    filename: str = Field(
        description="Name der existierenden Markdown-Datei (z. B. 'analysis.md')"
    )
    content: str = Field(
        description="Der zusätzliche Inhalt, der hinzugefügt oder eingefügt werden soll"
    )
    mode: str = Field(
        default="append",
        description="""Bearbeitungsmodus:
        - 'append': Fügt Inhalt am Ende der Datei hinzu
        - 'prepend': Fügt Inhalt am Anfang der Datei hinzu
        - 'replace': Ersetzt den gesamten Inhalt der Datei
        """,
    )


# === Tool Descriptions ===

DOCUMENT_SEARCH_DESCRIPTION = """
### Tool Name:
document_search_tool

### Tool Description:
Sucht und extrahiert relevante Textpassagen, Tabelleninhalte und Datenpunkte aus einer Wissensdatenbank, die PDF-, Excel-, CSV- und Docx-Dokumente enthält.

### Wann dieses Tool verwenden:
1. IMMER, wenn Fakten, Zahlen oder spezifische Details aus den hochgeladenen Dokumenten benötigt werden.
2. Wenn ein dokumentübergreifender Vergleich (z. B. Excel vs. PDF) durchgeführt werden muss.
3. Wenn Unklarheiten oder potenzielle Widersprüche in der Datenlage geprüft werden müssen.

### Anleitung für die Query-Formulierung:
- Da die Daten in kleinen Segmenten (Chunks) vorliegen, generiere präzise, kontextreiche Suchbegriffe.
- Kombiniere Entitäten (z. B. Projektname, Datum, Rechnungsnummer) mit dem gesuchten Attribut (z. B. "Gesamtsumme", "Frist", "Status").
- Wenn ein Vergleich zwischen Dateitypen nötig ist, führe nacheinander spezifische Suchen für die jeweiligen Dokumentinhalte durch.
- Vermeide vage Anfragen; nutze Fachbegriffe aus dem Kontext der Quelldokumente.

### Input:
Ein String, der die spezifische Suchanfrage in natürlicher Sprache oder als Schlagwortkombination enthält.

### Output:
Eine Liste von Textsegmenten (Chunks) inklusive Metadaten wie Dateiname, Seitenzahl oder Zeilennummer.
"""

LIST_FILES_DESCRIPTION = """
### Tool Name:
list_files

### Tool Description:
Listet alle verfügbaren Excel- (.xlsx) und CSV-Dateien (.csv) im konfigurierten Datenverzeichnis auf.
Dies ist der erste Schritt für jede Datenanalyse, um zu verstehen, welche Datensätze überhaupt zur Verfügung stehen.

### Wann dieses Tool verwenden:
1. IMMER als ERSTES, wenn der Nutzer nach Zahlen, Statistiken, Berechnungen oder Datenanalysen fragt.
2. Wenn unklar ist, welche Datensätze verfügbar sind.
3. Um die richtige Datei für eine nachfolgende Analyse zu identifizieren.

### Workflow:
1. Rufe `list_files` auf, um alle verfügbaren Dateien zu erhalten.
2. Analysiere die Dateinamen auf Relevanz zur Nutzerfrage (z. B. "sales" für Umsatzfragen).
3. Nutze anschließend `preview_data`, um den Inhalt verdächtiger Dateien zu prüfen.
4. Führe dann mit `run_pandas` die eigentliche Analyse durch.

### Input:
Keine Parameter erforderlich.

### Output:
Eine Liste von Dateinamen als Strings (z. B. ["sales_2024.xlsx", "customers.csv", "products.xlsx"]).
"""

PREVIEW_DATA_DESCRIPTION = """
### Tool Name:
preview_data

### Tool Description:
Zeigt die Spaltennamen und die ersten N Zeilen einer Excel- oder CSV-Datei an.
Ermöglicht eine schnelle Bewertung, ob die Daten für eine spezifische Frage relevant sind,
ohne die gesamte Datei laden und analysieren zu müssen.

### Wann dieses Tool verwenden:
1. NACH dem Aufruf von `list_files`, um den Inhalt einer Datei zu inspizieren.
2. Um zu prüfen, ob bestimmte Spalten (z. B. 'Umsatz', 'Datum', 'Kunde') vorhanden sind.
3. Um das Datenformat zu verstehen, bevor `run_pandas` aufgerufen wird.
4. Wenn der Nutzer fragt, "was in einer Datei steht" oder "welche Spalten es gibt".

### Workflow:
1. Erhalte Dateiliste via `list_files`.
2. Wähle eine möglicherweise relevante Datei.
3. Rufe `preview_data` mit dem Dateinamen auf.
4. Prüfe die Spalten und Beispieldaten auf Relevanz zur Nutzerfrage.
5. Bei Passung: Nutze `run_pandas` für die tiefe Analyse.

### Input:
- filename: Name der Datei aus der Dateiliste (z. B. "sales.xlsx")
- rows: Anzahl der anzuzeigenden Zeilen (Standard: 5)

### Output:
Ein formatierter String mit:
- Liste aller Spaltennamen
- Erste N Zeilen der Daten als Tabelle

Beispiel:
Spalten: ['Datum', 'Produkt', 'Umsatz', 'Menge']

   Datum     | Produkt  | Umsatz | Menge
------------|----------|--------|------
2024-01-01  | ProduktA | 1500   | 10
2024-01-02  | ProduktB | 2300   | 15
"""

RUN_PANDAS_DESCRIPTION = """
### Tool Name:
run_pandas

### Tool Description:
Führt Python/Pandas-Code auf einem geladenen DataFrame aus.
Dieses Tool ermöglicht komplexe Datenanalysen, statistische Berechnungen,
Filterungen, Gruppierungen und Aggregationen – wie ein professioneller Datenanalyst.

### Wann dieses Tool verwenden:
1. NACHDEM du mit `preview_data` die Struktur der Daten verstanden hast.
2. Wenn Berechnungen, Summen, Durchschnitte, Zählungen oder andere Statistiken benötigt werden.
3. Wenn der Nutzer spezifische Fragen zu Zahlen in den Daten stellt.
4. Wenn `document_search_tool` keine ausreichenden Ergebnisse liefert (für numerische Daten).

### Workflow:
1. Liste Dateien auf (`list_files`).
2. Inspiziere relevante Dateien (`preview_data`).
3. Formuliere den Pandas-Code basierend auf Spaltennamen und Analyseziel.
4. Weise das Ergebnis der Variable `result` zu (wichtig!).
5. Rufe `run_pandas` auf und erhalte das analytische Ergebnis.

### Input:
- filename: Name der Datei (z. B. "sales.xlsx")
- code: Python/Pandas-Code, der auf dem DataFrame 'df' ausgeführt wird

### Code-Beispiele:
- Gesamtumsatz: result = df['Umsatz'].sum()
- Durchschnitt: result = df['Wert'].mean()
- Gefilterte Summe: result = df[df['Jahr'] == 2024]['Umsatz'].sum()
- Gruppierung: result = df.groupby('Kategorie')['Umsatz'].sum()
- Count unique: result = df['Kunde'].nunique()
- Mehrere Metriken: result = df.agg({'Umsatz': ['sum', 'mean'], 'Menge': 'sum'})

### WICHTIG:
- Der DataFrame ist als Variable `df` verfügbar.
- Pandas ist als `pd` importiert.
- NumPy ist als `np` importiert.
- Das Ergebnis MUSS der Variable `result` zugewiesen werden.
- Nur lesende Operationen erlaubt (keine Datei-Exports).

### Sicherheit:
Der Code läuft isoliert in einem eigenen Container und wird vorher geprüft. Nicht erlaubt sind:
Importe, Zugriffe auf Namen mit "__", open, eval, exec, compile, getattr, type, Lesen und Schreiben von Dateien
(read_*, to_csv, to_excel, …). Nur df, pd, np und result stehen zur Verfügung.

### Output:
Ein String mit dem Analyseergebnis oder einer Fehlermeldung bei Problemen.
"""

LIST_SUMMARIES_DESCRIPTION = """
### Tool Name:
list_summaries

### Tool Description:
Listet alle Markdown-Dateien aus "Meine Ablagen" des Nutzers auf.
Dies ist der erste Schritt, um zu verstehen, welche Ablagen bereits existieren.

### Wann dieses Tool verwenden:
1. Wenn der Nutzer wissen möchte, welche Ablagen bereits gespeichert wurden.
2. Vor dem Lesen oder Bearbeiten einer Ablage, um den korrekten Dateinamen zu ermitteln.
3. Um den Fortschritt einer Ablage zu überprüfen (weitere Teile hinzugefügt?).

### Workflow:
1. Rufe `list_summaries` auf, um alle Ablagen zu sehen.
2. Wähle eine passende Ablage basierend auf dem Dateinamen.
3. Nutze `read_summary`, um den Inhalt zu lesen.
4. Nutze `edit_summary`, um die Ablage zu erweitern.

### Input:
Keine Parameter erforderlich.

### Output:
Eine Liste von Dateinamen als Strings (z. B. ["analysis_2024.md", "sales_overview.md"]).
"""

READ_SUMMARY_DESCRIPTION = """
### Tool Name:
read_summary

### Tool Description:
Liest den vollständigen Inhalt einer Ablage (Markdown-Datei) aus dem summaries-Verzeichnis.

### Wann dieses Tool verwenden:
1. Um eine existierende Ablage zu überprüfen.
2. Um eine Ablage als Grundlage für eine neue Analyse zu nutzen.
3. Um den aktuellen Stand einer fortlaufenden Ablage zu sehen.
4. Wenn der Nutzer nach dem Inhalt einer bestimmten Ablage fragt.

### Workflow:
1. Liste die Ablagen mit `list_summaries` auf.
2. Wähle die gewünschte Datei.
3. Rufe `read_summary` mit dem Dateinamen auf.
4. Analysiere den Inhalt und fahre fort.

### Input:
- filename: Name der Markdown-Datei (z. B. "analysis_2024.md")

### Output:
Der vollständige Inhalt der Markdown-Datei als String.

### WICHTIG:
- Nur Dateien mit .md Endung erlaubt.
- Pfad-Traversal (../ etc.) ist blockiert.
"""

WRITE_SUMMARY_DESCRIPTION = """
### Tool Name:
write_summary

### Tool Description:
Legt eine neue Markdown-Datei in "Meine Ablagen" des Nutzers an oder überschreibt eine existierende.

### Wann dieses Tool verwenden:
1. Der Nutzer bittet ausdrücklich darum, etwas abzulegen oder zu speichern.
2. Um Erkenntnisse einer Dokumentenanalyse oder eine Zusammenfassung des Gesprächs zu persistieren.
3. Um Teilergebnisse zu speichern, die später zu einem Gesamtbild kombiniert werden.

### WICHTIG - Inhalt:
- Der Inhalt stammt aus dem, was der Nutzer ablegen will: das **Gespräch ODER** eine **Dokumentanalyse**. Vermische beides nicht.
- Nimm nichts auf, was im Gespräch oder in den Suchergebnissen nicht vorkam.
- Einen Quellenabschnitt nur, wenn tatsächlich Dokumente verwendet wurden.

### Workflow:
1. Falls Dokumente analysiert werden sollen: Führe die Analyse durch (document_search, run_pandas, etc.). Bei einer Gesprächs-Zusammenfassung ist keine Suche nötig.
2. Strukturiere die Ergebnisse als Markdown.
3. Speichere mit `write_summary`.
4. Erweitere später mit `edit_summary` bei Bedarf.

### Input:
- filename: Name der neuen Datei (z. B. "quarterly_report.md")
- content: Der Markdown-Inhalt der Datei

### Markdown-Struktur empfohlen:
```markdown
# [Titel]

## Zusammenfassung
[Kurze Zusammenfassung]

## Hauptergebnisse
- Ergebnis 1
- Ergebnis 2

## Details
[Ausführliche Analyse]

## Quellen (nur bei Dokumentanalysen)
- Dokument A
- Tabelle B
```

### Output:
Bestätigung, dass die Datei erstellt wurde, oder eine Fehlermeldung.
"""

EDIT_SUMMARY_DESCRIPTION = """
### Tool Name:
edit_summary

### Tool Description:
Bearbeitet eine existierende Ablage (Markdown-Datei in "Meine Ablagen"), um Inhalt hinzuzufügen oder zu aktualisieren.

### Wann dieses Tool verwenden:
1. Um eine existierende Ablage zu erweitern (append).
2. Um neue Erkenntnisse am Anfang hinzuzufügen (prepend).
3. Um eine Ablage komplett zu aktualisieren (replace).
4. Um mehrere Teilergebnisse zu einem Gesamtbild zusammenzuführen.

### Workflow für Gesamtbild-Erstellung:
1. Lege die Ablage zuerst mit `write_summary` an.
2. Ergänze sie bei Bedarf mit neuen Ergebnissen.
3. Füge neue Abschnitte mit `edit_summary` (mode='append') hinzu.
4. Wiederhole, bis das Gesamtbild vollständig ist.

### Input:
- filename: Name der existierenden Datei (z. B. "analysis.md")
- content: Der hinzuzufügende oder ersetzende Inhalt
- mode: Bearbeitungsmodus:
  * 'append': Fügt Inhalt am Ende hinzu (Standard)
  * 'prepend': Fügt Inhalt am Anfang hinzu
  * 'replace': Ersetzt den gesamten Inhalt

### Beispiele:
- Neue Sektion anhängen: mode='append', content='\n\n## Neue Erkenntnisse\n\n...'
- Wichtige Erkenntnis voranstellen: mode='prepend', content='# WICHTIG: ...\n\n...'
- Komplett neu schreiben: mode='replace', content='# Analyse: ...\n\n...'

### Output:
Bestätigung, dass die Datei bearbeitet wurde, oder eine Fehlermeldung.
"""


class CustomTools:
    """Custom tools for RAG agent with pandas analytics and summary management capabilities."""

    def __init__(
        self,
        llm,
        retriever,
        data_dir=None,
        summaries_dir=None,
    ):
        self.llm = llm
        self.retriever = retriever
        # Pfad -> (mtime, DataFrame): neu laden, wenn die Datei geaendert wurde
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}
        # DB-Aufgeloeste Werte aus init_resources; Fallback auf env (Standalone).
        self.DATA_DIR = data_dir or os.getenv("DATA_DIR", "./data")
        self._summaries_base = summaries_dir

    def _detect_encoding(self, file_path: str) -> str:
        """Erkennt automatisch das Encoding einer Datei (UTF-8, CP1252, Latin-1, etc.)."""
        try:
            with open(file_path, "rb") as file:
                # Stichprobe auf 50kb um Umlaute zu finden
                raw_data = file.read(50000)

            results = from_bytes(raw_data).best()

            if results and results.encoding():
                encoding = results.encoding
                logger.info(
                    f"Encoding erkannt: {encoding} (Confidence: {results.coherence})"
                )
                return encoding
            return "cp1252"
        except Exception as e:
            logger.error(f"Fehler bei Encoding-Erkennung: {e}")
            return "cp1252"

    def _allowed_files(self) -> dict[str, str]:
        """Strukturierte Dateien, die diese Anfrage nutzen darf: Name -> Pfad.

        Nur die Datenordner der durchsuchten Knowledge-Bereiche (siehe
        ``RetrievalScope.data_dirs``). Ohne Chat-Scope (Skripte) gilt ``DATA_DIR``.
        Der Name ist, was die Tools erwarten (``datei.csv`` bzw. ``knowledge_3/datei.csv``);
        alles andere ist nicht erreichbar, auch nicht per ``../``.
        """
        scope = get_scope_or_none()
        dirs = scope.data_dirs if scope is not None else (("", self.DATA_DIR),)
        files: dict[str, str] = {}
        for prefix, directory in dirs:
            folder = Path(directory)
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in STRUCTURED_SUFFIXES:
                    files[f"{prefix}{path.name}"] = str(path)
        return files

    def _get_dataframe(self, filename: str) -> pd.DataFrame:
        """Lädt DataFrame mit Caching und automatischer Encoding-Erkennung."""
        path = self._allowed_files().get(filename)
        if path is None:
            raise FileNotFoundError(f"Datei '{filename}' ist nicht verfügbar")
        mtime = os.path.getmtime(path)
        cached = self._cache.get(path)
        if cached is None or cached[0] != mtime:
            try:
                if path.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(path)
                else:
                    encoding = self._detect_encoding(path)
                    df = pd.read_csv(path, encoding=encoding, on_bad_lines="skip")
            except Exception as e:
                logger.error(f"Fehler beim Laden von {filename}: {e}")
                raise
            self._cache[path] = (mtime, df)
        return self._cache[path][1]

    def list_files(self) -> list[str]:
        """Listet die verfügbaren xlsx/csv Dateien der durchsuchten Knowledge-Bereiche auf."""
        start_time = time.time()
        phase_logger.log_phase(Phase.TOOL_EXECUTION, "Tool: list_files aufgerufen")

        data_files = list(self._allowed_files())
        duration = time.time() - start_time
        if not data_files:
            phase_logger.log_phase(
                Phase.TOOL_EXECUTION,
                "Tool: list_files | Keine Dateien gefunden",
                duration=duration,
            )
            return "Keine Excel- oder CSV-Dateien im ausgewählten Wissen gefunden."

        phase_logger.log_phase(
            Phase.TOOL_EXECUTION,
            f"Tool: list_files | {len(data_files)} Dateien gefunden",
            duration=duration,
        )
        return data_files

    def preview_data(self, filename: str, rows: int = 5) -> str:
        """Zeigt Spalten und erste Zeilen einer Datei an."""
        try:
            df = self._get_dataframe(filename)

            if df.empty:
                return f"Die Datei '{filename}' ist leer."

            columns_str = ", ".join([f"'{col}'" for col in df.columns])
            preview_str = df.head(rows).to_string()

            return f"Spalten ({len(df.columns)}): {columns_str}\n\nAnzahl Zeilen: {len(df)}\n\nVorschau ({rows} Zeilen):\n{preview_str}"

        except Exception as e:
            return f"FEHLER beim Laden von '{filename}': {type(e).__name__}: {e}"

    def run_pandas(self, filename: str, code: str) -> str:
        """Führt Pandas-Code auf einer Datei aus. Die Ausführung läuft im Tool-Worker (eigener Container)."""
        phase_logger.log_phase(Phase.TOOL_EXECUTION, "Tool: run_pandas")
        start_time = time.time()

        path = self._allowed_files().get(filename)
        rel = os.path.relpath(path, self.DATA_DIR) if path else ""
        if not path or rel.startswith(".."):
            return (
                f"FEHLER: Datei '{filename}' konnte nicht geladen werden: "
                f"Datei '{filename}' ist nicht verfügbar"
            )
        limits = worker_config.load()
        try:
            out = get_client().run_pandas(
                rel.replace(os.sep, "/"),
                code,
                timeout_s=limits.timeout_s,
                memory_mb=limits.memory_mb,
                max_output_chars=limits.max_output_chars,
            )
        except ToolWorkerUnavailable as e:
            phase_logger.log_phase(
                Phase.TOOL_EXECUTION,
                "Tool: run_pandas | Worker nicht verfügbar",
                duration=time.time() - start_time,
                level=logging.WARNING,
            )
            return f"FEHLER: Codeausführung nicht verfügbar ({e}) Die Auswertung kann gerade nicht durchgeführt werden."
        phase_logger.log_phase(
            Phase.TOOL_EXECUTION,
            "Tool: run_pandas | fertig",
            duration=time.time() - start_time,
        )
        return out

    def _validate_summary_filename(self, filename: str) -> str:
        """Validiert und bereinigt den Dateinamen für summaries."""
        if not filename:
            raise ValueError("Dateiname darf nicht leer sein.")

        filename = os.path.basename(filename)

        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        if ".." in filename or "/" in filename or "\\" in filename:
            raise ValueError(f"Ungültiger Dateiname: {filename}")

        return filename

    @property
    def SUMMARIES_DIR(self) -> str:
        """Analysen-Ordner des Nutzers dieses Requests (siehe app/summaries.py)."""
        from app.chat import summaries
        from app.rag.retrieval import get_scope_or_none

        scope = get_scope_or_none()
        return str(summaries.user_dir(scope.user_id if scope else None))

    def _ensure_summaries_dir(self):
        """Stellt sicher, dass das summaries-Verzeichnis existiert."""
        os.makedirs(self.SUMMARIES_DIR, exist_ok=True)

    def list_summaries(self) -> list[str]:
        """Listet alle Markdown-Dateien im summaries-Verzeichnis auf."""
        self._ensure_summaries_dir()

        if not os.path.exists(self.SUMMARIES_DIR):
            return (
                f"FEHLER: Summaries-Verzeichnis '{self.SUMMARIES_DIR}' existiert nicht."
            )

        files = os.listdir(self.SUMMARIES_DIR)
        md_files = [f for f in files if f.endswith(".md")]

        if not md_files:
            return f"Keine Markdown-Dateien im Verzeichnis '{self.SUMMARIES_DIR}' gefunden."

        return sorted(md_files)

    def read_summary(self, filename: str) -> str:
        """Liest den Inhalt einer Markdown-Datei aus dem summaries-Verzeichnis."""
        try:
            filename = self._validate_summary_filename(filename)
            self._ensure_summaries_dir()

            filepath = os.path.join(self.SUMMARIES_DIR, filename)

            if not os.path.exists(filepath):
                return f"FEHLER: Datei '{filename}' nicht gefunden. Nutze list_summaries, um verfügbare Dateien zu sehen."

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            return f"Datei: {filename}\n\n{content}"

        except ValueError as e:
            return f"FEHLER: {e}"
        except Exception as e:
            return f"FEHLER beim Lesen von '{filename}': {type(e).__name__}: {e}"

    def _track_summary_for_current_user(self, filename: str) -> None:
        """Verknuepft die geschriebene Summary-Datei mit der App, damit sie unter
        "Analysen" erscheint (Single-User).
        """
        try:
            from app.core.database_service import get_db_service
            from app.rag.retrieval import get_scope_or_none

            scope = get_scope_or_none()
            owner = scope.user_id if scope else None
            db = get_db_service()
            existing = {
                s.filename
                for s in db.get_all_summaries()
                if getattr(s, "user_id", None) == owner
            }
            if filename not in existing:
                db.create_summary_file(filename=filename, user_id=owner)
        except Exception as e:
            logger.error(f"Konnte Summary-Zuordnung fuer {filename} nicht anlegen: {e}")

    def write_summary(self, filename: str, content: str) -> str:
        """Erstellt oder überschreibt eine Markdown-Datei im summaries-Verzeichnis."""
        try:
            filename = self._validate_summary_filename(filename)
            self._ensure_summaries_dir()

            if not content or not content.strip():
                return "FEHLER: Inhalt darf nicht leer sein."

            filepath = os.path.join(self.SUMMARIES_DIR, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            self._track_summary_for_current_user(filename)

            logger.info(f"Summary erstellt: {filename}")
            return f"ERFOLG: Datei '{filename}' wurde erstellt im Verzeichnis '{self.SUMMARIES_DIR}'."

        except ValueError as e:
            return f"FEHLER: {e}"
        except Exception as e:
            logger.error(f"Fehler beim Schreiben von {filename}: {e}")
            return f"FEHLER beim Schreiben von '{filename}': {type(e).__name__}: {e}"

    def edit_summary(self, filename: str, content: str, mode: str = "append") -> str:
        """Bearbeitet eine existierende Markdown-Datei."""
        try:
            filename = self._validate_summary_filename(filename)
            self._ensure_summaries_dir()

            valid_modes = ["append", "prepend", "replace"]
            if mode not in valid_modes:
                return f"FEHLER: Ungültiger Modus '{mode}'. Gültige Modi: {', '.join(valid_modes)}"

            if not content or not content.strip():
                return "FEHLER: Inhalt darf nicht leer sein."

            filepath = os.path.join(self.SUMMARIES_DIR, filename)

            if mode == "replace":
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"Summary ersetzt: {filename}")
                return f"ERFOLG: Datei '{filename}' wurde vollständig ersetzt."

            if not os.path.exists(filepath):
                return f"FEHLER: Datei '{filename}' existiert nicht. Nutze write_summary, um eine neue Datei zu erstellen."

            with open(filepath, "r", encoding="utf-8") as f:
                existing_content = f.read()

            if mode == "append":
                new_content = existing_content.rstrip() + "\n\n" + content.lstrip()
            elif mode == "prepend":
                new_content = content.rstrip() + "\n\n" + existing_content.lstrip()

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)

            logger.info(f"Summary bearbeitet ({mode}): {filename}")
            return f"ERFOLG: Inhalt wurde {'angehängt' if mode == 'append' else 'vorangestellt'} an Datei '{filename}'."

        except ValueError as e:
            return f"FEHLER: {e}"
        except Exception as e:
            logger.error(f"Fehler beim Bearbeiten von {filename}: {e}")
            return f"FEHLER beim Bearbeiten von '{filename}': {type(e).__name__}: {e}"

    def get_tools(self) -> list:
        """Gibt alle verfügbaren Tools zurück."""

        # Dokumentsuche: Treffer mit Quellenangabe, mit Logging
        def document_search(query: str) -> str:
            start_time = time.time()
            phase_logger.log_phase(Phase.RETRIEVAL, "Vector-Search gestartet")
            docs = self.retriever.invoke(query)
            phase_logger.log_phase(
                Phase.RETRIEVAL,
                f"Vector-Search abgeschlossen | Results: {len(docs)} Treffer",
                duration=time.time() - start_time,
            )
            return format_hits(docs, record_sources(docs))

        document_search_tool = StructuredTool.from_function(
            func=document_search,
            name="document_search_tool",
            description=DOCUMENT_SEARCH_DESCRIPTION,
            args_schema=DocumentSearchInput,
        )

        # Inhaltsverzeichnis der durchsuchbaren Knowledge-Bereiche
        read_index_tool = StructuredTool.from_function(
            func=lambda: read_indexes(),
            name="read_knowledge_index",
            description=(
                "Liefert das Inhaltsverzeichnis der durchsuchbaren Wissensbereiche: welche Dokumente "
                "es gibt, ihre Gliederung und welche Tabellen/Datendateien (mit Spalten) für run_pandas "
                'bereitstehen. Nutze es bei ausdrücklichen Übersichtsfragen ("welche Dokumente gibt es?") und wenn '
                "unklar ist, wo etwas steht, bevor du blind suchst. Gib das Ergebnis nicht vollständig wieder, "
                "wenn nur nach einer bestimmten Sache gefragt wurde."
            ),
            args_schema=ReadIndexInput,
        )

        # List Files Tool
        list_files_tool = StructuredTool.from_function(
            func=self.list_files,
            name="list_files",
            description=LIST_FILES_DESCRIPTION,
            args_schema=ListFilesInput,
        )

        # Preview Data Tool
        preview_data_tool = StructuredTool.from_function(
            func=self.preview_data,
            name="preview_data",
            description=PREVIEW_DATA_DESCRIPTION,
            args_schema=PreviewDataInput,
        )

        # Run Pandas Tool
        run_pandas_tool = StructuredTool.from_function(
            func=self.run_pandas,
            name="run_pandas",
            description=RUN_PANDAS_DESCRIPTION,
            args_schema=RunPandasInput,
        )

        # List Summaries Tool
        list_summaries_tool = StructuredTool.from_function(
            func=self.list_summaries,
            name="list_summaries",
            description=LIST_SUMMARIES_DESCRIPTION,
            args_schema=ListSummariesInput,
        )

        # Read Summary Tool
        read_summary_tool = StructuredTool.from_function(
            func=self.read_summary,
            name="read_summary",
            description=READ_SUMMARY_DESCRIPTION,
            args_schema=ReadSummaryInput,
        )

        # Write Summary Tool
        write_summary_tool = StructuredTool.from_function(
            func=self.write_summary,
            name="write_summary",
            description=WRITE_SUMMARY_DESCRIPTION,
            args_schema=WriteSummaryInput,
        )

        # Edit Summary Tool
        edit_summary_tool = StructuredTool.from_function(
            func=self.edit_summary,
            name="edit_summary",
            description=EDIT_SUMMARY_DESCRIPTION,
            args_schema=EditSummaryInput,
        )

        # Tool-Liste zusammenstellen
        tools = [
            document_search_tool,
            read_index_tool,
            list_files_tool,
            preview_data_tool,
            run_pandas_tool,
            list_summaries_tool,
            read_summary_tool,
            write_summary_tool,
            edit_summary_tool,
        ]

        # Die Websuche kommt nicht hierher: sie wird pro Modell zugeschaltet
        # (app/agent/tools/web_access.py, app/api/routes.py).
        return tools
