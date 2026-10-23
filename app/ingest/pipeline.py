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
Document ingestion
"""

# Standard library
import glob
import logging
import os
import shutil
import time
import uuid
from logging import getLogger
from pathlib import Path

# Third party imports
from langchain_community.document_loaders import (
    TextLoader,
)
from langchain_qdrant import FastEmbedSparse
from qdrant_client import models as qdrant_models
from tqdm import tqdm

from app.core.database_service import get_db_service
from app.core.phase_logger import Phase, phase_logger
from app.core.qdrant_client import get_qdrant_client
from app.core.settings_service import SettingsService
from app.ingest.indexer import (
    PARSER_VERSION,
    build_points,
    document_metadata,
    save_markdown,
    upsert_points,
)

# Custom imports
from app.ingest.loaders import CustomDOCXLoader, CustomPDFLoader
from app.ingest.tables import (
    column_hints_text,
    delete_tables,
    is_extracted_table,
    make_sink,
)
from app.knowledge.files import IMAGE_EXTS, read_uploader
from app.knowledge.index_doc import refresh_index
from app.llm.local_llm import TaskLLM
from app.rag import model_settings, models_runtime

logger = getLogger(__name__)

DATA_REFERENCE_TYPE = "data_reference"
STRUCTURED_EXTS = {".csv", ".xlsx", ".xls"}


def _detect_encoding(file_path: str) -> str:
    """Encoding einer CSV-Datei erkennen (Fallback cp1252)."""
    import chardet

    try:
        with open(file_path, "rb") as f:
            raw = f.read(10000)
        result = chardet.detect(raw)
        encoding = result.get("encoding") or "cp1252"
        if (result.get("confidence") or 0) < 0.7:
            encoding = "cp1252"
        return encoding
    except Exception:
        return "cp1252"


def _load_dataframe(file_path: str):
    """Lädt eine strukturierte Datei (CSV/XLSX/XLS) als DataFrame."""
    import pandas as pd

    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(
            file_path, encoding=_detect_encoding(file_path), on_bad_lines="skip"
        )
    return pd.read_excel(file_path)


def describe_data_file(file_path: str, display_name: str | None = None) -> str:
    """Erzeugt eine kompakte, suchoptimierte Beschreibung für den Verweis-Chunk
    einer strukturierten Datei.

    Der Chunk sagt dem Modell, WORUM es in der Datei geht und DASS die Details
    per Pandas-Tool abgefragt werden müssen — die Daten selbst werden nicht
    dupliziert. ``display_name`` ist der Name, den das Pandas-Tool braucht
    (relativ zu DATA_DIR, z. B. ``knowledge_3/x.csv``)."""
    filename = display_name or os.path.basename(file_path)
    ext = Path(file_path).suffix.lower().lstrip(".")

    columns: list[str] = []
    n_rows = 0
    sample = ""
    column_hints = ""
    try:
        df = _load_dataframe(file_path)
        n_rows = len(df)
        columns = [str(c) for c in df.columns[:30]]
        sample = df.head(3).to_markdown(index=False)
        column_hints = column_hints_text(df)
    except Exception as e:
        logger.warning(f"Datenvorschau fehlgeschlagen für {filename}: {e}")

    column_list = ", ".join(f"`{c}`" for c in columns) or "unbekannt"
    return (
        f"[STRUKTURIERTE DATEN] Die Datei `{filename}` ({ext.upper()}) ist eine "
        f"Tabelle mit {n_rows} Zeilen und den Spalten: {column_list}.\n"
        f"{column_hints}"
        f"Die Rohdaten sind NICHT als Text verfügbar. Für konkrete Werte, "
        f"Summen, Filter oder Auswertungen zu dieser Tabelle muss das "
        f"Pandas-Tool `run_pandas` mit dem Dateinamen `{filename}` genutzt "
        f"werden (vorab `list_files` / `preview_data`).\n"
        f"Vorschau (erste Zeilen):\n{sample}"
    )


def index_data_references(
    embeddings,
    sparse_model,
    collection_name,
    data_dir,
    name_prefix: str = "",
    knowledge_id: int | None = None,
) -> int:
    """Indexiert je einen schlanken Verweis-Chunk pro strukturierter Datei in
    ``data_dir`` (idempotent: alte Verweis-Punkte werden vorher entfernt).

    ``name_prefix`` wird dem Dateinamen vorangestellt (Knowledge-Unterordner
    relativ zu DATA_DIR).

    Returns:
        Anzahl der geschriebenen Verweis-Chunks.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return 0

    files = sorted(
        p
        for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in STRUCTURED_EXTS
    )
    # Aus Dokumenten extrahierte Tabellen gehoeren zum Dokument: ihr Verweis steht in dessen
    # Chunks, ein eigener Punkt waere ein zweites "Dokument". Alt-Punkte werden entfernt.
    for p in [p for p in files if is_extracted_table(p.name)]:
        try:
            delete_data_reference(collection_name, f"{name_prefix}{p.name}")
        except Exception as e:
            logger.debug(f"Alt-Verweis fuer {p.name} nicht loeschbar: {e}")
    files = [p for p in files if not is_extracted_table(p.name)]
    if not files:
        return 0

    texts: list[str] = []
    metadatas: list[dict] = []
    for file_path in files:
        rel_name = f"{name_prefix}{file_path.name}"
        try:
            texts.append(describe_data_file(str(file_path), display_name=rel_name))
        except Exception as e:
            logger.error(f"Verweis-Beschreibung fehlgeschlagen {file_path.name}: {e}")
            continue
        metadatas.append(
            {
                "source": str(file_path),
                "filename": rel_name,
                "title": rel_name,
                "type": DATA_REFERENCE_TYPE,
                "file_group_id": rel_name,
                "file_type": file_path.suffix.lower().lstrip("."),
                **({"knowledge_id": knowledge_id} if knowledge_id is not None else {}),
            }
        )

    if not texts:
        return 0

    dense_vectors = embeddings.embed_documents(texts)
    sparse_vectors = sparse_model.embed_documents(texts)

    client = get_qdrant_client(collection_name=collection_name)
    try:
        for meta in metadatas:
            try:
                client.delete(
                    collection_name=collection_name,
                    points_selector=qdrant_models.FilterSelector(
                        filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="type",
                                    match=qdrant_models.MatchValue(
                                        value=DATA_REFERENCE_TYPE
                                    ),
                                ),
                                qdrant_models.FieldCondition(
                                    key="filename",
                                    match=qdrant_models.MatchValue(
                                        value=meta["filename"]
                                    ),
                                ),
                            ]
                        )
                    ),
                )
            except Exception as e:
                logger.debug(f"Konnte alte Verweis-Chunks nicht löschen: {e}")

        points = []
        for i in range(len(texts)):
            sv = sparse_vectors[i]
            qdrant_sparse = qdrant_models.SparseVector(
                indices=sv.indices if hasattr(sv, "indices") else sv["indices"],
                values=sv.values if hasattr(sv, "values") else sv["values"],
            )
            points.append(
                qdrant_models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"": dense_vectors[i], "langchain-sparse": qdrant_sparse},
                    payload={
                        "page_content": texts[i],
                        "original_content": texts[i],
                        **metadatas[i],
                    },
                )
            )
        client.upsert(collection_name=collection_name, points=points)
    finally:
        client.close()

    logger.info(f"{len(texts)} Verweis-Chunk(s) für strukturierte Daten ingestet.")
    return len(texts)


def delete_data_reference(collection_name: str, filename: str) -> None:
    """Entfernt den Verweis-Chunk einer strukturierten Datei aus Qdrant."""
    client = get_qdrant_client(collection_name=collection_name)
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="type",
                            match=qdrant_models.MatchValue(value=DATA_REFERENCE_TYPE),
                        ),
                        qdrant_models.FieldCondition(
                            key="filename",
                            match=qdrant_models.MatchValue(value=filename),
                        ),
                    ]
                )
            ),
        )
    finally:
        client.close()


def _ingest_settings() -> tuple[int, int, bool, bool]:
    """(Chunk-Groesse, Ueberlappung, Kontextanreicherung, Original behalten) aus den Einstellungen."""
    try:
        session = get_db_service().get_session()
        try:
            settings = SettingsService(session)
            return (
                settings.get_int("INGEST_CHUNK_SIZE", 1000),
                settings.get_int("INGEST_CHUNK_OVERLAP", 150),
                settings.get_bool("INGEST_CONTEXT_ENRICHMENT", True),
                settings.get_bool("INGEST_KEEP_ORIGINAL", True),
            )
        finally:
            session.close()
    except Exception:
        return 1000, 150, True, True


class DocumentIngestion:
    """Document ingestion class"""

    def __init__(
        self,
        embeddings=None,
        sparse_embeddings=None,
        knowledge: dict | None = None,
        collection_name: str | None = None,
        extra_metadata: dict | None = None,
        enrich_context: bool | None = None,
        track_documents: bool = True,
        files_dir: str | None = None,
        embedding_spec=None,
        attachment_markdown_dir: str | None = None,
    ):
        """
        Args:
            knowledge: Knowledge-Bereich (dict aus ``app.knowledge.service``); ``None``
                steht fuer den Standard-Bereich.
            collection_name: ueberschreibt die Collection (z. B. Chat-Anhaenge).
            extra_metadata: wird jedem Chunk-Payload hinzugefuegt (z. B. session_id).
            enrich_context: LLM-Kontext-Anreicherung der Chunks.
            track_documents: UserDocument anlegen und Datei ins Archiv verschieben.
            files_dir: ueberschreibt den Quellordner der Ingestion.
            embedding_spec: Modell fuer die Vektoren (Standard: das des Knowledge).
            attachment_markdown_dir: Chat-Anhaenge: hier wird das Markdown abgelegt
                (``self.last_markdown_path``), sonst nirgends.
        """
        from app.knowledge.service import get_knowledge, knowledge_paths

        if knowledge is None:
            knowledge = get_knowledge(None)
        if knowledge is None:  # Standalone ohne DB-Eintrag
            knowledge = {
                "id": None,
                "is_default": True,
                "collection_name": "open_knowledge_ui",
            }
        self.knowledge = knowledge
        self.paths = knowledge_paths(knowledge)
        self.files_dir = Path(files_dir) if files_dir else self.paths.files_dir
        self.extra_metadata = extra_metadata or {}
        self.chunk_size, self.chunk_overlap, setting_enrich, self.keep_original = (
            _ingest_settings()
        )
        self.enrich_context = (
            setting_enrich if enrich_context is None else enrich_context
        )
        self.track_documents = track_documents
        self.attachment_markdown_dir = attachment_markdown_dir
        self.last_markdown_path: str | None = None
        # Ergebnis des letzten Laufs fuer die Oberflaeche: nichts schlaegt mehr stumm fehl.
        self.errors: list[dict] = []
        self.warnings: list[str] = []
        self.processed = 0
        self._enrich_failures = 0
        self._enrich_error = ""

        self.collection_name = collection_name or knowledge["collection_name"]

        # Embedding: ein Knowledge wird mit dem Modell erweitert, mit dem es indexiert wurde
        # (sonst mischen sich Vektoren); Chat-Anhaenge uebergeben das aktuelle globale Modell.
        if embeddings is not None:  # vorgebautes Objekt (Tests)
            self.embeddings = embeddings
        else:
            self.embeddings = models_runtime.get_embedder(
                embedding_spec or model_settings.knowledge_spec(knowledge)
            )
        self.dimensions = (
            getattr(self.embeddings, "dimension", None)
            or knowledge.get("embedding_dimension")
            or 384
        )

        # Geteilte Sparse-Instanz aus app.extensions (optional); Fallback, falls
        # DocumentIngestion standalone (ohne uebergebene Instanz) genutzt wird.
        self._shared_sparse = sparse_embeddings

        # Supported file extensions. Strukturierte Daten (CSV/Excel) werden
        # NICHT ingestet — sie liegen als Pandas-Referenz in DATA_DIR und
        # werden ausschliesslich über die run_pandas-Tools genutzt.
        self.loaders = {
            ".pdf": CustomPDFLoader,
            ".docx": CustomDOCXLoader,
            ".doc": CustomDOCXLoader,
            ".odt": CustomDOCXLoader,
            ".md": TextLoader,
        }

        # Qdrant client wird erst bei Bedarf geöffnet
        self._client = None

        # Setup model
        self.model = TaskLLM("vision")  # Bildbeschreibung
        self.context_model = TaskLLM("context")  # Kontext-Anreicherung

        # DATA_DIR: strukturierte Daten (CSV/Excel) liegen hier als
        # Pandas-Referenz. Pro Datei wird nur ein schlanker Verweis-Chunk in
        # die Vektordatenbank geschrieben (siehe _ingest_data_references).
        self.DATA_DIR = str(self.paths.data_dir)

        # Setup the collection (öffnet Client temporär)
        self._setup_collection()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_client()
        return False

    def __del__(self):
        self._close_client()

    def _get_client(self):
        """Öffnet Qdrant Client bei Bedarf (Windows-sicher)"""
        if self._client is None:
            self._client = get_qdrant_client(collection_name=self.collection_name)
        return self._client

    def _close_client(self):
        """Schließt den Client sauber (wichtig für Windows Locks)"""
        if self._client is not None:
            self._client.close()
            self._client = None

    def _setup_collection(self):
        """Setting up the collection"""
        from app.rag.collections import ensure_collection

        ensure_collection(self._get_client(), self.collection_name, self.dimensions)
        self._close_client()

    def _move_processed_file(self, file_path: str):
        """Verschiebt die verarbeitete Datei in den Archiv-Ordner (behaelt die Unterordner-Struktur)"""
        processed_dir = str(self.paths.processed_dir)
        os.makedirs(processed_dir, exist_ok=True)

        if not self.keep_original:
            # Das Markdown ist die Grundlage fuer Neu-Indexieren; das Original wird nicht benoetigt.
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Original verworfen: {os.path.basename(file_path)}")
            return

        if os.path.exists(file_path):
            rel_path = os.path.relpath(file_path, str(self.files_dir))
            target_path = os.path.join(processed_dir, rel_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            try:
                if os.path.exists(target_path):
                    os.remove(target_path)
                shutil.move(file_path, target_path)
                logger.info(f"Verschoben in Archiv: {rel_path}")
            except Exception as e:
                logger.error(f"Konnte Datei nicht verschieben {rel_path}: {e}")

    def _enrich_safe(self, full_document_text: str, chunk_text: str) -> str:
        """Anreicherung, die das Einlesen nie abbricht: bei Fehlern gilt der reine Text."""
        try:
            return self._enrich_with_context(full_document_text, chunk_text)
        except Exception as e:
            self._enrich_failures += 1
            if self._enrich_failures == 1:
                logger.warning(f"Kontext-Anreicherung fehlgeschlagen: {e}")
                self._enrich_error = f"{type(e).__name__}: {e}"[:200]
            return chunk_text

    def _enrich_with_context(self, full_document_text: str, chunk_text: str) -> str:
        """Enrich the chunks for contextual RAG"""
        from langchain_core.messages import HumanMessage

        start_time = time.time()
        phase_logger.log_phase(
            Phase.CONTEXT_ENRICHMENT,
            f"Context-Anreicherung gestartet | Chunk-Text: {len(chunk_text)} Zeichen",
        )

        from app.ingest.tasks import load_task, render_context_prompt

        cfg = load_task("context")
        prompt = render_context_prompt(
            cfg["prompt"], full_document_text[: cfg["excerpt_chars"]], chunk_text
        )

        message = HumanMessage(content=prompt)
        context = self.context_model.generate(message)

        duration = time.time() - start_time
        phase_logger.log_phase(
            Phase.CONTEXT_ENRICHMENT,
            f"Context angereichert | Result: {len(context)} Zeichen",
            duration=duration,
        )

        return f"Kontext: {context}\n\nInhalt: {chunk_text}"

    def _sparse_for_index(self, sparse):
        """BM25-Encoder in der Sprache dieses Index: die des Knowledge, bei Anhaengen die Standardsprache."""
        from app.rag import sparse as sparse_lang

        language = (
            sparse_lang.knowledge_language(self.knowledge)
            if self.track_documents
            else sparse_lang.default_language()
        )
        return sparse_lang.resolve(sparse, language)

    def _load_markdown(
        self, file_path: str, extension: str, table_sink=None
    ) -> str | None:
        """Liest die Datei als Markdown (Docling fuer PDF/Office, Text fuer .md)."""
        if extension == ".pdf":
            docs = list(
                CustomPDFLoader(
                    file_path, model=self.model, table_sink=table_sink
                ).lazy_load()
            )
        elif extension in (".docx", ".doc", ".odt"):
            docs = list(
                CustomDOCXLoader(
                    file_path, model=self.model, table_sink=table_sink
                ).lazy_load()
            )
        elif extension == ".md":
            docs = TextLoader(file_path, autodetect_encoding=True).load()
        elif extension in IMAGE_EXTS:
            from app.ingest.image_text import image_to_markdown

            return image_to_markdown(file_path, os.path.basename(file_path))
        else:
            logger.warning(f"Unbekannter Dateityp: {extension}")
            return None
        return "\n\n".join(d.page_content for d in docs)

    def index_markdown(
        self,
        markdown: str,
        filename: str,
        file_group_id: str,
        sparse_model=None,
        uploaded_by: int | None = None,
        uploaded_at=None,
    ) -> int:
        """Zerlegt gespeichertes Markdown neu und schreibt es in die Collection (Neu-Indexieren).

        Gleicher Weg wie beim Einlesen: splitten, ggf. anreichern, Dense + BM25, Upsert.
        Liefert die Anzahl Chunks (0 = kein Inhalt).
        """
        sparse = self._sparse_for_index(
            sparse_model
            or self._shared_sparse
            or FastEmbedSparse(model_name="Qdrant/bm25")
        )
        points = build_points(
            markdown,
            filename=filename,
            source=filename,
            file_group_id=file_group_id,
            dense=self.embeddings,
            sparse=sparse,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            extra_metadata={
                **document_metadata(
                    filename=filename,
                    knowledge_id=self.knowledge.get("id"),
                    uploaded_by=uploaded_by,
                    uploaded_at=uploaded_at,
                ),
                **self.extra_metadata,
            },
            enrich=self._enrich_safe if self.enrich_context else None,
        )
        if points:
            upsert_points(self._get_client(), self.collection_name, points)
        return len(points)

    def _process_file(self, file_path: str, sparse_model) -> bool:
        """Verarbeitet eine Datei; Fehler werden notiert und raeumen angefangene Tabellen-CSVs weg.

        Eine fehlgeschlagene Datei bleibt im Eingangsordner und kann erneut eingelesen werden.
        """
        filename = os.path.basename(file_path)
        file_group_id = str(uuid.uuid4())
        self._enrich_failures = 0
        try:
            ok = self._process_file_inner(file_path, sparse_model, file_group_id)
        except Exception as e:
            logger.exception(f"Fehler bei {filename}")
            delete_tables(self.paths, self.collection_name, file_group_id)
            self.errors.append(
                {"file": filename, "error": f"{type(e).__name__}: {e}"[:300]}
            )
            return False
        if not ok:
            if not any(err["file"] == filename for err in self.errors):
                self.errors.append(
                    {"file": filename, "error": "Kein lesbarer Inhalt gefunden"}
                )
            return False
        if self._enrich_failures:
            self.warnings.append(
                f"{filename}: Kontext-Anreicherung bei {self._enrich_failures} Abschnitt(en) "
                f"fehlgeschlagen, sie wurden ohne Zusatzkontext indexiert ({self._enrich_error})"
            )
        self.processed += 1
        return True

    def _process_file_inner(
        self, file_path: str, sparse_model, file_group_id: str
    ) -> bool:
        """
        Verarbeitet eine einzelne Datei komplett (sequentiell):
        parsen -> Markdown speichern -> splitten -> embedden -> Qdrant.

        Args:
            file_path: Pfad zur Datei
            sparse_model: BM25 Sparse Embedding Model

        Returns:
            bool: True wenn erfolgreich, False bei Fehler/leer
        """
        filename = os.path.basename(file_path)
        extension = Path(file_path).suffix.lower()
        start_time = time.time()
        phase_logger.log_phase(
            Phase.DOCUMENT_PROCESSING, f"Datei verarbeiten: {filename}"
        )

        # file_group_id (vom Aufrufer): Gruppen-ID aller Chunks dieser Datei, wird als
        # UserDocument.document_id gespeichert (Liste/Loeschen per Payload-Filter).

        # 1. Datei -> Markdown. Strukturierte Daten (CSV/Excel) sind hier bereits
        # ausgeschlossen (siehe self.loaders / DATA_DIR-Referenz).
        # Tabellen werden dabei als CSV im Datenordner des Knowledge abgelegt (nicht bei
        # Chat-Anhaengen: dort gibt es keinen Datenordner).
        table_sink = (
            make_sink(self.paths, Path(filename).stem, file_group_id)
            if self.track_documents
            else None
        )
        try:
            markdown = self._load_markdown(file_path, extension, table_sink)
        except Exception as e:
            logger.error(f"Fehler beim Laden von {filename}: {e}")
            delete_tables(self.paths, self.collection_name, file_group_id)
            if (
                extension in IMAGE_EXTS
            ):  # die Meldung ist fuer den Nutzer gedacht (z. B. Bildmodell nicht erreichbar)
                self.errors.append({"file": filename, "error": str(e)[:300]})
            return False
        if not markdown or not markdown.strip():
            logger.info(f"Kein Inhalt in {filename}")
            delete_tables(self.paths, self.collection_name, file_group_id)
            return False

        # 2. Markdown speichern (Grundlage fuer Neu-Indexieren und die Ansicht im UI).
        markdown_path = None
        if self.track_documents:
            markdown_path = str(
                save_markdown(self.paths.markdown_dir, file_group_id, markdown)
            )
        elif self.attachment_markdown_dir:
            self.last_markdown_path = str(
                save_markdown(
                    self.attachment_markdown_dir, Path(filename).stem, markdown
                )
            )

        uploader = (
            read_uploader(file_path)
            if self.track_documents
            else self.extra_metadata.get("uploaded_by")
        )

        # 3. Splitten, ggf. anreichern, Vektoren berechnen
        phase_logger.log_phase(
            Phase.RETRIEVAL, f"Embeddings berechnen | Datei: {filename}"
        )
        points = build_points(
            markdown,
            filename=filename,
            source=file_path,
            file_group_id=file_group_id,
            dense=self.embeddings,
            sparse=self._sparse_for_index(sparse_model),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            extra_metadata={
                **document_metadata(
                    filename=filename,
                    knowledge_id=self.knowledge.get("id"),
                    uploaded_by=uploader,
                ),
                **self.extra_metadata,
            },
            enrich=self._enrich_safe if self.enrich_context else None,
        )
        if not points:
            logger.info(f"Keine Chunks für {filename}")
            delete_tables(self.paths, self.collection_name, file_group_id)
            return False
        if any("[BILD-BESCHREIBUNG:" in p.payload["original_content"] for p in points):
            phase_logger.log_phase(
                Phase.VISION_PROCESSING, f"Bild-Kontext extrahiert | Datei: {filename}"
            )

        # 4. Upload Qdrant
        upsert_points(self._get_client(), self.collection_name, points)

        if self.track_documents:
            # 5. Datei verschieben (Dokumente -> Archiv)
            self._move_processed_file(file_path)

            # 6. Dokument-Zuordnung anlegen, damit die Datei-Liste das ingestierte
            # Dokument findet und es per file_group_id gelöscht werden kann.
            try:
                from app.core.database_service import get_db_service

                get_db_service().add_document(
                    document_id=file_group_id,
                    filename=filename,
                    knowledge_id=self.knowledge.get("id"),
                    markdown_path=markdown_path,
                    parser_version=PARSER_VERSION,
                    uploaded_by=uploader,
                )
            except Exception as e:
                logger.error(
                    f"Fehler beim Anlegen des UserDocument-Eintrags fuer {filename}: {e}"
                )
            read_uploader(file_path, consume=True)  # Begleitdatei im Eingang aufraeumen

        duration = time.time() - start_time
        phase_logger.log_phase(
            Phase.DOCUMENT_PROCESSING,
            f"Datei fertig: {filename} | Dauer: {duration:.2f}s",
        )

        return True

    def ingest_documents(self):
        """Sequentielle Ingestion pro Datei"""
        phase_logger.log_phase(
            Phase.DOCUMENT_PROCESSING,
            "Dokument-Ingestion gestartet",
            level=logging.INFO,
        )

        # Dateien sammeln
        all_files = []
        for extension in self.loaders:
            pattern = os.path.join(str(self.files_dir), f"**/*{extension}")
            files = glob.glob(pattern, recursive=True)
            all_files.extend(files)

        if not all_files:
            logger.info(f"Keine Dokumente in {self.files_dir} gefunden.")

        sparse_model = self._shared_sparse or FastEmbedSparse(model_name="Qdrant/bm25")

        if all_files:
            pbar = tqdm(all_files, desc="Processing Files", unit="file")

            success_count = 0
            error_count = 0

            for file_path in pbar:
                filename = os.path.basename(file_path)
                pbar.set_description(f"Processing: {filename}")

                try:
                    success = self._process_file(file_path, sparse_model)
                    if success:
                        success_count += 1
                        pbar.write(f"✓ {filename} verarbeitet")
                    else:
                        error_count += 1
                        pbar.write(f"✗ {filename} fehlgeschlagen (keine Chunks)")
                except Exception as e:
                    error_count += 1
                    self.errors.append(
                        {"file": filename, "error": f"{type(e).__name__}: {e}"[:300]}
                    )
                    logger.error(f"Fehler bei {filename}: {e}")
                    pbar.write(f"✗ {filename} fehlgeschlagen: {str(e)[:50]}")
                    continue

            logger.info(
                f"{success_count} Dateien erfolgreich verarbeitet, {error_count} Fehler."
            )

        # Verweis-Chunks für strukturierte Daten (CSV/Excel) erzeugen, damit das
        # Modell weiß, welche Tabellen es per Pandas-Tool abfragen kann.
        try:
            if self.track_documents:
                index_data_references(
                    embeddings=self.embeddings,
                    sparse_model=self._sparse_for_index(sparse_model),
                    collection_name=self.collection_name,
                    data_dir=str(self.paths.data_dir),
                    name_prefix=self.paths.name_prefix,
                    knowledge_id=self.knowledge.get("id"),
                )
        except Exception as e:
            logger.error(
                f"Verweis-Ingestion für strukturierte Daten fehlgeschlagen: {e}"
            )

        if self.track_documents:
            refresh_index(self.knowledge.get("id"))

        self._close_client()
        phase_logger.log_phase(
            Phase.DOCUMENT_PROCESSING,
            "Dokument-Ingestion abgeschlossen",
            level=logging.INFO,
        )
