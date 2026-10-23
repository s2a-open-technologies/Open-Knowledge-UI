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
Laden von Embedding-Modellen, Vectorstore, Retriever und Agent.

``load_resources()`` ist eine reine Funktion — sie laedt alle KI-Ressourcen
und speichert sie in ``app.extensions``. Bei Fehlern wird eine Exception
ausgeloest; der Aufrufer (``__init__.py``) entscheidet ueber Sync/Async,
Event-Handling und Fehler-Logging.
"""

import logging

from app.llm.local_llm import VisionLLM
from app.rag.retrieval import MultiCollectionRetriever
from app.rag.sparse import SparseRouter
from app.sqlsource.tools import SqlTools
from app.tools.custom_tools import CustomTools

logger = logging.getLogger(__name__)


def load_resources(
    app,
    qdrant_client,
    collection_name,
    retriever_k,
    retriever_fetch_k,
    retriever_lambda,
    retriever_threshold,
    data_dir,
    summaries_dir,
):
    """Laedt Embedding-Modelle, Vectorstore, Retriever und Agent.

    Alle Ressourcen werden in ``app.extensions`` abgelegt. Bei Fehlern
    (z.B. Modell-Download gescheitert) wird eine Exception ausgeloest.

    Args:
        app: Flask-Applikation (fuer ``app.extensions``).
        qdrant_client: Bereits verbundener Qdrant-Client.
        collection_name: Name der Qdrant-Collection.
        retriever_k: Anzahl MMR-Ergebnisse.
        retriever_fetch_k: Anzahl Dokumente zum Abrufen (vor MMR).
        retriever_lambda: MMR-Balance (0=divers, 1=relevant).
        retriever_threshold: Mindest-Ähnlichkeitsscore.
        data_dir: Verzeichnis fuer CSV-/Excel-Dateien.
        summaries_dir: Verzeichnis fuer Markdown-Analysen.
    """
    # Sparse (BM25): Sprache je Knowledge (app/rag/sparse.py); die Dense-Modelle kommen aus der Registry
    # (app/rag/models_runtime.py) und werden je Knowledge gewaehlt.
    sparse_embeddings = SparseRouter()  # BM25 je Sprache (app/rag/sparse.py)
    try:  # Modell jetzt laden (Download), nicht erst bei der ersten Frage
        from app.rag.sparse import LEGACY_LANGUAGE, default_language

        sparse_embeddings.for_language(default_language())
        sparse_embeddings.for_language(LEGACY_LANGUAGE)
    except Exception as e:
        logger.warning(f"BM25-Modell konnte nicht vorgeladen werden: {e}")

    # Retriever ueber die Collections des jeweiligen Chat-Requests
    # (siehe app/retrieval.py); ohne Scope gilt der Standard-Bestand.
    retriever = MultiCollectionRetriever(
        client=qdrant_client,
        sparse_embeddings=sparse_embeddings,
    )

    llm = VisionLLM()
    custom_tools = CustomTools(
        llm=llm,
        retriever=retriever,
        data_dir=data_dir,
        summaries_dir=summaries_dir,
    )
    tools = custom_tools.get_tools() + SqlTools().get_tools()

    app.extensions["sparse_embeddings"] = sparse_embeddings
    app.extensions["llm"] = llm
    # Geteilte Tools fuer die modellabhaengigen Agents (app/llm/registry.py).
    app.extensions["agent_tools"] = tools

    logger.info("Ressourcen bereit (Embedding-Modell + Agent).")
