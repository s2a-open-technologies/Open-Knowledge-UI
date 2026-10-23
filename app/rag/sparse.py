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
Sprache der Stichwortsuche (BM25).

FastEmbeds BM25 stemmt und entfernt Stoppwoerter fuer genau eine Sprache (Standard: Englisch).
Die Sprache steckt in den Vektoren eines Index: Dokumente und Fragen muessen dieselbe nutzen.
Darum merkt sich jedes Knowledge, womit es indexiert wurde (``knowledge.bm25_language``;
leer = Altbestand = Englisch), und die Suche fragt jede Collection in ihrer Sprache. Die globale
Einstellung ``BM25_LANGUAGE`` gilt fuer neue und neu indexierte Knowledge sowie fuer Chat-Anhaenge.

Gemessen (118 deutsche Absaetze mit Fragen in anderen Wortformen, nur BM25): Treffer@1 englisch
39, deutsch 48 (MRR 0,425 -> 0,499). Die Gruppe haengt am Stemmer: deutsch ohne Stemmer 36.
"""

import threading
from logging import getLogger

logger = getLogger(__name__)

LEGACY_LANGUAGE = "english"
DEFAULT_LANGUAGE = "german"
MODEL = "Qdrant/bm25"


def supported_languages() -> list[str]:
    from fastembed.sparse.bm25 import supported_languages as langs

    return sorted(langs)


def default_language() -> str:
    """Sprache fuer neue Knowledge (Einstellung ``BM25_LANGUAGE``)."""
    from app.core.database_service import get_db_service
    from app.core.settings_service import SettingsService

    session = get_db_service().get_session()
    try:
        value = (SettingsService(session).get("BM25_LANGUAGE") or "").strip().lower()
    except Exception:
        value = ""
    finally:
        session.close()
    return value if value in supported_languages() else DEFAULT_LANGUAGE


def knowledge_language(knowledge: dict | None) -> str:
    """Sprache, mit der das Knowledge indexiert wurde (Altbestand: Englisch)."""
    return (
        (knowledge or {}).get("bm25_language") or ""
    ).strip().lower() or LEGACY_LANGUAGE


class SparseRouter:
    """Stichwort-Encoder je Sprache. Ohne Angabe gilt die Standardsprache der Einstellung.

    Hat die Schnittstelle von ``FastEmbedSparse`` (``embed_documents``/``embed_query``), damit
    bestehender Code und Tests unveraendert damit arbeiten; ``for_language`` liefert den Encoder
    einer bestimmten Sprache.
    """

    def __init__(self):
        self._by_language: dict[str, object] = {}
        self._lock = threading.Lock()

    def for_language(self, language: str):
        language = (language or LEGACY_LANGUAGE).lower()
        with self._lock:
            if language not in self._by_language:
                from langchain_qdrant import FastEmbedSparse

                self._by_language[language] = FastEmbedSparse(
                    model_name=MODEL, language=language
                )
            return self._by_language[language]

    def embed_documents(self, texts):
        return self.for_language(default_language()).embed_documents(texts)

    def embed_query(self, text):
        return self.for_language(default_language()).embed_query(text)


def resolve(sparse, language: str):
    """Encoder fuer ``language``; Encoder ohne ``for_language`` (z. B. Test-Fakes) bleiben unveraendert."""
    getter = getattr(sparse, "for_language", None)
    return getter(language) if getter else sparse
