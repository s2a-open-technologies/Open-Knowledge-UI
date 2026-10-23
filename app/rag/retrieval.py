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
Retrieval ueber mehrere Qdrant-Collections.

Der Agent ist ein Singleton und wird einmal beim Start gebaut. Welche
Collections durchsucht werden und mit welchem Retrieval-Profil, haengt aber vom
Chat-Request ab (gewaehlte Knowledge-Bereiche, Anhaenge der Session). Der Scope
wird deshalb ueber eine ``ContextVar`` an den Retriever uebergeben; sie wird von
asyncio und den LangChain-Executors in Tool-Aufrufe kopiert.

Die eigentliche Suche (Hybrid, Query-Enhancement, Reranking, ...) liegt in
``app/rag/pipeline.py``.
"""

import json
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from qdrant_client import models as qdrant_models

from app.knowledge import external as sources
from app.rag import model_settings, models_runtime, pipeline, profiles
from app.rag import sparse as sparse_lang
from app.rag.collections import USER_UPLOADS_COLLECTION

logger = getLogger(__name__)

DEFAULT_COLLECTION = "open_knowledge_ui"


@dataclass(frozen=True)
class RetrievalScope:
    """Durchsuchte Collections eines Requests.

    ``collections``: Knowledge-Collections. ``session_id``: wenn gesetzt, wird
    zusaetzlich die Anhang-Collection gefiltert nach dieser Session durchsucht.
    ``profiles``: je Collection die Konfiguration ihres Retrieval-Profils (fehlt
    eine, gilt das Standardprofil). ``helper``: LLM-Aufruf fuer das
    Query-Enhancement (``helper(prompt, config) -> Text``).
    """

    collections: tuple[str, ...] = field(default_factory=tuple)
    session_id: str | None = None
    profiles: tuple[tuple[str, dict], ...] = field(default=(), compare=False)
    helper: Callable[[str, dict], str] | None = field(default=None, compare=False)
    # Strukturierte Dateien (CSV/Excel) der durchsuchten Knowledge-Bereiche fuer die
    # pandas-Tools: (Namenspraefix, Ordner). Nur diese Ordner sind den Tools zugaenglich.
    data_dirs: tuple[tuple[str, str], ...] = field(default=(), compare=False)
    # Inhaltsverzeichnis je Knowledge: (Name, Text); haengt am Knowledge-Eintrag
    indexes: tuple[tuple[str, str], ...] = field(default=(), compare=False)
    # Name des Knowledge je Collection: (Collection, Name), fuer die Herkunftsangabe der Treffer
    knowledge_names: tuple[tuple[str, str], ...] = field(default=(), compare=False)
    # Embedding-Modell je Collection (JSON-Spec): womit die Vektoren entstanden sind. Fehlt eine
    # Collection, gilt das globale Modell.
    specs: tuple[tuple[str, str], ...] = field(default=(), compare=False)
    # BM25-Sprache je Collection (womit der Stichwort-Index entstand); fehlt eine: Altbestand = Englisch
    sparse_langs: tuple[tuple[str, str], ...] = field(default=(), compare=False)
    # Treffer, die die Dokumentsuche in diesem Request geliefert hat (fuer die Quellenanzeige
    # mit Score im Chat). Wird von der Suche befuellt, vom Chat-Stream ausgelesen.
    sources: list = field(default_factory=list, compare=False)
    # Externe Quellen: (Knowledge-Collection "ext_<id>", JSON der Quellen-Konfiguration inkl. Key)
    externals: tuple[tuple[str, str], ...] = field(default=(), compare=False)
    # Wer fragt (fuer Zugriffsfilter externer Quellen): username, display_name, groups,
    # external_groups, is_admin
    identity: dict = field(default_factory=dict, compare=False)
    # Hinweise zu nicht erreichbaren Quellen (fuer die Anzeige), vom Retriever befuellt
    warnings: list = field(default_factory=list, compare=False)
    # Nutzer des Requests: bestimmt den Ordner der Analysen (Werkzeuge des Modells)
    user_id: int | None = field(default=None, compare=False)
    # SQL-Quellen: (Knowledge-ID, Name, JSON der Konfiguration inkl. Passwort; bleibt im Prozess)
    sql_sources: tuple[tuple[int, str, str], ...] = field(default=(), compare=False)


_scope: ContextVar[RetrievalScope | None] = ContextVar("retrieval_scope", default=None)


def set_scope(scope: RetrievalScope | None):
    """Setzt den Scope fuer den aktuellen Kontext (gibt das Reset-Token zurueck)."""
    return _scope.set(scope)


def get_scope_or_none() -> RetrievalScope | None:
    """Der gesetzte Scope oder ``None`` (z. B. in Skripten ausserhalb eines Chats)."""
    return _scope.get()


def get_scope() -> RetrievalScope:
    # Ohne expliziten Scope (z. B. Skripte) gilt der Standard-Bestand.
    scope = _scope.get()
    if scope:
        return scope
    # Ohne Scope (Skripte): die aktuelle Collection des Standard-Knowledge
    try:
        from app.knowledge.service import get_knowledge

        return RetrievalScope(collections=(get_knowledge(None)["collection_name"],))
    except Exception:
        return RetrievalScope(collections=(DEFAULT_COLLECTION,))


class MultiCollectionRetriever(BaseRetriever):
    """Retriever ueber die Collections des Scopes; delegiert an die Pipeline."""

    client: Any
    sparse_embeddings: Any
    # Testnaht: ein fester Embedder statt der Modell-Registry
    dense_embeddings: Any = None
    # Testnaht: ersetzt den Cross-Encoder
    rerank_fn: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def _embedder(self, spec: models_runtime.ModelSpec):
        return self.dense_embeddings or models_runtime.get_embedder(spec)

    def _run(
        self,
        question: str,
        collections: list[str],
        cfg: dict,
        scope,
        spec,
        flt=None,
        language=None,
    ):
        kwargs = {"rerank_fn": self.rerank_fn} if self.rerank_fn else {}
        return pipeline.run(
            question,
            collections,
            cfg,
            client=self.client,
            dense=self._embedder(spec),
            sparse=sparse_lang.resolve(
                self.sparse_embeddings, language or sparse_lang.default_language()
            ),
            helper=scope.helper,
            query_filter=flt,
            **kwargs,
        )

    def _run_external(
        self, question: str, collection: str, raw: str, cfg: dict, scope, spec
    ):
        """Fragt eine externe Quelle (nur lesend) mit deren Collection, Vektornamen und Feldern ab."""
        src = sources.parse_config(raw)
        flt, blocked = sources.build_filter(src, scope.identity)
        if (
            blocked
        ):  # Zugriffsfilter ohne passende Identitaet: nichts liefern, nie ungefiltert
            return []
        client = sources.connect(src)
        try:
            layout = sources.layout_for(src)
            kwargs = {"rerank_fn": self.rerank_fn} if self.rerank_fn else {}
            return pipeline.run(
                question,
                [src.collection],
                cfg,
                client=client,
                dense=self._embedder(spec),
                sparse=sparse_lang.resolve(
                    self.sparse_embeddings, sparse_lang.LEGACY_LANGUAGE
                ),  # fremder Index: wie bisher
                helper=scope.helper,
                layout=layout,
                query_filter=flt,
                **kwargs,
            )
        finally:
            client.close()

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        scope = get_scope()
        default_cfg = profiles.default_config()
        by_collection = dict(scope.profiles)

        # Collections mit gleicher Konfiguration UND gleichem Embedding-Modell teilen sich
        # eine Pipeline-Ausfuehrung (Vektoren verschiedener Modelle sind nicht vergleichbar).
        global_spec = None if self.dense_embeddings else model_settings.embedding_spec()
        spec_of = dict(scope.specs)
        lang_of = dict(scope.sparse_langs)
        groups: dict[
            tuple[str, str, str], tuple[dict, models_runtime.ModelSpec, list[str]]
        ] = {}
        external = dict(scope.externals)
        for collection in scope.collections:
            if collection in external:
                continue  # fremde Quellen laufen einzeln (eigener Client und Aufbau)
            cfg = by_collection.get(collection, default_cfg)
            raw = spec_of.get(collection)
            spec = models_runtime.ModelSpec.from_dict(raw) if raw else global_spec
            language = lang_of.get(collection, sparse_lang.LEGACY_LANGUAGE)
            key = (json.dumps(cfg, sort_keys=True), spec.key if spec else "", language)
            groups.setdefault(key, (cfg, spec, []))[2].append(collection)

        result_lists: list[list[pipeline.Hit]] = []

        # Anhaenge zuerst: wer im Chat eine Datei anhaengt, meint sie meistens.
        if scope.session_id:
            flt = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="session_id",
                        match=qdrant_models.MatchValue(value=scope.session_id),
                    )
                ]
            )
            try:
                upload_spec = global_spec
                coll = (
                    USER_UPLOADS_COLLECTION
                    if self.dense_embeddings
                    else models_runtime.uploads_collection(upload_spec)
                )
                if self.client.collection_exists(coll):
                    hits = self._run(
                        query, [coll], default_cfg, scope, upload_spec, flt
                    )
                    if hits:
                        result_lists.append(hits)
            except Exception as e:
                logger.warning(f"Anhang-Suche fehlgeschlagen: {e}")

        for (_cfg_key, _spec_key, language), (cfg, spec, collections) in groups.items():
            try:
                hits = self._run(
                    query, collections, cfg, scope, spec, language=language
                )
            except Exception as e:
                logger.warning(f"Suche in {collections} fehlgeschlagen: {e}")
                continue
            if hits:
                result_lists.append(hits)

        for collection, raw in external.items():
            cfg = by_collection.get(collection, default_cfg)
            spec = (
                models_runtime.ModelSpec.from_dict(spec_of[collection])
                if spec_of.get(collection)
                else global_spec
            )
            try:
                hits = self._run_external(query, collection, raw, cfg, scope, spec)
            except Exception as e:
                logger.warning(
                    f"Externe Quelle {collection} nicht erreichbar: {type(e).__name__}: {e}"
                )
                scope.warnings.append(
                    f"Eine externe Quelle war nicht erreichbar ({type(e).__name__})."
                )
                continue
            if hits:
                result_lists.append(hits)

        # Round-Robin ueber die Trefferlisten (Scores sind ueber Profile hinweg
        # nicht vergleichbar). Mehr Quellen duerfen mehr Treffer beisteuern.
        final_k = max(
            [default_cfg["final_k"]] + [cfg["final_k"] for cfg, _, _ in groups.values()]
        )
        limit = final_k if len(result_lists) <= 1 else final_k * 2
        merged: list[pipeline.Hit] = []
        for rank in range(max((len(r) for r in result_lists), default=0)):
            for hits in result_lists:
                if rank < len(hits):
                    merged.append(hits[rank])
                    if len(merged) >= limit:
                        return [h.to_document() for h in merged]
        return [h.to_document() for h in merged]
