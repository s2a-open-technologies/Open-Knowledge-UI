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
Abruf-Pipeline eines Retrieval-Profils.

Stufen (alle ausser der Suche optional, siehe ``app/rag/profiles.py``):

1. Query-Enhancement: Helfer-LLM erzeugt zusaetzliche Suchfragen.
2. Suche je Frage und Collection direkt in Qdrant (dense, sparse oder hybrid mit
   gewichteter RRF-/DBSF-Fusion).
3. Zusammenfuehren aller Trefferlisten per Reciprocal Rank Fusion.
4. Reranking der besten Kandidaten gegen die *urspruengliche* Frage.
5. MMR fuer Diversitaet, danach die besten ``final_k``.
6. Kontext erweitern: benachbarte Chunks derselben Datei.

Jede Stufe schreibt in einen ``Trace``; Fehler in optionalen Stufen fuehren zu
einem Fallback statt zu einem Abbruch der Suche.
"""

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import getLogger

import numpy as np
from langchain_core.documents import Document
from qdrant_client import models as qm

from app.rag import queries

logger = getLogger(__name__)

RRF_K = 60
DENSE_VECTOR = ""
SPARSE_VECTOR = "langchain-sparse"


@dataclass
class Hit:
    collection: str
    id: str
    payload: dict
    score: float = 0.0
    vector: list[float] | None = field(default=None, repr=False)

    @property
    def key(self) -> tuple[str, str]:
        return (self.collection, self.id)

    @property
    def text(self) -> str:
        return (
            self.payload.get("page_content")
            or self.payload.get("original_content")
            or ""
        )

    @property
    def filename(self) -> str:
        return self.payload.get("filename") or self.payload.get("source") or "unbekannt"

    @property
    def chunk_index(self) -> int | None:
        idx = self.payload.get("chunk_index")
        return idx if isinstance(idx, int) else None

    def to_document(self) -> Document:
        return Document(
            page_content=self.text,
            metadata={
                "filename": self.filename,
                "source": self.payload.get("source"),
                "collection": self.collection,
                "score": round(self.score, 4),
                "chunk_index": self.chunk_index,
                "chunks": self.payload.get("_chunks"),
                "file_group_id": self.payload.get("file_group_id"),
                "headers": self.payload.get("headers"),
                "page_start": self.payload.get("page_start"),
                "page_end": self.payload.get("page_end"),
            },
        )


class Trace:
    """Ablaufprotokoll fuer den Test im Admin-Bereich und die Fehlersuche."""

    def __init__(self):
        self.stages: list[dict] = []

    @contextmanager
    def stage(self, name: str, **data):
        entry = {"stage": name, **data}
        self.stages.append(entry)
        start = time.perf_counter()
        try:
            yield entry
        finally:
            entry["ms"] = round((time.perf_counter() - start) * 1000)

    def to_dict(self) -> dict:
        return {
            "stages": self.stages,
            "total_ms": sum(s.get("ms", 0) for s in self.stages),
        }


def brief(hits: list[Hit], limit: int = 12) -> list[dict]:
    return [
        {
            "filename": h.filename,
            "collection": h.collection,
            "chunk_index": h.chunk_index,
            "score": round(h.score, 4),
            "snippet": " ".join(h.text.split())[:160],
        }
        for h in hits[:limit]
    ]


# --------------------------------------------------------------------------
# Suche
# --------------------------------------------------------------------------


def _vector_of(point, dense_name: str = DENSE_VECTOR) -> list[float] | None:
    vec = point.vector
    if isinstance(vec, dict):
        vec = vec.get(dense_name)
    return list(vec) if isinstance(vec, (list, tuple)) else None


def _query_collection(
    client,
    collection,
    dvec,
    svec,
    search: dict,
    flt,
    with_vectors,
    trace_entry,
    dense_name: str = DENSE_VECTOR,
    sparse_name: str = SPARSE_VECTOR,
):
    limit = search["candidates"]
    mode = search["mode"]
    vectors = [dense_name] if with_vectors else False
    common = {
        "limit": limit,
        "with_payload": True,
        "with_vectors": vectors,
        "query_filter": flt,
    }

    def dense_query():
        return {
            "query": dvec,
            "using": dense_name,
            "score_threshold": search["dense_min_score"],
        }

    def sparse_query():
        return {
            "query": qm.SparseVector(
                indices=list(svec.indices), values=list(svec.values)
            ),
            "using": sparse_name,
        }

    if mode == "dense":
        return client.query_points(collection, **dense_query(), **common).points
    if mode == "sparse":
        return client.query_points(collection, **sparse_query(), **common).points

    prefetch = [
        qm.Prefetch(limit=limit, filter=flt, **dense_query()),
        qm.Prefetch(limit=limit, filter=flt, **sparse_query()),
    ]
    weights = [search["dense_weight"], search["sparse_weight"]]
    if search["fusion"] == "dbsf":
        query = qm.FusionQuery(fusion=qm.Fusion.DBSF)
    elif weights[0] != weights[1]:
        query = qm.RrfQuery(rrf=qm.Rrf(weights=weights))
    else:
        query = qm.FusionQuery(fusion=qm.Fusion.RRF)
    try:
        return client.query_points(
            collection, prefetch=prefetch, query=query, **common
        ).points
    except Exception as e:
        if isinstance(query, qm.RrfQuery):
            # Aelterer Qdrant-Server ohne gewichtetes RRF: ungewichtet weiter.
            trace_entry.setdefault("warnings", []).append(
                f"Gewichtetes RRF nicht verfügbar ({type(e).__name__}), nutze ungewichtetes RRF"
            )
            return client.query_points(
                collection,
                prefetch=prefetch,
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                **common,
            ).points
        raise


def _merge(rank_lists: list[list[Hit]]) -> list[Hit]:
    """Reciprocal Rank Fusion ueber mehrere Trefferlisten, dedupliziert nach Punkt."""
    if len(rank_lists) == 1:
        return list(rank_lists[0])
    scores: dict[tuple[str, str], float] = {}
    first: dict[tuple[str, str], Hit] = {}
    for hits in rank_lists:
        for rank, hit in enumerate(hits):
            scores[hit.key] = scores.get(hit.key, 0.0) + 1.0 / (RRF_K + rank + 1)
            first.setdefault(hit.key, hit)
    merged = []
    for key, hit in first.items():
        hit.score = scores[key]
        merged.append(hit)
    merged.sort(key=lambda h: h.score, reverse=True)
    return merged


# --------------------------------------------------------------------------
# MMR und Nachbarn
# --------------------------------------------------------------------------


def _mmr(hits: list[Hit], k: int, lambda_mult: float) -> list[Hit]:
    """Maximal Marginal Relevance auf den Dense-Vektoren der Kandidaten."""
    scores = np.array([h.score for h in hits], dtype=float)
    spread = scores.max() - scores.min()
    relevance = (scores - scores.min()) / spread if spread > 0 else np.ones_like(scores)
    mat = np.array([h.vector for h in hits], dtype=float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms == 0, 1, norms)
    sim = mat @ mat.T

    selected = [int(np.argmax(relevance))]
    while len(selected) < min(k, len(hits)):
        best, best_val = None, -np.inf
        for i in range(len(hits)):
            if i in selected:
                continue
            val = lambda_mult * relevance[i] - (1 - lambda_mult) * max(
                sim[i][j] for j in selected
            )
            if val > best_val:
                best, best_val = i, val
        selected.append(best)
    return [hits[i] for i in selected]


def _expand_neighbors(
    client, hits: list[Hit], n: int, entry: dict, extra_filter=None
) -> list[Hit]:
    """Ersetzt jeden Treffer durch eine Passage mit ``n`` Chunks davor und danach."""
    groups: dict[tuple[str, str], list[Hit]] = {}
    passthrough: list[Hit] = []
    for h in hits:
        fg = h.payload.get("file_group_id")
        if h.chunk_index is not None and fg:
            groups.setdefault((h.collection, fg), []).append(h)
        else:
            passthrough.append(h)
    entry["without_chunk_index"] = len(passthrough)

    passages: list[Hit] = []
    for (collection, fg), group in groups.items():
        texts = {h.chunk_index: h.text for h in group}
        page_spans = {
            h.chunk_index: (h.payload.get("page_start"), h.payload.get("page_end"))
            for h in group
        }
        wanted: set[int] = set()
        for h in group:
            total = h.payload.get("chunk_total")
            top = (total - 1) if isinstance(total, int) else h.chunk_index + n
            wanted.update(
                range(max(0, h.chunk_index - n), min(top, h.chunk_index + n) + 1)
            )
        missing = sorted(wanted - texts.keys())
        if missing:
            points, _ = client.scroll(
                collection,
                scroll_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="file_group_id", match=qm.MatchValue(value=fg)
                        ),
                        qm.FieldCondition(
                            key="chunk_index", match=qm.MatchAny(any=missing)
                        ),
                        *([extra_filter] if extra_filter is not None else []),
                    ]
                ),
                limit=len(missing),
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                texts[payload.get("chunk_index")] = (
                    payload.get("original_content") or payload.get("page_content") or ""
                )
                page_spans[payload.get("chunk_index")] = (
                    payload.get("page_start"),
                    payload.get("page_end"),
                )

        best_by_index = {h.chunk_index: h for h in group}
        run: list[int] = []
        for idx in [*sorted(i for i in texts if i in wanted), None]:
            if run and (idx is None or idx != run[-1] + 1):
                members = [best_by_index[i] for i in run if i in best_by_index]
                if members:  # Luecken koennen Nachbarn ohne Treffer abtrennen
                    top_hit = max(members, key=lambda h: h.score)
                    payload = dict(top_hit.payload)
                    payload["page_content"] = "\n\n".join(texts[i] for i in run)
                    payload["_chunks"] = [run[0], run[-1]]
                    spans = [
                        page_spans[i]
                        for i in run
                        if page_spans.get(i, (None,))[0] is not None
                    ]
                    if spans:  # Seitenspanne der ganzen Passage
                        payload["page_start"] = min(sp[0] for sp in spans)
                        payload["page_end"] = max(sp[1] for sp in spans)
                    passages.append(
                        Hit(
                            collection,
                            top_hit.id,
                            payload,
                            top_hit.score,
                            top_hit.vector,
                        )
                    )
                run = []
            if idx is not None:
                run.append(idx)

    order = {h.key: pos for pos, h in enumerate(hits)}
    merged = passages + passthrough
    merged.sort(key=lambda h: (-h.score, order.get(h.key, 0)))
    return merged


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def run(
    question: str,
    collections: list[str],
    cfg: dict,
    *,
    client,
    dense,
    sparse,
    helper: queries.Helper | None = None,
    rerank_fn: Callable[[str, str, list[str]], list[float]] | None = None,
    query_filter=None,
    trace: Trace | None = None,
    layout: dict | None = None,
) -> list[Hit]:
    """Fuehrt die Pipeline fuer ``collections`` mit der Profil-Konfiguration ``cfg`` aus.

    ``layout`` beschreibt eine fremde Quelle: ``dense_name``, ``sparse_name`` (leer = keine
    Stichwortsuche), ``normalize`` (Payload -> unsere Feldnamen), ``neighbors`` (False =
    keine Nachbar-Chunks). Ohne ``layout`` gilt der Aufbau dieser App.
    """
    trace = trace or Trace()
    layout = layout or {}
    dense_name = layout.get("dense_name", DENSE_VECTOR)
    sparse_name = layout.get("sparse_name", SPARSE_VECTOR)
    normalize = layout.get("normalize") or (lambda payload: payload)
    if rerank_fn is None:
        from app.rag import model_settings

        rerank_fn = model_settings.rerank_score
    if not collections or not question.strip():
        return []

    search, rr, mmr, qe = (
        cfg["search"],
        cfg["rerank"],
        cfg["mmr"],
        cfg["query_enhancement"],
    )
    final_k = cfg["final_k"]
    notes: list[str] = []
    if not sparse_name and search["mode"] != "dense":
        search = {**search, "mode": "dense"}  # die Quelle hat keinen Sparse-Vektor
        notes.append("Quelle ohne Sparse-Vektor: reine Bedeutungssuche")
    neighbors = cfg["neighbors"] if layout.get("neighbors", True) else 0
    if neighbors != cfg["neighbors"]:
        notes.append("Nachbar-Chunks übersprungen (Feldzuordnung weicht ab)")

    # 1. Query-Enhancement
    qs = [question]
    if qe["mode"] != "off":
        with trace.stage("query_enhancement", mode=qe["mode"]) as st:
            if helper is None:
                st["skipped"] = "Kein Helfer-LLM verfügbar"
            else:
                try:
                    generated = queries.generate(question, qe, helper)
                    st["generated"] = generated
                    qs = ([question] if qe["include_original"] else []) + generated or [
                        question
                    ]
                except Exception as e:
                    logger.warning(f"Query-Enhancement fehlgeschlagen: {e}")
                    st["error"] = f"{type(e).__name__}: {e}"[:300]
            st["queries"] = qs

    # 2. Suche je Frage und Collection
    need_vectors = mmr["enabled"]  # Dense-Vektoren der Kandidaten nur fuer MMR
    rank_lists: list[list[Hit]] = []
    with trace.stage(
        "search", mode=search["mode"], fusion=search["fusion"], queries=len(qs)
    ) as st:
        st["per_collection"] = {}
        if notes:
            st["notes"] = notes
        embedded = []
        for q in qs:
            dvec = dense.embed_query(q) if search["mode"] != "sparse" else None
            svec = sparse.embed_query(q) if search["mode"] != "dense" else None
            embedded.append((dvec, svec))
        for collection in collections:
            count = 0
            for dvec, svec in embedded:
                try:
                    points = _query_collection(
                        client,
                        collection,
                        dvec,
                        svec,
                        search,
                        query_filter,
                        need_vectors,
                        st,
                        dense_name,
                        sparse_name,
                    )
                except Exception as e:
                    logger.warning(f"Suche in '{collection}' fehlgeschlagen: {e}")
                    st.setdefault("errors", []).append(
                        f"{collection}: {type(e).__name__}: {e}"[:200]
                    )
                    continue
                hits = [
                    Hit(
                        collection,
                        str(p.id),
                        normalize(p.payload or {}),
                        float(p.score),
                        _vector_of(p, dense_name),
                    )
                    for p in points
                ]
                count += len(hits)
                if hits:
                    rank_lists.append(hits)
            st["per_collection"][collection] = count

    if not rank_lists:
        return []

    # 3. Zusammenfuehren
    with trace.stage("fusion", lists=len(rank_lists)) as st:
        pool = _merge(rank_lists)
        pool_size = rr["candidates"] if rr["enabled"] else search["candidates"]
        pool = pool[: max(pool_size, final_k)]
        st["candidates"] = brief(pool)

    # 4. Reranking gegen die Originalfrage
    if rr["enabled"] and pool:
        with trace.stage("rerank", model=rr["model"]) as st:
            try:
                scores = rerank_fn(rr["model"], question, [h.text for h in pool])
                for h, s in zip(pool, scores, strict=True):
                    h.score = s
                pool.sort(key=lambda h: h.score, reverse=True)
                if rr["min_score"] is not None:
                    pool = [h for h in pool if h.score >= rr["min_score"]]
                st["result"] = brief(pool)
            except Exception as e:
                logger.warning(f"Reranking fehlgeschlagen: {e}")
                st["error"] = f"{type(e).__name__}: {e}"[:300]
                st["fallback"] = "ohne Reranking weiter"

    # 5. MMR und Top-k
    if mmr["enabled"] and len(pool) > final_k:
        with trace.stage("mmr", lambda_mult=mmr["lambda_mult"]) as st:
            if all(h.vector is not None for h in pool):
                final = _mmr(pool, final_k, mmr["lambda_mult"])
            else:
                st["skipped"] = "Dense-Vektoren der Kandidaten fehlen"
                final = pool[:final_k]
    else:
        final = pool[:final_k]

    # 6. Nachbar-Chunks
    if neighbors > 0 and final:
        with trace.stage("neighbors", n=neighbors) as st:
            try:
                final = _expand_neighbors(client, final, neighbors, st, query_filter)
            except Exception as e:
                logger.warning(f"Kontext-Erweiterung fehlgeschlagen: {e}")
                st["error"] = f"{type(e).__name__}: {e}"[:300]

    trace.stages.append({"stage": "final", "results": brief(final, 30)})
    return final
