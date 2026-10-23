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

"""Externe Embedding- und Rerank-Dienste hinter einer Connection.

``kind`` der Connection bestimmt das Protokoll:

* ``llm``       OpenAI-kompatibel (vLLM, LiteLLM, Ollama ...): ``/embeddings`` mit ``input``, ``/rerank`` mit ``documents``
* ``tei``       HuggingFace Text Embeddings Inference: ``/embed`` mit ``inputs``, ``/rerank`` mit ``texts``;
                ``/info`` nennt das (einzige) Modell und seinen Typ
* ``infinity``  Infinity: ``/embeddings`` mit ``model``/``input``, ``/rerank`` mit ``model``/``documents``;
                ``/models`` nennt die Modelle mit Faehigkeiten

TEI und Infinity laufen meist ohne ``/v1``; wir probieren die Adresse mit und ohne ``/v1``.
"""

import time
from logging import getLogger

import requests

logger = getLogger(__name__)

KINDS = ("llm", "tei", "infinity")
EMBED_BATCH = 32
HTTP_TIMEOUT = 120
PROBE_TIMEOUT = 15
MAX_RERANK_CHARS = 2000
RANK_QUERY = "Wie hoch ist die Miete?"
RANK_TEXTS = ("Die Miete beträgt 800 Euro.", "Das Wetter ist schön.")


class EndpointError(Exception):
    """Text ist fuer Admins gedacht (ohne API-Key)."""


def _guard(conn: dict) -> None:
    from app.llm.registry import RemoteModelsError, _reject_link_local

    try:
        _reject_link_local(conn["base_url"])
    except RemoteModelsError as e:
        raise EndpointError(str(e)) from None


def _bases(conn: dict) -> list[str]:
    base = conn["base_url"].rstrip("/")
    if conn.get("kind", "llm") == "llm":
        return [base]
    alt = base[:-3] if base.endswith("/v1") else f"{base}/v1"
    return [base, alt]


def _request(
    conn: dict,
    method: str,
    path: str,
    payload: dict | None,
    what: str,
    timeout: float = HTTP_TIMEOUT,
):
    """Probiert die Adressvarianten durch; erste Antwort ausser 404 gilt. Wiederholt bei 5xx und Netzfehlern."""
    key = conn.get("api_key") or ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    last: str = "keine Antwort"
    not_found = 0
    bases = _bases(conn)
    for base in bases:
        url = f"{base}{path}"
        for attempt in range(3):
            try:
                resp = requests.request(
                    method,
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=False,
                )
            except requests.exceptions.Timeout:
                last = f"Zeitüberschreitung ({timeout:g} s)"
            except requests.exceptions.RequestException:
                last = "Verbindung fehlgeschlagen"
            else:
                if 300 <= resp.status_code < 400:
                    raise EndpointError(
                        f"{what}: Weiterleitung wird nicht verfolgt (HTTP {resp.status_code})"
                    )
                if resp.status_code in (401, 403):
                    raise EndpointError(
                        f"{what}: Zugriff verweigert (HTTP {resp.status_code}), API-Key prüfen"
                    )
                if resp.status_code == 404:
                    not_found += 1
                    break  # naechste Adressvariante
                if resp.status_code < 500:
                    if resp.status_code != 200:
                        raise EndpointError(
                            f"{what}: HTTP {resp.status_code}: {resp.text[:200]}"
                        )
                    return resp
                last = f"HTTP {resp.status_code}"
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
        else:
            continue
    if not_found == len(bases):
        raise EndpointError(
            f"{what}: Pfad {path} nicht gefunden (HTTP 404), Typ der Connection und Adresse prüfen"
        )
    raise EndpointError(f"{what}: {last}")


def _json(resp, what: str):
    try:
        return resp.json()
    except ValueError:
        raise EndpointError(f"{what}: Antwort ist kein JSON") from None


# --------------------------------------------------------------------------------------
# Modelle
# --------------------------------------------------------------------------------------

_CAPABILITIES = {
    "embed": "embedding",
    "embedding": "embedding",
    "rerank": "reranker",
    "reranker": "reranker",
}


def describe(conn: dict) -> list[dict]:
    """``[{"id", "capabilities": ["embedding"|"reranker", ...]}]`` fuer ``tei`` und ``infinity``."""
    _guard(conn)
    kind = conn.get("kind", "llm")
    if kind == "tei":
        info = _json(_request(conn, "GET", "/info", None, "TEI", PROBE_TIMEOUT), "TEI")
        model_type = info.get("model_type") if isinstance(info, dict) else None
        caps = [
            c
            for c in ("embedding", "reranker")
            if isinstance(model_type, dict) and c in model_type
        ]
        model_id = (
            info.get("model_id") if isinstance(info, dict) else None
        ) or "tei-model"
        return [{"id": model_id, "capabilities": caps}]
    if kind == "infinity":
        body = _json(
            _request(conn, "GET", "/models", None, "Infinity", PROBE_TIMEOUT),
            "Infinity",
        )
        out = []
        for m in (body.get("data") if isinstance(body, dict) else None) or []:
            if not isinstance(m, dict) or not m.get("id"):
                continue
            caps = [
                _CAPABILITIES[c]
                for c in (m.get("capabilities") or [])
                if c in _CAPABILITIES
            ]
            out.append(
                {
                    "id": m["id"],
                    "capabilities": sorted(set(caps)) or ["embedding", "reranker"],
                }
            )
        return out
    raise EndpointError("Nur für TEI und Infinity")


# --------------------------------------------------------------------------------------
# Embedding und Rerank
# --------------------------------------------------------------------------------------


def embed(conn: dict, model: str, texts: list[str]) -> list[list[float]]:
    kind = conn.get("kind", "llm")
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        if kind == "tei":
            resp = _request(
                conn,
                "POST",
                "/embed",
                {"inputs": batch, "truncate": True},
                "Embedding-Endpoint (TEI)",
            )
            body = _json(resp, "Embedding-Endpoint (TEI)")
            try:
                vectors = [list(map(float, v)) for v in body]
            except (TypeError, ValueError):
                raise EndpointError(
                    "Embedding-Endpoint (TEI): unerwartete Antwort"
                ) from None
        else:
            resp = _request(
                conn,
                "POST",
                "/embeddings",
                {"model": model, "input": batch},
                "Embedding-Endpoint",
            )
            body = _json(resp, "Embedding-Endpoint")
            try:
                rows = sorted(body["data"], key=lambda r: r["index"])
                vectors = [list(map(float, r["embedding"])) for r in rows]
            except (KeyError, ValueError, TypeError):
                raise EndpointError(
                    "Embedding-Endpoint: unerwartete Antwort (kein OpenAI-Format)"
                ) from None
        if len(vectors) != len(batch):
            raise EndpointError(
                "Embedding-Endpoint: Anzahl der Vektoren passt nicht zur Anfrage"
            )
        out.extend(vectors)
    return out


def rerank(conn: dict, model: str, query: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    clipped = [t[:MAX_RERANK_CHARS] for t in texts]
    kind = conn.get("kind", "llm")
    if kind == "tei":
        payload = {"query": query, "texts": clipped, "truncate": True}
        what = "Rerank-Endpoint (TEI)"
    else:
        payload = {"model": model, "query": query, "documents": clipped}
        what = "Rerank-Endpoint"
    body = _json(_request(conn, "POST", "/rerank", payload, what), what)
    try:
        rows = body["results"] if isinstance(body, dict) else body
        scores: list[float | None] = [None] * len(texts)
        for row in rows:
            scores[int(row["index"])] = float(
                row.get("relevance_score", row.get("score"))
            )
    except (KeyError, ValueError, TypeError, IndexError):
        raise EndpointError(f"{what}: unerwartete Antwort") from None
    if any(s is None for s in scores):
        raise EndpointError(f"{what}: nicht zu jedem Text kam ein Score zurück")
    return scores  # type: ignore[return-value]


# --------------------------------------------------------------------------------------
# Test der Connection
# --------------------------------------------------------------------------------------


def test_connection(conn: dict) -> dict:
    """Prueft Erreichbarkeit und beide APIs wirklich (ein Text einbetten, zwei Texte ranken)."""
    try:
        models = describe(conn)
    except EndpointError as e:
        return {"success": False, "error": str(e)}
    embedding = reranker = None
    emb_models = [m for m in models if "embedding" in m["capabilities"]]
    rr_models = [m for m in models if "reranker" in m["capabilities"]]
    if emb_models:
        model = emb_models[0]["id"]
        try:
            embedding = {
                "model": model,
                "dimension": len(embed(conn, model, ["Test"])[0]),
                "ok": True,
            }
        except EndpointError as e:
            embedding = {"model": model, "ok": False, "error": str(e)}
    if rr_models:
        model = rr_models[0]["id"]
        try:
            scores = rerank(conn, model, RANK_QUERY, list(RANK_TEXTS))
            reranker = {"model": model, "ok": scores[0] > scores[1]}
            if not reranker["ok"]:
                reranker["error"] = (
                    "Die Reihenfolge der Test-Texte stimmt nicht (Modell geeignet?)"
                )
        except EndpointError as e:
            reranker = {"model": model, "ok": False, "error": str(e)}
    parts = []
    if embedding:
        parts.append(
            f"Embedding {'✓ (Dimension ' + str(embedding['dimension']) + ')' if embedding['ok'] else '✗ ' + embedding['error']}"
        )
    if reranker:
        parts.append(
            f"Reranker {'✓' if reranker['ok'] else '✗ ' + reranker.get('error', '')}"
        )
    if not parts:
        return {
            "success": False,
            "error": "Der Dienst meldet weder ein Embedding- noch ein Rerank-Modell.",
            "models": [m["id"] for m in models],
        }
    ok = bool((embedding and embedding["ok"]) or (reranker and reranker["ok"]))
    result = {
        "success": ok,
        "message": "Verbunden: " + ", ".join(parts),
        "models": [m["id"] for m in models],
        "embedding": embedding,
        "reranker": reranker,
    }
    if not ok:
        result["error"] = result["message"]
    return result
