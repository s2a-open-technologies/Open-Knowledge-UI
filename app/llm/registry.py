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
Modell-Registry: Modelle der Connections auflisten, Referenzen aufloesen und
pro Anfrage den Agent bauen.

Der Agent ist an LLM, System-Prompt und Tools gebunden. Er wird bewusst pro
Anfrage neu gebaut und nicht gecacht: jede Chat-Anfrage laeuft in einer eigenen
Event-Loop, die danach geschlossen wird. Ein gecachtes LLM haelt einen async
HTTP-Pool, dessen Keep-Alive-Verbindungen an die tote Loop gebunden waeren
("Event loop is closed" bei der Folge-Anfrage). Aenderungen an Connection oder
Modell-Konfiguration greifen dadurch sofort.
"""

import hashlib
import ipaddress
import json
import socket
import time
from dataclasses import dataclass, field
from logging import getLogger
from urllib.parse import urlparse

import requests

from app.agent.document_agent import DocumentAgent
from app.llm import connections as store
from app.llm.factory import build_chat_llm

logger = getLogger(__name__)

MODELS_TTL_SECONDS = 60

_models_cache: dict[tuple[int, str, str], tuple[float, list[str]]] = {}


class ModelUnavailableError(Exception):
    """Modell oder Connection existiert nicht bzw. ist deaktiviert."""


class RemoteModelsError(Exception):
    """Die Modellliste des Endpunkts konnte nicht abgerufen werden."""


@dataclass(frozen=True)
class ResolvedModel:
    ref: str
    display_name: str
    connection_id: int
    base_url: str
    api_key: str = field(repr=False)
    model_id: str
    params: dict
    system_prompt: str
    knowledge_ids: tuple[int, ...] | None
    # Websuche-Werkzeug zugeschaltet (greift nur, wenn SearXNG und PII-Filter eingerichtet sind)
    web_search: bool = False
    # Bild-Anhaenge gehen direkt an das Modell (zusaetzlich zum gelesenen Text)
    vision: bool = False
    # Nutzer duerfen zusaetzlich eigenes Knowledge waehlen; sonst durchsucht das Modell nur sein eigenes
    allow_extra_knowledge: bool = True

    def signature(self) -> str:
        payload = json.dumps(
            [
                self.base_url,
                self.api_key,
                self.model_id,
                self.params,
                self.system_prompt,
                self.web_search,
            ],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Modelle eines Endpunkts
# --------------------------------------------------------------------------


def _reject_link_local(base_url: str) -> None:
    """Sperrt Link-Local-Ziele (169.254.0.0/16, fe80::/10), v. a. die
    Cloud-Metadaten-Adresse, auch nach DNS-Aufloesung. Private LAN-Adressen
    bleiben erlaubt: lokale Server (vLLM, Ollama, LiteLLM) sind der Normalfall.

    Raises:
        RemoteModelsError: bei gesperrtem Ziel.
    """
    host = urlparse(base_url).hostname
    if not host:
        raise RemoteModelsError("Ungültige Base-URL")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return  # nicht aufloesbar: requests meldet den Fehler verstaendlich
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_link_local:
            raise RemoteModelsError(
                "Link-Local-Adressen (z. B. 169.254.x.x) sind als Ziel nicht erlaubt"
            )


def fetch_remote_models(connection: dict, force: bool = False) -> list[str]:
    """Ruft ``GET <base_url>/models`` ab (kurz gecacht).

    ``connection`` braucht ``id``, ``base_url`` und ``api_key``.

    Raises:
        RemoteModelsError: bei Netzwerk-, HTTP- oder Formatfehlern.
    """
    cache_key = (
        connection["id"],
        connection["base_url"],
        connection.get("api_key", ""),
    )
    cached = _models_cache.get(cache_key)
    if cached and not force and time.time() - cached[0] < MODELS_TTL_SECONDS:
        return cached[1]

    _reject_link_local(connection["base_url"])
    url = f"{connection['base_url']}/models"
    headers = {}
    if connection.get("api_key"):
        headers["Authorization"] = f"Bearer {connection['api_key']}"
    try:
        # Keine Weiterleitungen: ein Ziel koennte sonst auf eine gesperrte Adresse umleiten.
        resp = requests.get(url, headers=headers, timeout=10, allow_redirects=False)
    except requests.exceptions.Timeout:
        raise RemoteModelsError(f"Timeout (10s) bei {url}") from None
    except requests.exceptions.RequestException as e:
        raise RemoteModelsError(f"Verbindung fehlgeschlagen: {e}") from None

    if 300 <= resp.status_code < 400:
        raise RemoteModelsError(
            f"Weiterleitung von {url} wird nicht verfolgt (HTTP {resp.status_code})"
        )
    if resp.status_code in (401, 403):
        raise RemoteModelsError(
            f"Zugriff verweigert (HTTP {resp.status_code}): API-Key prüfen"
        )
    if resp.status_code != 200:
        raise RemoteModelsError(f"HTTP {resp.status_code} von {url}")
    try:
        ids = sorted(
            {
                m["id"]
                for m in resp.json().get("data", [])
                if isinstance(m, dict) and m.get("id")
            }
        )
    except (ValueError, AttributeError):
        raise RemoteModelsError(
            f"Unerwartete Antwort von {url} (kein OpenAI-Format)"
        ) from None

    _models_cache[cache_key] = (time.time(), ids)
    return ids


def test_connection(connection: dict) -> dict:
    try:
        ids = fetch_remote_models(connection, force=True)
    except RemoteModelsError as e:
        return {"success": False, "error": str(e)}
    return {
        "success": True,
        "message": f"Verbunden, {len(ids)} Modell(e) verfügbar.",
        "models": ids,
    }


def _is_customized(cfg: dict | None) -> bool:
    return bool(
        cfg
        and (
            cfg["system_prompt"]
            or cfg["params"]
            or cfg["knowledge_ids"]
            or cfg.get("web_search")
            or cfg.get("vision")
            or (cfg["name"] and not cfg["is_clone"])
        )
    )


def list_models(force: bool = False) -> dict:
    """Alle Modelle aller aktiven Connections plus Klone.

    Returns:
        ``{"models": [...], "errors": [{connection_id, connection, error}]}``.
        Faellt eine Connection aus, erscheinen ihre Klone trotzdem (mit
        ``available=False``), und der Fehler steht in ``errors``.
    """
    configs = store.list_model_configs()
    base_cfg = {
        (c["connection_id"], c["base_model"]): c for c in configs if not c["is_clone"]
    }
    clones = [c for c in configs if c["is_clone"]]

    models: list[dict] = []
    errors: list[dict] = []

    for conn in store.list_connections():
        if not conn["enabled"] or conn.get("kind", "llm") != "llm":
            continue  # TEI/Infinity liefern Embeddings und Reranker, keine Chat-Modelle
        full = store.get_connection(conn["id"], with_key=True)
        try:
            remote = fetch_remote_models(full, force=force)
            reachable = True
        except RemoteModelsError as e:
            remote, reachable = [], False
            errors.append(
                {
                    "connection_id": conn["id"],
                    "connection": conn["name"],
                    "error": str(e),
                }
            )

        for model_id in remote:
            cfg = base_cfg.get((conn["id"], model_id))
            models.append(
                {
                    "ref": store.make_base_ref(conn["id"], model_id),
                    "name": (cfg["name"] if cfg and cfg["name"] else model_id),  # Alias
                    "model_id": model_id,
                    "connection_id": conn["id"],
                    "connection_name": conn["name"],
                    "is_clone": False,
                    "customized": _is_customized(cfg),
                    "visible": cfg["visible"] if cfg else True,
                    "available": True,
                    "config": cfg,
                }
            )

        for cfg in (c for c in clones if c["connection_id"] == conn["id"]):
            models.append(
                {
                    "ref": cfg["ref"],
                    "name": cfg["name"] or cfg["base_model"],
                    "model_id": cfg["base_model"],
                    "connection_id": conn["id"],
                    "connection_name": conn["name"],
                    "is_clone": True,
                    "customized": _is_customized(cfg),
                    "visible": cfg["visible"],
                    "available": reachable and cfg["base_model"] in remote,
                    "config": cfg,
                }
            )

    from app.agent.tools.web_access import web_search_available, web_search_masked
    from app.llm.factory import global_chat_defaults

    return {
        "models": models,
        # Werte, die gelten, wenn ein Modell einen Parameter leer laesst (fuer den Konfig-Dialog)
        "param_defaults": global_chat_defaults(),
        # fuer die Budget-Anzeige im Konfig-Dialog (Prozentwerte der Compaction)
        "compaction": _compaction_summary(),
        "errors": errors,
        "default": default_model_ref(),
        # Websuche pro Modell ist nur zuschaltbar, wenn SearXNG UND der PII-Filter eingerichtet sind
        "web_search_available": web_search_available(),
        # maskiert der PII-Filter die Suchanfragen? (sonst Warnhinweis im Konfig-Dialog)
        "web_search_masked": web_search_masked(),
    }


# --------------------------------------------------------------------------
# Aufloesen
# --------------------------------------------------------------------------


def default_model_ref() -> str:
    """Konfiguriertes Standardmodell; ohne Angabe das bisherige ``CHAT_MODEL``
    an der ersten aktiven Connection (damit Bestandsinstallationen weiterlaufen)."""
    ref = store.get_default_model()
    if ref:
        try:
            kind, key, _ = store.parse_ref(ref)
            if (kind == "base" and store.get_connection(key)) or (
                kind == "clone" and store.get_model_config(ref)
            ):
                return ref
        except ValueError:
            pass

    from app.core.database_service import get_db_service
    from app.core.settings_service import SettingsService

    db = get_db_service().get_session()
    try:
        legacy = SettingsService(db).get("CHAT_MODEL", "") or ""
    finally:
        db.close()
    first = next((c for c in store.list_connections() if c["enabled"]), None)
    return store.make_base_ref(first["id"], legacy) if first and legacy else ""


def resolve_model(ref: str | None = None) -> ResolvedModel:
    """Loest eine Referenz (oder das Standardmodell) zu einer Konfiguration auf.

    Raises:
        ModelUnavailableError: wenn nichts aufgeloest werden kann.
    """
    ref = ref or default_model_ref()
    if not ref:
        raise ModelUnavailableError(
            "Kein Modell ausgewählt und kein Standardmodell konfiguriert."
        )
    try:
        kind, key, model = store.parse_ref(ref)
    except ValueError as e:
        raise ModelUnavailableError(str(e)) from None

    cfg = store.get_model_config(ref)
    if kind == "clone":
        if not cfg:
            raise ModelUnavailableError("Das Modell existiert nicht mehr.")
        connection_id, model_id = cfg["connection_id"], cfg["base_model"]
    else:
        connection_id, model_id = key, model

    conn = store.get_connection(connection_id, with_key=True)
    if not conn:
        raise ModelUnavailableError("Die Connection des Modells existiert nicht mehr.")
    if not conn["enabled"]:
        raise ModelUnavailableError(f"Connection „{conn['name']}“ ist deaktiviert.")

    _reject_link_local(
        conn["base_url"]
    )  # auch vor dem Chat-Aufruf, nicht nur beim Abruf der Liste

    return ResolvedModel(
        ref=ref,
        display_name=(cfg["name"] if cfg and cfg["name"] else model_id),
        connection_id=connection_id,
        base_url=conn["base_url"],
        api_key=conn["api_key"],
        model_id=model_id,
        params=cfg["params"] if cfg else {},
        system_prompt=cfg["system_prompt"] if cfg else "",
        knowledge_ids=tuple(cfg["knowledge_ids"])
        if cfg and cfg["knowledge_ids"]
        else None,
        web_search=bool(cfg and cfg.get("web_search")),
        vision=bool(cfg and cfg.get("vision")),
        allow_extra_knowledge=bool(cfg.get("allow_extra_knowledge", True))
        if cfg
        else True,
    )


def _compaction_summary() -> dict:
    from app.chat import compaction

    cfg = compaction.load_settings()
    return {
        k: cfg[k]
        for k in (
            "enabled",
            "headroom_percent",
            "keep_percent",
            "output_reserve_percent",
        )
    }


def get_agent(resolved: ResolvedModel, tools: list) -> DocumentAgent:
    """Frischer Agent fuer das aufgeloeste Modell (pro Anfrage, siehe Modul-Docstring)."""
    from app.chat import compaction

    cfg = compaction.load_settings()
    budget = compaction.compute_budget(
        compaction.effective_params(resolved.params), cfg
    )
    system_prompt = DocumentAgent.compose_prompt(resolved.system_prompt)

    llm = build_chat_llm(
        resolved.base_url,
        resolved.api_key,
        resolved.model_id,
        resolved.params,
        streaming=True,
    )
    middleware = None
    if budget:
        from app.agent.context_guard import build_middleware

        overhead = compaction.estimate_tokens(system_prompt) + compaction.tools_tokens(
            tools
        )
        middleware = build_middleware(
            compaction.summary_llm(resolved, budget), budget, cfg["prompt"], overhead
        )
    return DocumentAgent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
    )


def clear_caches() -> None:
    _models_cache.clear()
