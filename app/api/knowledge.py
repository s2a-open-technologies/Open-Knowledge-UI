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

"""Knowledge-Bereiche: Anlegen, Bearbeiten, Inhaltsverzeichnis, SQL-Quellen, externe Quellen, Neu-Indexieren."""

import json
from logging import getLogger

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api._common import abort_if_reindexing
from app.api.errors import ApiError
from app.api.schemas.common import Loose, Success
from app.api.schemas.knowledge import (
    FilterTest,
    FilterTestResult,
    IndexPayload,
    KnowledgeCreate,
    KnowledgeOut,
    KnowledgeUpdate,
    ReindexStarted,
    SourceTest,
    ValuesColumns,
    ValuesOverview,
    ValuesSearch,
    ValuesSearchResult,
)
from app.auth.access import (
    accessible_knowledge_ids,
    add_grant,
    can_access_knowledge,
)
from app.auth.auth import Admin, CurrentUser
from app.auth.users import role_at_least
from app.core import runtime
from app.core.database_service import get_db_service
from app.core.settings_service import SettingsService
from app.ingest import reindex
from app.knowledge import external as sources
from app.knowledge.index_doc import ensure_index, get_index, refresh_index
from app.knowledge.service import (
    create_external_knowledge,
    create_knowledge,
    create_sql_knowledge,
    delete_knowledge,
    get_knowledge,
    is_external,
    is_remote,
    is_sql,
    list_knowledge,
    update_external_source,
    update_knowledge,
    update_sql_source,
)
from app.knowledge.sql import config as sqlcfg
from app.rag import model_settings

router = APIRouter(prefix="/api", tags=["knowledge"])
logger = getLogger(__name__)

_NOT_FOUND = "Knowledge nicht gefunden"


def _knowledge_with_counts(knowledge: dict, user: dict) -> dict:
    """Ergaenzt Dokumentanzahl, Chunk-Anzahl (best-effort) und ``can_write``."""
    docs = get_db_service().get_all_documents(knowledge_id=knowledge["id"])
    entry = {
        **knowledge,
        "document_count": len(docs),
        "chunk_count": None,
        "can_write": can_access_knowledge(user, knowledge["id"], "write"),
        # Index passt nicht zur aktuellen Modell-/Zerlege-Konfiguration (Neu indexieren)
        "stale": model_settings.is_stale(knowledge),
    }
    if is_sql(knowledge):
        entry["source_config"] = sqlcfg.parse_config(knowledge.get("source_config")).public()
        return entry
    if is_external(knowledge):
        entry["source_config"] = _public_source(knowledge)
        return entry
    entry.pop("source_config", None)
    client = runtime.state.client
    if client:
        try:
            entry["chunk_count"] = client.count(knowledge["collection_name"]).count
        except Exception:
            pass
    return entry


def _public_source(knowledge: dict) -> dict:
    """Konfiguration der externen Quelle ohne Key (nur ``has_key``)."""
    return sources.parse_config(knowledge.get("source_config")).public()


@router.get("/knowledge", response_model=list[KnowledgeOut])
def knowledge_list(user: CurrentUser):
    readable = accessible_knowledge_ids(user, "read")
    return [_knowledge_with_counts(k, user) for k in list_knowledge() if k["id"] in readable]


@router.post("/knowledge", response_model=KnowledgeOut, status_code=201)
def knowledge_create(user: CurrentUser, body: KnowledgeCreate | None = None):
    from app.rag.collections import ensure_collection

    body = body or KnowledgeCreate()
    name = (body.name or "").strip()
    if not name:
        raise ApiError(400, "Name erforderlich", success=False)

    from app.rag import models_runtime

    if body.kind == "external":
        return _create_external(name, body, user)
    if body.kind == "sql":
        return _create_sql(name, body, user)

    try:  # Modell laden und Dimension per Probe ermitteln (beim ersten Mal ggf. Download)
        dimension = models_runtime.get_embedder(model_settings.embedding_spec()).dimension
    except models_runtime.ModelLoadError as e:
        raise ApiError(503, f"Embedding-Modell nicht verfügbar: {e}", success=False)

    knowledge = create_knowledge(name, body.description, dimension)
    # Der Ersteller darf sein Knowledge verwalten (Admins ohnehin alles).
    add_grant("knowledge", knowledge["id"], "user", user["id"], "write")
    try:
        ensure_collection(
            runtime.state.client, knowledge["collection_name"], knowledge["embedding_dimension"]
        )
    except Exception as e:
        delete_knowledge(knowledge["id"])
        raise ApiError(500, f"Qdrant Fehler: {e!s}", success=False)
    return _knowledge_with_counts(knowledge, user)


def _index_payload(knowledge: dict) -> dict:
    """Inhaltsverzeichnis (``index.md``) des Knowledge fuer die Anzeige. Fehlt die Datei (Bestand aus einer
    aelteren Version, abgeschaltet gewesen), wird sie bei eingeschalteter Funktion nachtraeglich erzeugt."""
    if is_remote(knowledge):
        return {"available": False, "enabled": True, "markdown": None, "updated_at": None}
    session = get_db_service().get_session()
    try:
        enabled = SettingsService(session).get_bool("INGEST_INDEX_ENABLED", True)
    finally:
        session.close()
    text = ensure_index(knowledge) if enabled else None
    if not text:
        return {"available": True, "enabled": enabled, "markdown": None, "updated_at": None}
    _, updated = get_index(knowledge["id"])
    return {
        "available": True,
        "enabled": True,
        "markdown": text,
        "updated_at": updated.isoformat() if updated else None,
    }


@router.get("/knowledge/{knowledge_id:int}/index", response_model=IndexPayload)
def knowledge_index(knowledge_id: int, user: CurrentUser):
    """Das Inhaltsverzeichnis, das das Modell zur Orientierung bekommt (lesen genuegt)."""
    knowledge = get_knowledge(knowledge_id)
    if not knowledge or not can_access_knowledge(user, knowledge_id, "read"):
        raise ApiError(404, _NOT_FOUND, success=False)
    return _index_payload(knowledge)


@router.post("/knowledge/{knowledge_id:int}/index/refresh", response_model=IndexPayload)
def knowledge_index_refresh(knowledge_id: int, user: CurrentUser):
    """Erzeugt das Inhaltsverzeichnis neu (Schreibrecht)."""
    knowledge = get_knowledge(knowledge_id)
    if not knowledge or not can_access_knowledge(user, knowledge_id, "write"):
        raise ApiError(404, _NOT_FOUND, success=False)
    if is_remote(knowledge):
        raise ApiError(409, "Diese Quelle hat kein Inhaltsverzeichnis", success=False)
    refresh_index(knowledge_id)
    return _index_payload(knowledge)


@router.get("/knowledge/{knowledge_id:int}", response_model=KnowledgeOut)
def knowledge_detail(knowledge_id: int, user: CurrentUser):
    knowledge = get_knowledge(knowledge_id)
    if not knowledge or not can_access_knowledge(user, knowledge_id, "read"):
        raise ApiError(404, _NOT_FOUND, success=False)
    return _knowledge_with_counts(knowledge, user)


def _parse_external_body(data: dict, stored: dict | None = None):
    """Liest Quelle und Embedding-Modell aus dem Request und prueft sie hart.

    Ein leeres Key-Feld behaelt den gespeicherten Key, aber nur fuer dieselbe Adresse.
    Returns: (SourceConfig, ModelSpec, Dimension, Fakten)
    Raises: ValueError mit lesbarer Meldung.
    """
    from app.rag import models_runtime

    raw = dict(data.get("source") or {})
    cfg = sources.parse_config(raw)
    if stored is not None and not cfg.api_key:
        old = sources.parse_config(stored.get("source_config"))
        if old.url == cfg.url:
            cfg = sources.parse_config({**cfg.to_dict(), "api_key": old.api_key})
    try:
        spec = models_runtime.ModelSpec.from_dict(data.get("embedding"))
    except (ValueError, TypeError):
        raise ValueError("Das Embedding-Modell der Quelle muss angegeben werden") from None
    try:
        dimension = models_runtime.get_embedder(spec).dimension
    except models_runtime.ModelLoadError as e:
        raise ValueError(f"Embedding-Modell nicht verfügbar: {e}") from None
    try:
        sources.validate_access(cfg)
        embedder = models_runtime.get_embedder(spec)
        facts = sources.validate_source(
            cfg, dimension, embed=lambda text: embedder.embed_documents([text])[0]
        )
    except sources.SourceError as e:
        raise ValueError(str(e)) from None
    return cfg, spec, dimension, facts


def _create_external(name: str, body: KnowledgeCreate, user: dict):
    if not role_at_least(user["role"], "admin"):
        raise ApiError(403, "Externe Quellen legen nur Admins an", success=False)
    try:
        cfg, spec, dimension, _facts = _parse_external_body(body.model_dump())
    except ValueError as e:
        raise ApiError(400, str(e), success=False)
    knowledge = create_external_knowledge(name, body.description, spec, dimension, cfg.to_dict())
    add_grant("knowledge", knowledge["id"], "user", user["id"], "write")
    return _knowledge_with_counts(knowledge, user)


def _sql_probe(cfg: "sqlcfg.SqlConfig") -> dict:
    """Verbindungsprobe ueber den SQL-Worker. Wirft ValueError (Text fuer den Admin) bzw. RuntimeError (Worker nicht erreichbar)."""
    from app.agent.toolworker import client as worker
    from app.agent.toolworker import config as worker_config

    try:
        out = worker.get_client("sql").sql(
            "test",
            {"connection": cfg.connection()},
            worker_config.load_sql().timeout_s + 15,
        )
    except worker.ToolWorkerUnavailable as e:
        raise RuntimeError(str(e)) from None
    if not out.get("ok"):
        raise ValueError(out.get("error") or "Verbindung fehlgeschlagen")
    return {k: out.get(k) for k in ("version", "tables", "write_privilege")}


def _sql_source_from(data: dict, stored: dict | None = None) -> "sqlcfg.SqlConfig":
    raw = data.get("source") or {}
    cfg = sqlcfg.parse_config(raw)
    old = sqlcfg.parse_config(stored.get("source_config")) if stored else None
    cfg = sqlcfg.merge_password(cfg, old)
    if old is not None and "value_columns" not in raw:  # der Dialog kennt das Wertverzeichnis nicht
        cfg = sqlcfg.SqlConfig(**{**cfg.__dict__, "value_columns": old.value_columns})
    # Gesperrte oder nicht mehr freigegebene Spalten fliegen aus der Auswahl des Wertverzeichnisses
    from app.knowledge.sql import valueindex

    cfg = sqlcfg.SqlConfig(**{**cfg.__dict__, "value_columns": valueindex.clean_value_columns(cfg)})
    sqlcfg.validate(cfg)
    return cfg


def _sql_error(e: Exception) -> ApiError:
    return ApiError(503 if isinstance(e, RuntimeError) else 400, str(e), success=False)


def _create_sql(name: str, body: KnowledgeCreate, user: dict):
    if not role_at_least(user["role"], "admin"):
        raise ApiError(403, "SQL-Quellen legen nur Admins an", success=False)
    try:
        cfg = _sql_source_from(body.model_dump())
        _sql_probe(cfg)
    except (sqlcfg.SqlConfigError, ValueError, RuntimeError) as e:
        raise _sql_error(e)
    knowledge = create_sql_knowledge(name, body.description, cfg.to_dict())
    add_grant("knowledge", knowledge["id"], "user", user["id"], "write")
    return _knowledge_with_counts(knowledge, user)


def _stored_id(raw) -> int | None:
    return int(raw) if str(raw or "").isdigit() else None


@router.post("/knowledge/sql/test", response_model=Loose)
def knowledge_sql_test(_user: Admin, body: SourceTest | None = None):
    """Verbindungsprobe: Version, Tabellen mit Spalten, Warnung bei Schreibrechten (nichts wird gespeichert)."""
    body = body or SourceTest()
    kid = _stored_id(body.knowledge_id)
    stored = get_knowledge(kid) if kid is not None else None
    try:
        cfg = _sql_source_from(body.model_dump(), stored if is_sql(stored) else None)
        facts = _sql_probe(cfg)
    except (sqlcfg.SqlConfigError, ValueError, RuntimeError) as e:
        raise _sql_error(e)
    return {"success": True, **facts}


def _sql_knowledge_or_404(knowledge_id: int):
    k = get_knowledge(knowledge_id)
    if not is_sql(k):
        raise ApiError(404, "SQL-Quelle nicht gefunden", success=False)
    return k


def _values_overview(k: dict) -> dict:
    from app.agent.toolworker import config as worker_config
    from app.knowledge.sql import valueindex

    cfg = sqlcfg.parse_config(k["source_config"])
    max_distinct, max_results = worker_config.values_settings()
    return {
        **valueindex.status(k["id"]),
        "columns": {t: list(c) for t, c in cfg.value_columns.items()},
        "max_distinct": max_distinct,
        "max_results": max_results,
    }


@router.get("/knowledge/{knowledge_id:int}/sql-values", response_model=ValuesOverview)
def sql_values_overview(knowledge_id: int, _user: Admin):
    return _values_overview(_sql_knowledge_or_404(knowledge_id))


@router.post("/knowledge/{knowledge_id:int}/sql-values/suggest", response_model=Loose)
def sql_values_suggest(knowledge_id: int, _user: Admin):
    """Textspalten mit Vorschlag (aufnehmen oder nicht, mit Begruendung). Liest nur Zaehlungen aus der Quelle."""
    from app.knowledge.sql import valueindex

    k = _sql_knowledge_or_404(knowledge_id)
    try:
        return {"success": True, "columns": valueindex.profile(k["source_config"])}
    except ValueError as e:
        raise ApiError(502, str(e), success=False)


@router.put("/knowledge/{knowledge_id:int}/sql-values", response_model=ValuesOverview)
def sql_values_save(knowledge_id: int, _user: Admin, body: ValuesColumns | None = None):
    """Speichert die Spaltenauswahl (``{"columns": {"tabelle": ["spalte", ...]}}``); gebaut wird separat."""
    k = _sql_knowledge_or_404(knowledge_id)
    cfg = sqlcfg.parse_config(k["source_config"])
    raw = (body.columns if body else None) or {}
    chosen = sqlcfg.parse_config({"value_columns": raw}).value_columns
    allowed = {t.lower() for t in cfg.tables}
    blocked = {t.lower(): {c.lower() for c in cols} for t, cols in cfg.blocked_columns.items()}
    for table, columns in chosen.items():
        if allowed and table.lower() not in allowed:
            raise ApiError(400, f"Die Tabelle {table} ist nicht freigegeben", success=False)
        for column in columns:
            if column.lower() in blocked.get(table.lower(), set()):
                raise ApiError(400, f"Die Spalte {table}.{column} ist gesperrt", success=False)
    updated = sqlcfg.SqlConfig(**{**cfg.__dict__, "value_columns": chosen})
    try:
        sqlcfg.validate(updated)
    except sqlcfg.SqlConfigError as e:
        raise ApiError(400, str(e), success=False)
    update_sql_source(knowledge_id, updated.to_dict())
    return _values_overview(get_knowledge(knowledge_id))


@router.post("/knowledge/{knowledge_id:int}/sql-values/build", response_model=Loose, status_code=202)
def sql_values_build(knowledge_id: int, _user: Admin):
    from app.knowledge.sql import valueindex

    k = _sql_knowledge_or_404(knowledge_id)
    if not sqlcfg.parse_config(k["source_config"]).value_columns:
        raise ApiError(400, "Erst Spalten auswählen und speichern", success=False)
    if not valueindex.start_build(knowledge_id, k["source_config"]):
        raise ApiError(409, "Der Aufbau läuft bereits", success=False)
    return {"success": True, **valueindex.status(knowledge_id)}


@router.post("/knowledge/{knowledge_id:int}/sql-values/search", response_model=ValuesSearchResult)
def sql_values_search(knowledge_id: int, _user: Admin, body: ValuesSearch | None = None):
    """Probe: welche Werte findet die Suche zu einem Begriff (wie das Modell sie sieht)."""
    from app.knowledge.sql import valueindex

    _sql_knowledge_or_404(knowledge_id)
    term = str((body.term if body else None) or "")
    hits = valueindex.search(knowledge_id, term)
    return {
        "results": [
            {
                "table": e["table"],
                "column": e["column"],
                "value": e["value"],
                "count": e["count"],
                "score": round(s, 2),
            }
            for e, s in hits
        ]
    }


def _with_stored_key(cfg, knowledge_id) -> object:
    """Leeres Key-Feld: gespeicherten Key nehmen, aber nie fuer eine andere Adresse."""
    kid = _stored_id(knowledge_id) if knowledge_id not in (None, "", 0) else None
    if kid is None:
        return cfg
    stored = get_knowledge(kid)
    if is_external(stored) and not cfg.api_key:
        old = sources.parse_config(stored.get("source_config"))
        if old.url == cfg.url:
            return sources.parse_config({**cfg.to_dict(), "api_key": old.api_key})
    return cfg


@router.post("/knowledge/external/test", response_model=Loose)
def knowledge_external_test(_user: Admin, body: SourceTest | None = None):
    """Fakten zu einer fremden Quelle zum Abgleich (nichts wird geraten oder gespeichert)."""
    body = body or SourceTest()
    cfg = _with_stored_key(sources.parse_config(body.source), body.knowledge_id)
    try:
        facts = sources.probe_source(cfg)
    except sources.SourceError as e:
        raise ApiError(502, str(e), success=False)
    return {"success": True, **facts}


@router.post("/knowledge/external/filter-test", response_model=FilterTestResult)
def knowledge_external_filter_test(_user: Admin, body: FilterTest | None = None):
    """Zeigt, was ein Benutzer mit diesem Zugriffsfilter sehen wuerde (ohne zu speichern).

    Body: ``source`` (inkl. ``access``), optional ``knowledge_id`` (gespeicherter Key) und
    ``identity`` (``username``, ``display_name``, ``groups``, ``external_groups``).
    """
    body = body or FilterTest()
    cfg = _with_stored_key(sources.parse_config(body.source), body.knowledge_id)
    ident = body.identity or {}
    identity = {
        "username": str(ident.get("username") or ""),
        "display_name": str(ident.get("display_name") or ""),
        "groups": [str(g) for g in (ident.get("groups") or [])],
        "external_groups": [str(g) for g in (ident.get("external_groups") or [])],
        "is_admin": False,
    }
    try:
        sources.validate_access(cfg)
        flt, blocked = sources.build_filter(cfg, identity)
        if blocked:
            return {"success": True, "blocked": True, "count": 0, "samples": []}
        client = sources.connect(cfg)
        try:
            count = int(client.count(cfg.collection, count_filter=flt, exact=True).count)
            points, _ = client.scroll(cfg.collection, scroll_filter=flt, limit=3, with_payload=True)
        finally:
            client.close()
    except sources.SourceError as e:
        raise ApiError(400, str(e), success=False)
    except Exception as e:
        raise ApiError(502, f"Abfrage fehlgeschlagen: {type(e).__name__}: {e}"[:300], success=False)
    texts = cfg.fields["text"]
    return {
        "success": True,
        "blocked": False,
        "count": count,
        "samples": [str((p.payload or {}).get(texts, ""))[:120] for p in points],
    }


@router.patch("/knowledge/{knowledge_id:int}", response_model=KnowledgeOut)
def knowledge_update(knowledge_id: int, user: CurrentUser, body: KnowledgeUpdate | None = None):
    if not can_access_knowledge(user, knowledge_id, "write"):
        raise ApiError(404, _NOT_FOUND, success=False)
    abort_if_reindexing(knowledge_id)
    data = body.model_dump(exclude_unset=True) if body else {}
    current = get_knowledge(knowledge_id)
    if is_external(current) and ("source" in data or "embedding" in data):
        if not role_at_least(user["role"], "admin"):
            raise ApiError(403, "Nur Admins ändern die Quelle", success=False)
        try:
            merged = {
                "source": data.get("source")
                or sources.parse_config(current["source_config"]).to_dict(),
                "embedding": data.get("embedding") or json.loads(current["embedding_spec"]),
            }
            cfg, spec, dimension, _f = _parse_external_body(merged, stored=current)
        except ValueError as e:
            raise ApiError(400, str(e), success=False)
        update_external_source(knowledge_id, spec, dimension, cfg.to_dict())
    if is_sql(current) and "source" in data:
        if not role_at_least(user["role"], "admin"):
            raise ApiError(403, "Nur Admins ändern die Quelle", success=False)
        try:
            cfg = _sql_source_from(data, current)
            _sql_probe(cfg)
        except (sqlcfg.SqlConfigError, ValueError, RuntimeError) as e:
            raise _sql_error(e)
        update_sql_source(knowledge_id, cfg.to_dict())
        from app.knowledge.sql import valueindex

        valueindex.prune(knowledge_id, cfg.to_dict())  # bereits gespeicherte Werte gesperrter Spalten entfernen
    knowledge = update_knowledge(knowledge_id, data.get("name"), data.get("description"))
    if not knowledge:
        raise ApiError(404, _NOT_FOUND, success=False)
    if not is_remote(knowledge):
        refresh_index(knowledge_id)  # Name steht im Verzeichnis
    return _knowledge_with_counts(knowledge, user)


@router.delete("/knowledge/{knowledge_id:int}", response_model=Success)
def knowledge_delete(knowledge_id: int, user: CurrentUser):
    if not can_access_knowledge(user, knowledge_id, "write"):
        raise ApiError(404, _NOT_FOUND, success=False)
    status = runtime.state.ingestion_status
    if status["is_ingesting"] and status.get("knowledge_id") == knowledge_id:
        raise ApiError(409, "Ingestion läuft gerade", success=False)
    abort_if_reindexing(knowledge_id)

    knowledge = delete_knowledge(knowledge_id)
    if knowledge:
        from app.llm.connections import remove_knowledge_from_models

        remove_knowledge_from_models(knowledge_id)
    if not knowledge:
        raise ApiError(
            404, "Knowledge nicht gefunden oder Standard-Bereich (nicht löschbar)", success=False
        )

    client = runtime.state.client
    if client and not is_remote(knowledge):  # eine fremde Quelle wird nie angefasst
        try:
            client.delete_collection(knowledge["collection_name"])
        except Exception as e:
            logger.warning(f"Collection konnte nicht gelöscht werden: {e}")
    return Success()


@router.post(
    "/knowledge/{knowledge_id:int}/reindex",
    response_model=ReindexStarted,
    response_model_exclude_none=True,
    status_code=202,
)
def knowledge_reindex(knowledge_id: int, user: CurrentUser):
    if not can_access_knowledge(user, knowledge_id, "write"):
        raise ApiError(404, _NOT_FOUND, success=False)
    if is_remote(get_knowledge(knowledge_id)):
        raise ApiError(409, "Diese Quelle ist schreibgeschützt", success=False)
    return _start_reindex([knowledge_id])


@router.post("/reindex-all", response_model=ReindexStarted, response_model_exclude_none=True)
def reindex_all(_user: Admin):
    stale = [k["id"] for k in list_knowledge() if model_settings.is_stale(k)]
    if not stale:
        return {"success": True, "started": False, "message": "Alles aktuell"}
    return JSONResponse(_start_reindex(stale), status_code=202)


@router.get("/reindex/status", response_model=Loose)
def reindex_status(_user: CurrentUser):
    return reindex.status()


def _start_reindex(ids: list[int]) -> dict:
    client = runtime.state.client
    sparse = runtime.state.sparse_embeddings
    if not client or not sparse:
        raise ApiError(503, "RAG-System wird noch initialisiert", success=False)
    if runtime.state.ingestion_status["is_ingesting"]:
        raise ApiError(409, "Ingestion läuft gerade", success=False)
    try:
        reindex.start(ids, client, sparse)
    except reindex.ReindexBusy as e:
        raise ApiError(409, str(e), success=False)
    return {"success": True, "started": True, "knowledge_ids": ids}
