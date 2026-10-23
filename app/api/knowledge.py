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

from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    make_response,
    request,
)

from app.api._common import _abort_if_reindexing
from app.auth.access import (
    accessible_knowledge_ids,
    add_grant,
    can_access_knowledge,
)
from app.auth.auth import auth_required_api, current_user, require_role
from app.auth.users import (
    role_at_least,
)
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

bp = Blueprint("knowledge", __name__, url_prefix="/api")
logger = getLogger(__name__)


def _knowledge_with_counts(knowledge: dict) -> dict:
    """Ergaenzt Dokumentanzahl, Chunk-Anzahl (best-effort) und ``can_write``."""
    docs = get_db_service().get_all_documents(knowledge_id=knowledge["id"])
    entry = {
        **knowledge,
        "document_count": len(docs),
        "chunk_count": None,
        "can_write": can_access_knowledge(current_user(), knowledge["id"], "write"),
        # Index passt nicht zur aktuellen Modell-/Zerlege-Konfiguration (Neu indexieren)
        "stale": model_settings.is_stale(knowledge),
    }
    if is_sql(knowledge):
        entry["source_config"] = sqlcfg.parse_config(
            knowledge.get("source_config")
        ).public()
        return entry
    if is_external(knowledge):
        entry["source_config"] = _public_source(knowledge)
        return entry
    entry.pop("source_config", None)
    client = current_app.extensions.get("client")
    if client:
        try:
            entry["chunk_count"] = client.count(knowledge["collection_name"]).count
        except Exception:
            pass
    return entry


def _public_source(knowledge: dict) -> dict:
    """Konfiguration der externen Quelle ohne Key (nur ``has_key``)."""
    return sources.parse_config(knowledge.get("source_config")).public()


@bp.route("/knowledge", methods=["GET"])
@auth_required_api
def knowledge_list():
    readable = accessible_knowledge_ids(current_user(), "read")
    return jsonify(
        [_knowledge_with_counts(k) for k in list_knowledge() if k["id"] in readable]
    )


@bp.route("/knowledge", methods=["POST"])
@require_role("user")
def knowledge_create():
    from app.rag.collections import ensure_collection

    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"success": False, "error": "Name erforderlich"}), 400

    from app.rag import model_settings, models_runtime

    if data.get("kind") == "external":
        return _create_external(name, data)
    if data.get("kind") == "sql":
        return _create_sql(name, data)

    try:  # Modell laden und Dimension per Probe ermitteln (beim ersten Mal ggf. Download)
        dimension = models_runtime.get_embedder(
            model_settings.embedding_spec()
        ).dimension
    except models_runtime.ModelLoadError as e:
        return jsonify(
            {"success": False, "error": f"Embedding-Modell nicht verfügbar: {e}"}
        ), 503

    knowledge = create_knowledge(name, data.get("description", ""), dimension)
    # Der Ersteller darf sein Knowledge verwalten (Admins ohnehin alles).
    add_grant("knowledge", knowledge["id"], "user", current_user()["id"], "write")
    client = current_app.extensions.get("client")
    try:
        ensure_collection(
            client, knowledge["collection_name"], knowledge["embedding_dimension"]
        )
    except Exception as e:
        delete_knowledge(knowledge["id"])
        return jsonify({"success": False, "error": f"Qdrant Fehler: {e!s}"}), 500
    return jsonify(_knowledge_with_counts(knowledge)), 201


def _index_payload(knowledge: dict) -> dict:
    """Inhaltsverzeichnis (``index.md``) des Knowledge fuer die Anzeige. Fehlt die Datei (Bestand aus einer
    aelteren Version, abgeschaltet gewesen), wird sie bei eingeschalteter Funktion nachtraeglich erzeugt."""
    if is_remote(knowledge):
        return {
            "available": False,
            "enabled": True,
            "markdown": None,
            "updated_at": None,
        }
    session = get_db_service().get_session()
    try:
        enabled = SettingsService(session).get_bool("INGEST_INDEX_ENABLED", True)
    finally:
        session.close()
    text = ensure_index(knowledge) if enabled else None
    if not text:
        return {
            "available": True,
            "enabled": enabled,
            "markdown": None,
            "updated_at": None,
        }
    _, updated = get_index(knowledge["id"])
    return {
        "available": True,
        "enabled": True,
        "markdown": text,
        "updated_at": updated.isoformat() if updated else None,
    }


@bp.route("/knowledge/<int:knowledge_id>/index", methods=["GET"])
@auth_required_api
def knowledge_index(knowledge_id):
    """Das Inhaltsverzeichnis, das das Modell zur Orientierung bekommt (lesen genuegt)."""
    knowledge = get_knowledge(knowledge_id)
    if not knowledge or not can_access_knowledge(current_user(), knowledge_id, "read"):
        return jsonify({"success": False, "error": "Knowledge nicht gefunden"}), 404
    return jsonify(_index_payload(knowledge))


@bp.route("/knowledge/<int:knowledge_id>/index/refresh", methods=["POST"])
@require_role("user")
def knowledge_index_refresh(knowledge_id):
    """Erzeugt das Inhaltsverzeichnis neu (Schreibrecht)."""
    knowledge = get_knowledge(knowledge_id)
    if not knowledge or not can_access_knowledge(current_user(), knowledge_id, "write"):
        return jsonify({"success": False, "error": "Knowledge nicht gefunden"}), 404
    if is_remote(knowledge):
        return jsonify(
            {"success": False, "error": "Diese Quelle hat kein Inhaltsverzeichnis"}
        ), 409
    refresh_index(knowledge_id)
    return jsonify(_index_payload(knowledge))


@bp.route("/knowledge/<int:knowledge_id>", methods=["GET"])
@auth_required_api
def knowledge_detail(knowledge_id):
    knowledge = get_knowledge(knowledge_id)
    if not knowledge or not can_access_knowledge(current_user(), knowledge_id, "read"):
        return jsonify({"success": False, "error": "Knowledge nicht gefunden"}), 404
    return jsonify(_knowledge_with_counts(knowledge))


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
        raise ValueError(
            "Das Embedding-Modell der Quelle muss angegeben werden"
        ) from None
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


def _create_external(name: str, data: dict):
    if not role_at_least(current_user()["role"], "admin"):
        return jsonify(
            {"success": False, "error": "Externe Quellen legen nur Admins an"}
        ), 403
    try:
        cfg, spec, dimension, _facts = _parse_external_body(data)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    knowledge = create_external_knowledge(
        name, data.get("description", ""), spec, dimension, cfg.to_dict()
    )
    add_grant("knowledge", knowledge["id"], "user", current_user()["id"], "write")
    return jsonify(_knowledge_with_counts(knowledge)), 201


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
    if (
        old is not None and "value_columns" not in raw
    ):  # der Dialog kennt das Wertverzeichnis nicht
        cfg = sqlcfg.SqlConfig(**{**cfg.__dict__, "value_columns": old.value_columns})
    # Gesperrte oder nicht mehr freigegebene Spalten fliegen aus der Auswahl des Wertverzeichnisses
    from app.knowledge.sql import valueindex

    cfg = sqlcfg.SqlConfig(
        **{**cfg.__dict__, "value_columns": valueindex.clean_value_columns(cfg)}
    )
    sqlcfg.validate(cfg)
    return cfg


def _sql_error(e: Exception):
    if isinstance(e, RuntimeError):
        return jsonify({"success": False, "error": str(e)}), 503
    return jsonify({"success": False, "error": str(e)}), 400


def _create_sql(name: str, data: dict):
    if not role_at_least(current_user()["role"], "admin"):
        return jsonify(
            {"success": False, "error": "SQL-Quellen legen nur Admins an"}
        ), 403
    try:
        cfg = _sql_source_from(data)
        _sql_probe(cfg)
    except (sqlcfg.SqlConfigError, ValueError, RuntimeError) as e:
        return _sql_error(e)
    knowledge = create_sql_knowledge(name, data.get("description", ""), cfg.to_dict())
    add_grant("knowledge", knowledge["id"], "user", current_user()["id"], "write")
    return jsonify(_knowledge_with_counts(knowledge)), 201


@bp.route("/knowledge/sql/test", methods=["POST"])
@require_role("admin")
def knowledge_sql_test():
    """Verbindungsprobe: Version, Tabellen mit Spalten, Warnung bei Schreibrechten (nichts wird gespeichert)."""
    data = request.get_json() or {}
    stored = (
        get_knowledge(int(data["knowledge_id"]))
        if str(data.get("knowledge_id") or "").isdigit()
        else None
    )
    try:
        cfg = _sql_source_from(data, stored if is_sql(stored) else None)
        facts = _sql_probe(cfg)
    except (sqlcfg.SqlConfigError, ValueError, RuntimeError) as e:
        return _sql_error(e)
    return jsonify({"success": True, **facts})


def _sql_knowledge_or_404(knowledge_id: int):
    k = get_knowledge(knowledge_id)
    if not is_sql(k):
        abort(
            make_response(
                jsonify({"success": False, "error": "SQL-Quelle nicht gefunden"}), 404
            )
        )
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


@bp.route("/knowledge/<int:knowledge_id>/sql-values", methods=["GET"])
@require_role("admin")
def sql_values_overview(knowledge_id):
    return jsonify(_values_overview(_sql_knowledge_or_404(knowledge_id)))


@bp.route("/knowledge/<int:knowledge_id>/sql-values/suggest", methods=["POST"])
@require_role("admin")
def sql_values_suggest(knowledge_id):
    """Textspalten mit Vorschlag (aufnehmen oder nicht, mit Begruendung). Liest nur Zaehlungen aus der Quelle."""
    from app.knowledge.sql import valueindex

    k = _sql_knowledge_or_404(knowledge_id)
    try:
        return jsonify(
            {"success": True, "columns": valueindex.profile(k["source_config"])}
        )
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 502


@bp.route("/knowledge/<int:knowledge_id>/sql-values", methods=["PUT"])
@require_role("admin")
def sql_values_save(knowledge_id):
    """Speichert die Spaltenauswahl (``{"columns": {"tabelle": ["spalte", ...]}}``); gebaut wird separat."""
    k = _sql_knowledge_or_404(knowledge_id)
    cfg = sqlcfg.parse_config(k["source_config"])
    raw = (request.get_json() or {}).get("columns") or {}
    chosen = sqlcfg.parse_config({"value_columns": raw}).value_columns
    allowed = {t.lower() for t in cfg.tables}
    blocked = {
        t.lower(): {c.lower() for c in cols} for t, cols in cfg.blocked_columns.items()
    }
    for table, columns in chosen.items():
        if allowed and table.lower() not in allowed:
            return jsonify(
                {
                    "success": False,
                    "error": f"Die Tabelle {table} ist nicht freigegeben",
                }
            ), 400
        for column in columns:
            if column.lower() in blocked.get(table.lower(), set()):
                return jsonify(
                    {
                        "success": False,
                        "error": f"Die Spalte {table}.{column} ist gesperrt",
                    }
                ), 400
    updated = sqlcfg.SqlConfig(**{**cfg.__dict__, "value_columns": chosen})
    try:
        sqlcfg.validate(updated)
    except sqlcfg.SqlConfigError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    update_sql_source(knowledge_id, updated.to_dict())
    return jsonify(_values_overview(get_knowledge(knowledge_id)))


@bp.route("/knowledge/<int:knowledge_id>/sql-values/build", methods=["POST"])
@require_role("admin")
def sql_values_build(knowledge_id):
    from app.knowledge.sql import valueindex

    k = _sql_knowledge_or_404(knowledge_id)
    if not sqlcfg.parse_config(k["source_config"]).value_columns:
        return jsonify(
            {"success": False, "error": "Erst Spalten auswählen und speichern"}
        ), 400
    if not valueindex.start_build(knowledge_id, k["source_config"]):
        return jsonify({"success": False, "error": "Der Aufbau läuft bereits"}), 409
    return jsonify({"success": True, **valueindex.status(knowledge_id)}), 202


@bp.route("/knowledge/<int:knowledge_id>/sql-values/search", methods=["POST"])
@require_role("admin")
def sql_values_search(knowledge_id):
    """Probe: welche Werte findet die Suche zu einem Begriff (wie das Modell sie sieht)."""
    from app.knowledge.sql import valueindex

    _sql_knowledge_or_404(knowledge_id)
    term = str((request.get_json() or {}).get("term") or "")
    hits = valueindex.search(knowledge_id, term)
    return jsonify(
        {
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
    )


@bp.route("/knowledge/external/test", methods=["POST"])
@require_role("admin")
def knowledge_external_test():
    """Fakten zu einer fremden Quelle zum Abgleich (nichts wird geraten oder gespeichert)."""
    data = request.get_json() or {}
    stored = None
    if data.get("knowledge_id"):
        found = get_knowledge(int(data["knowledge_id"]))
        stored = found if is_external(found) else None
    cfg = sources.parse_config(data.get("source"))
    if stored is not None and not cfg.api_key:
        old = sources.parse_config(stored.get("source_config"))
        if old.url == cfg.url:  # gespeicherter Key nie an eine andere Adresse
            cfg = sources.parse_config({**cfg.to_dict(), "api_key": old.api_key})
    try:
        facts = sources.probe_source(cfg)
    except sources.SourceError as e:
        return jsonify({"success": False, "error": str(e)}), 502
    return jsonify({"success": True, **facts})


@bp.route("/knowledge/external/filter-test", methods=["POST"])
@require_role("admin")
def knowledge_external_filter_test():
    """Zeigt, was ein Benutzer mit diesem Zugriffsfilter sehen wuerde (ohne zu speichern).

    Body: ``source`` (inkl. ``access``), optional ``knowledge_id`` (gespeicherter Key) und
    ``identity`` (``username``, ``display_name``, ``groups``, ``external_groups``).
    """
    data = request.get_json() or {}
    cfg = sources.parse_config(data.get("source"))
    if data.get("knowledge_id"):
        stored = get_knowledge(int(data["knowledge_id"]))
        if is_external(stored) and not cfg.api_key:
            old = sources.parse_config(stored.get("source_config"))
            if old.url == cfg.url:
                cfg = sources.parse_config({**cfg.to_dict(), "api_key": old.api_key})
    ident = data.get("identity") or {}
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
            return jsonify(
                {"success": True, "blocked": True, "count": 0, "samples": []}
            )
        client = sources.connect(cfg)
        try:
            count = int(
                client.count(cfg.collection, count_filter=flt, exact=True).count
            )
            points, _ = client.scroll(
                cfg.collection, scroll_filter=flt, limit=3, with_payload=True
            )
        finally:
            client.close()
    except sources.SourceError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "error": f"Abfrage fehlgeschlagen: {type(e).__name__}: {e}"[:300],
            }
        ), 502
    texts = cfg.fields["text"]
    return jsonify(
        {
            "success": True,
            "blocked": False,
            "count": count,
            "samples": [str((p.payload or {}).get(texts, ""))[:120] for p in points],
        }
    )


@bp.route("/knowledge/<int:knowledge_id>", methods=["PATCH"])
@require_role("user")
def knowledge_update(knowledge_id):
    if not can_access_knowledge(current_user(), knowledge_id, "write"):
        return jsonify({"success": False, "error": "Knowledge nicht gefunden"}), 404
    _abort_if_reindexing(knowledge_id)
    data = request.get_json() or {}
    current = get_knowledge(knowledge_id)
    if is_external(current) and ("source" in data or "embedding" in data):
        if not role_at_least(current_user()["role"], "admin"):
            return jsonify(
                {"success": False, "error": "Nur Admins ändern die Quelle"}
            ), 403
        try:
            body = {
                "source": data.get("source")
                or sources.parse_config(current["source_config"]).to_dict(),
                "embedding": data.get("embedding")
                or json.loads(current["embedding_spec"]),
            }
            cfg, spec, dimension, _f = _parse_external_body(body, stored=current)
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400
        update_external_source(knowledge_id, spec, dimension, cfg.to_dict())
    if is_sql(current) and "source" in data:
        if not role_at_least(current_user()["role"], "admin"):
            return jsonify(
                {"success": False, "error": "Nur Admins ändern die Quelle"}
            ), 403
        try:
            cfg = _sql_source_from(data, current)
            _sql_probe(cfg)
        except (sqlcfg.SqlConfigError, ValueError, RuntimeError) as e:
            return _sql_error(e)
        update_sql_source(knowledge_id, cfg.to_dict())
        from app.knowledge.sql import valueindex

        valueindex.prune(
            knowledge_id, cfg.to_dict()
        )  # bereits gespeicherte Werte gesperrter Spalten entfernen
    knowledge = update_knowledge(
        knowledge_id, data.get("name"), data.get("description")
    )
    if not knowledge:
        return jsonify({"success": False, "error": "Knowledge nicht gefunden"}), 404
    if not is_remote(knowledge):
        refresh_index(knowledge_id)  # Name steht im Verzeichnis
    return jsonify(_knowledge_with_counts(knowledge))


@bp.route("/knowledge/<int:knowledge_id>", methods=["DELETE"])
@require_role("user")
def knowledge_delete(knowledge_id):
    if not can_access_knowledge(current_user(), knowledge_id, "write"):
        return jsonify({"success": False, "error": "Knowledge nicht gefunden"}), 404
    status = current_app.extensions["ingestion_status"]
    if status["is_ingesting"] and status.get("knowledge_id") == knowledge_id:
        return jsonify({"success": False, "error": "Ingestion läuft gerade"}), 409
    _abort_if_reindexing(knowledge_id)

    knowledge = delete_knowledge(knowledge_id)
    if knowledge:
        from app.llm.connections import remove_knowledge_from_models

        remove_knowledge_from_models(knowledge_id)
    if not knowledge:
        return jsonify(
            {
                "success": False,
                "error": "Knowledge nicht gefunden oder Standard-Bereich "
                "(nicht löschbar)",
            }
        ), 404

    client = current_app.extensions.get("client")
    if client and not is_remote(knowledge):  # eine fremde Quelle wird nie angefasst
        try:
            client.delete_collection(knowledge["collection_name"])
        except Exception as e:
            logger.warning(f"Collection konnte nicht gelöscht werden: {e}")
    return jsonify({"success": True})


@bp.route("/knowledge/<int:knowledge_id>/reindex", methods=["POST"])
@require_role("user")
def knowledge_reindex(knowledge_id):
    if not can_access_knowledge(current_user(), knowledge_id, "write"):
        return jsonify({"success": False, "error": "Knowledge nicht gefunden"}), 404
    if is_remote(get_knowledge(knowledge_id)):
        return jsonify(
            {"success": False, "error": "Diese Quelle ist schreibgeschützt"}
        ), 409
    return _start_reindex([knowledge_id])


@bp.route("/reindex-all", methods=["POST"])
@require_role("admin")
def reindex_all():
    stale = [k["id"] for k in list_knowledge() if model_settings.is_stale(k)]
    if not stale:
        return jsonify({"success": True, "started": False, "message": "Alles aktuell"})
    return _start_reindex(stale)


@bp.route("/reindex/status", methods=["GET"])
@auth_required_api
def reindex_status():
    return jsonify(reindex.status())


def _start_reindex(ids: list[int]):
    client = current_app.extensions.get("client")
    sparse = current_app.extensions.get("sparse_embeddings")
    if not client or not sparse:
        return jsonify(
            {"success": False, "error": "RAG-System wird noch initialisiert"}
        ), 503
    if current_app.extensions["ingestion_status"]["is_ingesting"]:
        return jsonify({"success": False, "error": "Ingestion läuft gerade"}), 409
    try:
        reindex.start(ids, client, sparse)
    except reindex.ReindexBusy as e:
        return jsonify({"success": False, "error": str(e)}), 409
    return jsonify({"success": True, "started": True, "knowledge_ids": ids}), 202
