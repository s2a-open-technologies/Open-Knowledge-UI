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
JSON-API für die React-SPA.

Die Route-/Business-Logik wird wo möglich aus ``app.web.user_routes`` wiederverwendet
(Service-Tests, Konstanten, Datei-Helfer). Die Endpunkte hier liefern reine
JSON-Antworten (kein ``render_template``, keine Redirects) — die SPA rendert
clientseitig.
"""

import json
import os
from datetime import UTC, datetime
from functools import wraps
from logging import getLogger
from pathlib import Path
from urllib.parse import quote

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    g,
    jsonify,
    make_response,
    redirect,
    request,
    send_file,
    session,
)
from qdrant_client import models

from app.auth.access import (
    accessible_knowledge_ids,
    add_grant,
    can_access_knowledge,
)
from app.auth.auth import auth_required_api, current_user, require_role
from app.auth.users import (
    LoginThrottledError,
    authenticate,
    role_at_least,
    set_password,
    verify_password,
)
from app.chat import data_export, my_uploads
from app.chat import summaries as summaries_store
from app.core.database_service import get_db_service
from app.core.settings_service import SettingsService
from app.ingest.index_doc import ensure_index, get_index, refresh_index
from app.rag import model_settings
from app.rag.knowledge import (
    create_external_knowledge,
    create_knowledge,
    create_sql_knowledge,
    delete_knowledge,
    get_knowledge,
    is_external,
    is_remote,
    is_sql,
    knowledge_paths,
    list_knowledge,
    update_external_source,
    update_knowledge,
    update_sql_source,
)
from app.sqlsource import config as sqlcfg
from app.utils.file_manager import check_data_file_exists, delete_associated_file
from app.vector import reindex, sources

api_bp = Blueprint("api", __name__, url_prefix="/api")

logger = getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _setting_to_dict(setting) -> dict:
    return {
        "key": setting.key,
        "value": setting.value,
        "default_value": setting.default_value,
        "category": setting.category,
        "is_sensitive": setting.is_sensitive,
        "description": setting.description,
    }


def _summaries_dir() -> Path:
    settings = SettingsService(get_db_service().get_session())
    return Path(settings.get("SUMMARIES_DIR", "./data/summaries"))


def _get_data_dir() -> str:
    settings = SettingsService(get_db_service().get_session())
    return settings.get("DATA_DIR", "./data")


def _abort_if_reindexing(knowledge_id: int) -> None:
    from app.vector import reindex

    if reindex.is_locked(knowledge_id):
        abort(
            make_response(
                jsonify(
                    {
                        "success": False,
                        "error": "Das Knowledge wird gerade neu indexiert",
                    }
                ),
                409,
            )
        )


def _current_knowledge() -> dict:
    """Knowledge-Bereich des Requests (``knowledge_id`` als Query-/Form-/JSON-Feld);
    ohne Angabe gilt der Standard-Bereich. Pro Request gecacht."""
    cached = getattr(g, "_knowledge", None)
    if cached is not None:
        return cached
    kid = request.args.get("knowledge_id", type=int)
    if kid is None:
        kid = request.form.get("knowledge_id", type=int)
    if kid is None and request.is_json:
        raw = (request.get_json(silent=True) or {}).get("knowledge_id")
        kid = int(raw) if str(raw or "").isdigit() else None
    knowledge = get_knowledge(kid)
    # Lesende Aufrufe brauchen Lese-, aendernde Schreibrechte auf dieses Knowledge.
    level = "read" if request.method in ("GET", "HEAD") else "write"
    if knowledge is None or not can_access_knowledge(
        current_user(), knowledge["id"], level
    ):
        abort(
            make_response(
                jsonify({"success": False, "error": "Knowledge nicht gefunden"}), 404
            )
        )
    if is_remote(knowledge) and (
        level == "write" or request.path.startswith("/api/vectordb")
    ):
        abort(
            make_response(
                jsonify(
                    {"success": False, "error": "Diese Quelle ist schreibgeschützt"}
                ),
                409,
            )
        )
    if level == "write":
        _abort_if_reindexing(knowledge["id"])
    g._knowledge = knowledge
    return knowledge


def knowledge_scoped(f):
    """Prueft den Zugriff auf das Knowledge des Requests *vor* dem eigentlichen
    Handler. Die Pruefung allein in ``_kb_col()`` wuerde teils erst in einem
    ``try/except Exception`` greifen (dort wird der Abbruch zu einem 500)."""

    @wraps(f)
    def decorated(*args, **kwargs):
        _current_knowledge()
        return f(*args, **kwargs)

    return decorated


def _kb_col() -> str:
    return _current_knowledge()["collection_name"]


def _get_collection_and_embeddings():
    """Liefert (dense_embeddings, sparse, collection) oder None, wenn die
    Embedding-Ressourcen noch nicht geladen sind."""
    from app import resources_ready

    if not resources_ready():
        return None
    from app.rag import model_settings

    sparse = current_app.extensions.get("sparse_embeddings")
    if not sparse:
        return None
    # Modell des aktuellen Knowledge (nicht blind das globale)
    return (
        model_settings.embedder_for(_current_knowledge()),
        sparse,
        "open_knowledge_ui",
    )


def _index_data_references_best_effort() -> None:
    """Indexiert Verweis-Chunks für alle strukturierten Dateien. Fehler werden
    geloggt, aber nie an den Upload-Request weitergereicht."""
    try:
        resources = _get_collection_and_embeddings()
        if resources is None:
            return
        dense, sparse, _ = resources
        from app.rag import sparse as sparse_lang
        from app.vector.ingestion import index_data_references

        knowledge = _current_knowledge()
        paths = knowledge_paths(knowledge)
        index_data_references(
            dense,
            sparse_lang.resolve(sparse, sparse_lang.knowledge_language(knowledge)),
            knowledge["collection_name"],
            str(paths.data_dir),
            name_prefix=paths.name_prefix,
        )
    except Exception as e:
        logger.warning(f"Verweis-Indexierung nach Upload fehlgeschlagen: {e}")


# =============================================================================
# Auth
# =============================================================================


@api_bp.route("/version", methods=["GET"])
def app_version():
    """Version der Anwendung (öffentlich, z. B. für Support und Monitoring)."""
    from app.core.version import get_version

    return jsonify({"version": get_version()})


@api_bp.route("/auth/me", methods=["GET"])
@auth_required_api
def auth_me():
    return jsonify({"authenticated": True, "user": current_user()})


@api_bp.route("/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json() or {}
    try:
        user = authenticate(
            data.get("username", ""),
            data.get("password", ""),
            client_ip=request.remote_addr,
        )
    except LoginThrottledError:
        return jsonify(
            {
                "success": False,
                "error": "Zu viele Fehlversuche. Bitte in einigen Minuten erneut versuchen.",
            }
        ), 429

    if user is None:
        return jsonify(
            {"success": False, "error": "Benutzername oder Passwort falsch"}
        ), 401

    # Neue Session statt Wiederverwendung (verhindert Session-Fixation).
    session.clear()
    session["user_id"] = user["id"]
    return jsonify({"success": True, "user": user})


@api_bp.route("/auth/providers", methods=["GET"])
def auth_providers():
    """Welche externen Anmeldungen der Login-Bildschirm anbieten soll (ohne Anmeldung abrufbar)."""
    from app.auth import auth_external

    cfg = auth_external.load_config()
    return jsonify(
        {
            "ldap": bool(cfg["ldap"]["enabled"]),
            "oidc": {
                "enabled": bool(cfg["oidc"]["enabled"]),
                "name": cfg["oidc"]["name"] or "SSO",
            },
        }
    )


def _public_base_url() -> str:
    from app.auth import auth_external

    return auth_external.load_config()["base_url"] or request.host_url.rstrip("/")


@api_bp.route("/auth/oidc/login", methods=["GET"])
def auth_oidc_login():
    from app.auth import auth_external, auth_oidc

    cfg = auth_external.load_config()["oidc"]
    if not cfg["enabled"]:
        return redirect("/?login_error=oidc_off")
    try:
        return redirect(auth_oidc.start_login(cfg, _public_base_url(), session))
    except auth_oidc.OidcError as e:
        logger.warning(f"OIDC-Start fehlgeschlagen: {e}")
        return redirect("/?login_error=oidc")


@api_bp.route("/auth/oidc/callback", methods=["GET"])
def auth_oidc_callback():
    from app.auth import auth_external, auth_oidc

    cfg = auth_external.load_config()["oidc"]
    if not cfg["enabled"]:
        return redirect("/?login_error=oidc_off")
    try:
        found = auth_oidc.finish_login(cfg, _public_base_url(), session, request.args)
        user = auth_external.login_external_user(
            "oidc",
            found["external_id"],
            found["username"],
            found["display_name"],
            found["groups"],
        )
    except (auth_oidc.OidcError, auth_external.ExternalLoginError) as e:
        logger.warning(f"OIDC-Anmeldung abgelehnt: {e}")
        session.pop("oidc", None)
        return redirect("/?login_error=oidc")
    session.clear()  # neue Session (verhindert Session-Fixation)
    session["user_id"] = user["id"]
    return redirect("/")


@api_bp.route("/auth/logout", methods=["POST"])
def auth_logout():
    from app.auth import audit

    audit.record(audit.LOGOUT)
    session.clear()
    return jsonify({"success": True})


@api_bp.route("/admin/audit", methods=["GET"])
@require_role("admin")
def admin_audit():
    """Neueste Protokolleinträge (Anmeldungen, Verwaltung, Freigaben)."""
    from app.auth import audit

    return jsonify(
        {
            "events": audit.list_events(
                limit=request.args.get("limit", type=int) or 100,
                offset=request.args.get("offset", type=int) or 0,
                action=request.args.get("action") or None,
            )
        }
    )


# =============================================================================
# Knowledge-Bereiche
# =============================================================================


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


@api_bp.route("/knowledge", methods=["GET"])
@auth_required_api
def knowledge_list():
    readable = accessible_knowledge_ids(current_user(), "read")
    return jsonify(
        [_knowledge_with_counts(k) for k in list_knowledge() if k["id"] in readable]
    )


@api_bp.route("/knowledge", methods=["POST"])
@require_role("user")
def knowledge_create():
    from app.vector.collections import ensure_collection

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


@api_bp.route("/knowledge/<int:knowledge_id>/index", methods=["GET"])
@auth_required_api
def knowledge_index(knowledge_id):
    """Das Inhaltsverzeichnis, das das Modell zur Orientierung bekommt (lesen genuegt)."""
    knowledge = get_knowledge(knowledge_id)
    if not knowledge or not can_access_knowledge(current_user(), knowledge_id, "read"):
        return jsonify({"success": False, "error": "Knowledge nicht gefunden"}), 404
    return jsonify(_index_payload(knowledge))


@api_bp.route("/knowledge/<int:knowledge_id>/index/refresh", methods=["POST"])
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


@api_bp.route("/knowledge/<int:knowledge_id>", methods=["GET"])
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
    from app.toolworker import client as worker
    from app.toolworker import config as worker_config

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
    from app.sqlsource import valueindex

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


@api_bp.route("/knowledge/sql/test", methods=["POST"])
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


# --- Wertverzeichnis einer SQL-Quelle (nur Admins) ---------------------------------------------------


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
    from app.sqlsource import valueindex
    from app.toolworker import config as worker_config

    cfg = sqlcfg.parse_config(k["source_config"])
    max_distinct, max_results = worker_config.values_settings()
    return {
        **valueindex.status(k["id"]),
        "columns": {t: list(c) for t, c in cfg.value_columns.items()},
        "max_distinct": max_distinct,
        "max_results": max_results,
    }


@api_bp.route("/knowledge/<int:knowledge_id>/sql-values", methods=["GET"])
@require_role("admin")
def sql_values_overview(knowledge_id):
    return jsonify(_values_overview(_sql_knowledge_or_404(knowledge_id)))


@api_bp.route("/knowledge/<int:knowledge_id>/sql-values/suggest", methods=["POST"])
@require_role("admin")
def sql_values_suggest(knowledge_id):
    """Textspalten mit Vorschlag (aufnehmen oder nicht, mit Begruendung). Liest nur Zaehlungen aus der Quelle."""
    from app.sqlsource import valueindex

    k = _sql_knowledge_or_404(knowledge_id)
    try:
        return jsonify(
            {"success": True, "columns": valueindex.profile(k["source_config"])}
        )
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 502


@api_bp.route("/knowledge/<int:knowledge_id>/sql-values", methods=["PUT"])
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


@api_bp.route("/knowledge/<int:knowledge_id>/sql-values/build", methods=["POST"])
@require_role("admin")
def sql_values_build(knowledge_id):
    from app.sqlsource import valueindex

    k = _sql_knowledge_or_404(knowledge_id)
    if not sqlcfg.parse_config(k["source_config"]).value_columns:
        return jsonify(
            {"success": False, "error": "Erst Spalten auswählen und speichern"}
        ), 400
    if not valueindex.start_build(knowledge_id, k["source_config"]):
        return jsonify({"success": False, "error": "Der Aufbau läuft bereits"}), 409
    return jsonify({"success": True, **valueindex.status(knowledge_id)}), 202


@api_bp.route("/knowledge/<int:knowledge_id>/sql-values/search", methods=["POST"])
@require_role("admin")
def sql_values_search(knowledge_id):
    """Probe: welche Werte findet die Suche zu einem Begriff (wie das Modell sie sieht)."""
    from app.sqlsource import valueindex

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


@api_bp.route("/knowledge/external/test", methods=["POST"])
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


@api_bp.route("/knowledge/external/filter-test", methods=["POST"])
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


@api_bp.route("/knowledge/<int:knowledge_id>", methods=["PATCH"])
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
        from app.sqlsource import valueindex

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


@api_bp.route("/knowledge/<int:knowledge_id>", methods=["DELETE"])
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


# =============================================================================
# Chat-Anhaenge (an die Chat-Session gebunden)
# =============================================================================


@api_bp.route("/knowledge/<int:knowledge_id>/reindex", methods=["POST"])
@require_role("user")
def knowledge_reindex(knowledge_id):
    if not can_access_knowledge(current_user(), knowledge_id, "write"):
        return jsonify({"success": False, "error": "Knowledge nicht gefunden"}), 404
    if is_remote(get_knowledge(knowledge_id)):
        return jsonify(
            {"success": False, "error": "Diese Quelle ist schreibgeschützt"}
        ), 409
    return _start_reindex([knowledge_id])


@api_bp.route("/reindex-all", methods=["POST"])
@require_role("admin")
def reindex_all():
    stale = [k["id"] for k in list_knowledge() if model_settings.is_stale(k)]
    if not stale:
        return jsonify({"success": True, "started": False, "message": "Alles aktuell"})
    return _start_reindex(stale)


@api_bp.route("/reindex/status", methods=["GET"])
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


@api_bp.route("/chat/attachments", methods=["POST"])
@auth_required_api
def upload_chat_attachment():
    from app import resources_ready
    from app.chat.attachments import is_image, is_supported, start_attachment_ingestion

    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"success": False, "error": "Keine Datei"}), 400
    if not is_supported(file.filename):
        return jsonify(
            {
                "success": False,
                "error": "Dateityp im Chat nicht unterstützt "
                "(PDF, DOCX, DOC, ODT, MD, Bilder). CSV/Excel bitte über Knowledge hochladen.",
            }
        ), 415
    if is_image(file.filename):
        from app.ingest.image_text import enabled

        if not enabled():
            return jsonify(
                {
                    "success": False,
                    "error": "Bilder im Chat sind abgeschaltet (Administration › Einlesen & Modelle).",
                }
            ), 415

    sparse = current_app.extensions.get("sparse_embeddings")
    if not resources_ready() or not sparse:
        return jsonify(
            {
                "success": False,
                "error": "RAG-System wird noch initialisiert. Bitte kurz warten.",
            }
        ), 503

    db = get_db_service()
    session_id = request.form.get("session_id") or None
    user_id = current_user()["id"]
    if session_id and not db.get_chat_session(session_id, user_id=user_id):
        return jsonify({"success": False, "error": "Ungültige Session"}), 404
    if not session_id:
        session_id = db.create_session(user_id=user_id).id

    attachment = start_attachment_ingestion(
        file, session_id, sparse, current_user()["id"]
    )
    return jsonify({"success": True, "session_id": session_id, **attachment}), 202


@api_bp.route("/chat/attachments/<attachment_id>", methods=["GET"])
@auth_required_api
def chat_attachment_status(attachment_id):
    db = get_db_service()
    att = db.get_attachment(attachment_id)
    # Fremde Anhaenge sind nicht auffindbar (404 statt 403).
    if not att or not db.get_chat_session(att.session_id, user_id=current_user()["id"]):
        return jsonify({"error": "Anhang nicht gefunden"}), 404
    return jsonify(
        {
            "id": att.id,
            "session_id": att.session_id,
            "filename": att.filename,
            "status": att.status,
            "error": att.error,
        }
    )


# =============================================================================
# Documents (file-level)
# =============================================================================


@api_bp.route("/documents", methods=["GET"])
@require_role("user")
@knowledge_scoped
def documents():
    db = get_db_service()
    knowledge = _current_knowledge()
    user_docs = db.get_all_documents(knowledge_id=knowledge["id"])

    client = current_app.extensions.get("client")
    result = []

    for user_doc in user_docs:
        entry = {
            "id": user_doc.id,
            "document_id": user_doc.document_id,
            "filename": user_doc.filename,
            "uploaded_at": user_doc.uploaded_at.isoformat()
            if user_doc.uploaded_at
            else None,
            "content_preview": "[Nicht mehr in Qdrant]",
            "source": "N/A",
            "has_data_file": False,
            "has_markdown": _markdown_path(user_doc) is not None,
        }

        if client:
            try:
                points, _ = client.scroll(
                    collection_name=_kb_col(),
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_group_id",
                                match=models.MatchValue(value=user_doc.document_id),
                            )
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
                if points:
                    payload = points[0].payload or {}
                    entry["content_preview"] = (
                        payload.get("page_content", "")
                        or payload.get("original_content", "")
                    )[:100] + "..."
                    entry["source"] = payload.get("source", "N/A")
                    entry["has_data_file"] = check_data_file_exists(user_doc.filename)
            except Exception as e:
                entry["content_preview"] = f"[Fehler: {e!s}]"

        result.append(entry)

    return jsonify(result)


@api_bp.route("/documents/<int:doc_id>", methods=["DELETE"])
@require_role("user")
def delete_document(doc_id):
    from app.core.models import UserDocument

    db_session = get_db_service().get_session()
    try:
        user_doc = db_session.query(UserDocument).filter_by(id=doc_id).first()
        if user_doc:
            db_session.expunge(user_doc)
    finally:
        db_session.close()
    if not user_doc:
        return jsonify({"success": False, "error": "Dokument nicht gefunden"}), 404

    # Collection des Bereichs, dem das Dokument gehoert (nicht des Requests).
    doc_knowledge = get_knowledge(user_doc.knowledge_id) or _current_knowledge()
    if reindex.is_locked(doc_knowledge["id"]):
        return jsonify(
            {"success": False, "error": "Das Knowledge wird gerade neu indexiert"}
        ), 409
    user = current_user()
    # Schreibrecht, oder: wer das Dokument hochgeladen hat, darf es auch loeschen (und sieht das Knowledge)
    uploader = user_doc.uploaded_by is not None and user_doc.uploaded_by == user["id"]
    if not (
        can_access_knowledge(user, doc_knowledge["id"], "write")
        or (uploader and can_access_knowledge(user, doc_knowledge["id"], "read"))
    ):
        return jsonify({"success": False, "error": "Dokument nicht gefunden"}), 404
    try:
        my_uploads.delete_knowledge_document(
            user_doc, doc_knowledge, current_app.extensions.get("client")
        )
    except my_uploads.UploadError as e:
        return jsonify({"success": False, "error": e.message}), e.status
    return jsonify({"success": True})


MARKDOWN_ROOT = Path("./data/markdown")


def _markdown_path(user_doc) -> Path | None:
    return my_uploads.markdown_path(user_doc)


# =============================================================================
# Meine Uploads (eigene Dateien als Markdown ansehen und loeschen)
# =============================================================================


def _upload_error(e):
    return jsonify({"success": False, "error": e.message}), e.status


@api_bp.route("/my-uploads", methods=["GET"])
@auth_required_api
def my_uploads_list():
    return jsonify(my_uploads.list_uploads(current_user()["id"]))


@api_bp.route("/my-uploads/<kind>/<item_id>/markdown", methods=["GET"])
@auth_required_api
def my_uploads_markdown(kind, item_id):
    try:
        row = my_uploads.get_owned(kind, item_id, current_user()["id"])
    except my_uploads.UploadError as e:
        return _upload_error(e)
    text = my_uploads.read_markdown(kind, row)
    if text is None:
        return jsonify({"success": False, "error": "Kein Markdown gespeichert"}), 404
    name = Path(row.filename).stem
    if request.args.get("download") == "1":
        return Response(
            text,
            mimetype="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}.md"
            },
        )
    return jsonify({"filename": f"{name}.md", "markdown": text})


@api_bp.route("/my-uploads/<kind>/<item_id>", methods=["DELETE"])
@auth_required_api
def my_uploads_delete(kind, item_id):
    try:
        row = my_uploads.get_owned(kind, item_id, current_user()["id"])
        my_uploads.delete_upload(kind, row, current_app.extensions.get("client"))
    except my_uploads.UploadError as e:
        return _upload_error(e)
    return jsonify({"success": True})


@api_bp.route("/documents/<int:doc_id>/markdown", methods=["GET"])
@require_role("user")
def document_markdown(doc_id):
    """Das beim Einlesen erzeugte Markdown (so sieht der Splitter das Dokument).
    ``?download=1`` liefert eine Datei."""
    from app.core.models import UserDocument

    db_session = get_db_service().get_session()
    try:
        user_doc = db_session.query(UserDocument).filter_by(id=doc_id).first()
        knowledge = get_knowledge(user_doc.knowledge_id) if user_doc else None
        if (
            not user_doc
            or not knowledge
            or not can_access_knowledge(current_user(), knowledge["id"], "read")
        ):
            return jsonify({"success": False, "error": "Dokument nicht gefunden"}), 404
        path = _markdown_path(user_doc)
        filename = user_doc.filename
    finally:
        db_session.close()

    if path is None:
        return jsonify(
            {
                "success": False,
                "error": "Für dieses Dokument ist kein Markdown gespeichert "
                "(früher eingelesen). Es entsteht beim Neu-Indexieren.",
            }
        ), 404

    text = path.read_text(encoding="utf-8")
    if request.args.get("download") == "1":
        return Response(
            text,
            mimetype="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(Path(filename).stem)}.md"
            },
        )
    return jsonify({"filename": filename, "markdown": text})


# =============================================================================
# Upload / Ingestion
# =============================================================================


@api_bp.route("/documents/upload", methods=["POST"])
@require_role("user")
@knowledge_scoped
def upload_document():
    from app.utils.file_manager import store_uploaded_files

    result = store_uploaded_files(
        request.files.getlist("files"),
        paths=knowledge_paths(_current_knowledge()),
        uploader_id=current_user()["id"],
    )

    if result["duplicate"]:
        return jsonify(
            {
                "success": False,
                "error": (
                    f'Datei "{result["duplicate"]}" wurde bereits hochgeladen und '
                    f"verarbeitet. Bitte zuerst das bestehende Dokument löschen, "
                    f"um es zu ersetzen."
                ),
            }
        ), 409

    documents = result["documents"]
    data_files = result["data_files"]

    if not documents and not data_files:
        return jsonify({"success": False, "error": "Keine gültigen Dateien"}), 400

    if data_files and not documents:
        message = (
            f"{len(data_files)} strukturierte Datei(en) direkt verfügbar "
            f"(Excel/CSV, keine Ingestion nötig)."
        )
    elif documents and not data_files:
        message = f"{len(documents)} Datei(en) hochgeladen. Bitte Ingestion starten."
    else:
        message = (
            f"{len(documents)} Datei(en) hochgeladen (Ingestion starten), "
            f"{len(data_files)} strukturierte Datei(en) sofort verfügbar."
        )

    # Referenz-Chunks für neu hochgeladene strukturierte Daten best-effort
    # indexieren, damit sie ohne zusätzliche Ingestion semantisch auffindbar
    # sind. Falls die Embedding-Ressourcen noch nicht geladen sind, werden sie
    # beim nächsten "Ingestion starten" nachgezogen.
    if data_files:
        _index_data_references_best_effort()
        refresh_index(_current_knowledge()["id"])

    return jsonify(
        {
            "success": True,
            "documents": documents,
            "data_files": data_files,
            "skipped": result["skipped"],
            "files": documents + data_files,
            "message": message,
        }
    )


@api_bp.route("/documents/upload/clear", methods=["POST"])
@require_role("user")
@knowledge_scoped
def clear_uploaded_files():
    target_dir = knowledge_paths(_current_knowledge()).files_dir
    deleted = 0
    errors = []

    if target_dir.exists():
        for file_path in target_dir.iterdir():
            if not file_path.is_file():
                continue
            try:
                file_path.unlink()
                deleted += 1
            except Exception as e:
                errors.append(str(e))

    return jsonify({"success": True, "deleted": deleted, "errors": errors})


# =============================================================================
# Strukturierte Daten (CSV/Excel) — nur Pandas-Referenz, keine Vektoren
# =============================================================================


@api_bp.route("/data-files", methods=["GET"])
@require_role("user")
@knowledge_scoped
def data_files():
    from app.utils.file_manager import list_data_files

    return jsonify(list_data_files(knowledge_paths(_current_knowledge()).data_dir))


DATA_PREVIEW_ROWS = 200


def _data_file_path(filename: str) -> Path | None:
    """Strukturierte Datei im Datenordner des aktuellen Knowledge (nie ausserhalb)."""
    from app.utils.file_manager import STRUCTURED_EXTS

    folder = knowledge_paths(_current_knowledge()).data_dir.resolve()
    path = (folder / os.path.basename(filename)).resolve()
    if (
        path.parent != folder
        or path.suffix.lower() not in STRUCTURED_EXTS
        or not path.is_file()
    ):
        return None
    return path


@api_bp.route("/data-files/<path:filename>", methods=["GET"])
@require_role("user")
@knowledge_scoped
def data_file_view(filename):
    """Vorschau (erste Zeilen) einer CSV/Excel-Datei; ``?download=1`` liefert die Datei selbst."""
    path = _data_file_path(filename)
    if path is None:
        return jsonify({"success": False, "error": "Datei nicht gefunden"}), 404
    if request.args.get("download") == "1":
        return send_file(path, as_attachment=True, download_name=path.name)

    from app.vector.ingestion import _load_dataframe

    try:
        df = _load_dataframe(str(path))
    except Exception as e:
        return jsonify({"success": False, "error": f"Datei nicht lesbar: {e}"}), 422
    head = df.head(DATA_PREVIEW_ROWS)
    return jsonify(
        {
            "filename": path.name,
            "columns": [str(c) for c in df.columns],
            "rows": [
                ["" if v != v else str(v) for v in row]
                for row in head.itertuples(index=False)
            ],
            "total_rows": len(df),
            "truncated": len(df) > DATA_PREVIEW_ROWS,
        }
    )


@api_bp.route("/data-files/<path:filename>", methods=["DELETE"])
@require_role("user")
@knowledge_scoped
def delete_data_file_route(filename):
    from app.utils.file_manager import delete_data_file

    paths = knowledge_paths(_current_knowledge())
    if not delete_data_file(filename, data_dir=paths.data_dir):
        return jsonify({"success": False, "error": "Datei nicht gefunden"}), 404

    # Zugehörigen Verweis-Chunk aus der Vektordatenbank entfernen (best-effort).
    try:
        from app.vector.ingestion import delete_data_reference

        delete_data_reference(_kb_col(), f"{paths.name_prefix}{filename}")
    except Exception as e:
        logger.warning(
            f"Verweis-Chunk für {filename} konnte nicht gelöscht werden: {e}"
        )

    refresh_index(_current_knowledge()["id"])
    return jsonify({"success": True})


@api_bp.route("/documents/ingest", methods=["POST"])
@require_role("user")
@knowledge_scoped
def trigger_ingestion():
    from app import wait_for_resources
    from app.vector import DocumentIngestion

    try:
        if not wait_for_resources(timeout=180):
            return jsonify(
                {
                    "success": False,
                    "error": "RAG-System wird noch initialisiert (Embedding-Modell "
                    "lädt). Bitte in wenigen Sekunden erneut versuchen.",
                }
            ), 503
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 503

    knowledge = _current_knowledge()
    status = current_app.extensions["ingestion_status"]
    if status["is_ingesting"]:
        return jsonify({"success": False, "error": "Ingestion läuft bereits"}), 409

    status["is_ingesting"] = True
    status["knowledge_id"] = knowledge["id"]
    try:
        ingest = DocumentIngestion(
            sparse_embeddings=current_app.extensions.get("sparse_embeddings"),
            knowledge=knowledge,
        )
        ingest.ingest_documents()
        # Fehler und Warnungen einzelner Dateien gehen an die Oberflaeche (nichts schlaegt stumm fehl).
        return jsonify(
            {
                "success": not ingest.errors or ingest.processed > 0,
                "message": f"{ingest.processed} Datei(en) eingelesen"
                + (f", {len(ingest.errors)} fehlgeschlagen" if ingest.errors else ""),
                "processed": ingest.processed,
                "errors": ingest.errors,
                "warnings": ingest.warnings,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        status["is_ingesting"] = False
        status["knowledge_id"] = None


# =============================================================================
# Vector DB (chunk-level)
# =============================================================================


@api_bp.route("/vectordb", methods=["GET"])
@require_role("user")
@knowledge_scoped
def vectordb():
    client = current_app.extensions.get("client")
    if not client:
        return jsonify({"error": "Qdrant Client nicht initialisiert"}), 503

    limit = request.args.get("limit", 20, type=int)
    offset = request.args.get("offset", None)
    filename_filter = request.args.get("filename", "").strip()
    file_status_filter = request.args.get("file_status", "").strip()

    try:
        scroll_filter = None
        if filename_filter:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename", match=models.MatchValue(value=filename_filter)
                    )
                ]
            )

        points, next_offset = client.scroll(
            collection_name=_kb_col(),
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=scroll_filter,
        )

        documents = []
        for point in points:
            payload = point.payload or {}
            filename = payload.get("filename", "N/A")
            has_data_file = (
                check_data_file_exists(filename) if filename != "N/A" else False
            )

            if file_status_filter == "has_file" and not has_data_file:
                continue
            if file_status_filter == "no_file" and has_data_file:
                continue

            documents.append(
                {
                    "id": str(point.id),
                    "filename": filename,
                    "source": payload.get("source", "N/A"),
                    "content_preview": (
                        payload.get("page_content", "")
                        or payload.get("original_content", "")
                    )[:50]
                    + "...",
                    "has_data_file": has_data_file,
                }
            )

        all_points, _ = client.scroll(
            collection_name=_kb_col(),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        filenames = sorted(
            {p.payload.get("filename", "N/A") for p in all_points if p.payload}
        )
    except Exception as e:
        return jsonify({"error": f"Fehler beim Laden der Dokumente: {e!s}"}), 500

    return jsonify(
        {
            "documents": documents,
            "next_offset": next_offset,
            "filenames": filenames,
            "limit": limit,
        }
    )


@api_bp.route("/vectordb/<doc_id>", methods=["GET"])
@require_role("user")
@knowledge_scoped
def vectordb_detail(doc_id):
    client = current_app.extensions.get("client")
    if not client:
        return jsonify({"error": "Qdrant Client nicht initialisiert"}), 503

    try:
        result = client.retrieve(
            collection_name=_kb_col(),
            ids=[doc_id],
            with_payload=True,
            with_vectors=False,
        )
        if not result:
            return jsonify({"error": "Dokument nicht gefunden"}), 404

        point = result[0]
        payload = point.payload or {}
        document = {
            "id": str(point.id),
            "page_content": payload.get("page_content", ""),
            "original_content": payload.get("original_content", ""),
            "metadata": {
                k: v
                for k, v in payload.items()
                if k not in ["page_content", "original_content"]
            },
        }
    except Exception as e:
        return jsonify({"error": f"Fehler beim Laden des Dokuments: {e!s}"}), 500

    return jsonify(document)


@api_bp.route("/vectordb/<doc_id>", methods=["DELETE"])
@require_role("user")
@knowledge_scoped
def delete_vectordb_document(doc_id):
    client = current_app.extensions.get("client")
    if not client:
        return jsonify(
            {"success": False, "error": "Qdrant Client nicht initialisiert"}
        ), 503

    try:
        result = client.retrieve(
            collection_name=_kb_col(),
            ids=[doc_id],
            with_payload=True,
            with_vectors=False,
        )
        payload = result[0].payload if result else None

        client.delete(
            collection_name=_kb_col(),
            points_selector=models.PointIdsList(points=[doc_id]),
        )

        if payload:
            delete_associated_file(payload)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/vectordb/batch-delete", methods=["POST"])
@require_role("user")
@knowledge_scoped
def vectordb_batch_delete():
    client = current_app.extensions.get("client")
    if not client:
        return jsonify(
            {"success": False, "error": "Qdrant Client nicht initialisiert"}
        ), 503

    data = request.get_json() or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"success": False, "error": "Filename erforderlich"}), 400

    try:
        points, _ = client.scroll(
            collection_name=_kb_col(),
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename", match=models.MatchValue(value=filename)
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        if points:
            if points[0].payload:
                delete_associated_file(points[0].payload)

            point_ids = [str(p.id) for p in points]
            client.delete(
                collection_name=_kb_col(),
                points_selector=models.PointIdsList(points=point_ids),
            )

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/vectordb/clear", methods=["POST"])
@require_role("user")
@knowledge_scoped
def vectordb_clear_all():
    client = current_app.extensions.get("client")
    if not client:
        return jsonify(
            {"success": False, "error": "Qdrant Client nicht initialisiert"}
        ), 503

    data = request.get_json() or {}
    if data.get("confirmation", "") != "DELETE_ALL":
        return jsonify({"success": False, "error": "Bestätigung erforderlich"}), 400

    try:
        from app.vector.collections import ensure_collection

        client.delete_collection(_kb_col())
        ensure_collection(
            client, _kb_col(), _current_knowledge()["embedding_dimension"]
        )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Konto
# =============================================================================


@api_bp.route("/settings/change-password", methods=["POST"])
@auth_required_api
def change_password():
    data = request.get_json() or {}
    old_password = data.get("old_password", "")
    new_password = data.get("new_password", "")

    if not new_password or len(new_password) < 4:
        return jsonify(
            {"success": False, "error": "Passwort muss mindestens 4 Zeichen haben"}
        ), 400

    user = current_user()
    if not verify_password(user["id"], old_password):
        return jsonify({"success": False, "error": "Altes Passwort ist falsch"}), 403

    set_password(user["id"], new_password)
    from app.auth import audit

    audit.record(
        audit.PASSWORD_CHANGE,
        actor_user_id=user["id"],
        object_type="user",
        object_id=user["id"],
    )
    return jsonify({"success": True})


@api_bp.route("/me/export", methods=["GET"])
@auth_required_api
def export_my_data():
    """Alle persoenlichen Daten des angemeldeten Kontos als JSON (Auskunft/Übertragbarkeit)."""
    user = current_user()
    try:
        data = data_export.build_export(user["id"])
    except LookupError:
        return jsonify({"success": False, "error": "Konto nicht gefunden"}), 404
    response = Response(
        data_export.to_response_bytes(data),
        content_type="application/json",
    )
    response.headers["Content-Disposition"] = (
        f'attachment; filename="{data_export.export_filename(user["username"])}"'
    )
    from app.auth import audit

    audit.record(
        audit.DATA_EXPORT,
        actor_user_id=user["id"],
        object_type="user",
        object_id=user["id"],
    )
    return response


# =============================================================================
# Summaries
# =============================================================================


@api_bp.route("/summaries", methods=["GET"])
@auth_required_api
def summaries():
    user = current_user()
    if role_at_least(user["role"], "admin"):
        summaries_store.adopt_legacy(
            user["id"]
        )  # Altbestand einmalig dem Admin zuordnen
    summaries_dir = summaries_store.user_dir(user["id"])
    result = []

    if summaries_dir.exists():
        for file_path in sorted(summaries_dir.glob("*.md")):
            stat = file_path.stat()
            result.append(
                {
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "size_human": _format_file_size(stat.st_size),
                    "modified": stat.st_mtime,
                    "created_at": datetime.fromtimestamp(
                        stat.st_ctime, tz=UTC
                    ).isoformat(),
                }
            )

    return jsonify(result)


def _safe_summary_path(filename: str) -> Path | None:
    """Datei im Analysen-Ordner des angemeldeten Nutzers (nie die anderer Nutzer)."""
    return summaries_store.safe_path(current_user()["id"], filename)


@api_bp.route("/summaries/<path:filename>", methods=["GET"])
@auth_required_api
def read_summary(filename):
    file_path = _safe_summary_path(filename)
    if file_path is None:
        return jsonify({"error": "Ungültige Datei"}), 400
    if not file_path.exists():
        return jsonify({"error": "Datei nicht gefunden"}), 404

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return jsonify({"error": f"Fehler beim Lesen: {e!s}"}), 500

    return jsonify({"filename": filename, "content": content})


@api_bp.route("/summaries/<path:filename>", methods=["PUT"])
@auth_required_api
def save_summary(filename):
    file_path = _safe_summary_path(filename)
    if file_path is None:
        return jsonify({"success": False, "error": "Ungültige Datei"}), 400

    data = request.get_json() or {}
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(data.get("content", ""), encoding="utf-8")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route("/summaries/<path:filename>", methods=["DELETE"])
@auth_required_api
def delete_summary(filename):
    file_path = _safe_summary_path(filename)
    if file_path is None:
        return jsonify({"success": False, "error": "Ungültige Datei"}), 400

    try:
        if file_path.exists():
            file_path.unlink()
        get_db_service().delete_summary_by_filename(
            filename, user_id=current_user()["id"]
        )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
