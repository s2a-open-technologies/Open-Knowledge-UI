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

"""Admin-Einstellungen: Embedding, Werkzeuge, Anmeldung, Compaction, Agent-Verhalten, Tool-Worker, Audit-Log."""

from logging import getLogger

from flask import (
    Blueprint,
    jsonify,
    request,
)

from app.api._common import _error
from app.auth.auth import require_role
from app.core.database_service import get_db_service
from app.core.device import torch_device
from app.core.settings_service import SettingsService
from app.knowledge.service import (
    list_knowledge,
)
from app.llm import connections as store
from app.llm import registry
from app.rag import model_settings

bp = Blueprint("admin_api", __name__, url_prefix="/api")
logger = getLogger(__name__)


@bp.route("/admin/audit", methods=["GET"])
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


INGEST_FIELDS = {
    "chunk_size": ("INGEST_CHUNK_SIZE", int, (200, 4000)),
    "chunk_overlap": ("INGEST_CHUNK_OVERLAP", int, (0, 1000)),
    "keep_original": ("INGEST_KEEP_ORIGINAL", bool, None),
    "index_enabled": ("INGEST_INDEX_ENABLED", bool, None),
    "bm25_language": ("BM25_LANGUAGE", str, None),
}


MODEL_SUGGESTIONS = {
    "embedding": [
        "BAAI/bge-m3",
        "intfloat/multilingual-e5-large",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ],
    "reranker": ["BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-base"],
}


def _ingest_values() -> dict:
    from app.rag import sparse as sparse_lang

    db = get_db_service().get_session()
    try:
        s = SettingsService(db)
        return {
            "chunk_size": s.get_int("INGEST_CHUNK_SIZE", 1000),
            "chunk_overlap": s.get_int("INGEST_CHUNK_OVERLAP", 150),
            "keep_original": s.get_bool("INGEST_KEEP_ORIGINAL", True),
            "index_enabled": s.get_bool("INGEST_INDEX_ENABLED", True),
            "bm25_language": sparse_lang.default_language(),
        }
    finally:
        db.close()


def _parse_spec(raw):
    from app.rag import models_runtime

    if not isinstance(raw, dict):
        raise ValueError("Modell fehlt")
    spec = models_runtime.ModelSpec.from_dict(raw)  # prueft Quelle, Modell, Connection
    if spec.source == "endpoint" and not store.get_connection(spec.connection_id):
        raise ValueError("Connection nicht gefunden")
    return spec


def _setting_value(key: str) -> str:

    db = get_db_service().get_session()
    try:
        return SettingsService(db).get(key, "") or ""
    finally:
        db.close()


def _tasks_overview() -> dict:
    from app.ingest.tasks import TASKS, load_task

    out = {}
    for name in TASKS:
        cfg = load_task(name)
        out[name] = {
            "enabled": cfg["enabled"],
            "model_ref": cfg["model_ref"],
            "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
            "prompt": cfg["prompt"],
            "default_prompt": cfg["default_prompt"],
            **({"excerpt_chars": cfg["excerpt_chars"]} if name == "context" else {}),
            **(
                {"max_row_deviation": cfg["max_row_deviation"]}
                if name == "tables"
                else {}
            ),
        }
    return out


def _validate_tasks(data: dict) -> tuple[dict[str, str], str | None]:
    """Prueft die Einstellungen der Einlese-Aufgaben; liefert (Werte, Fehlermeldung)."""
    from app.ingest.tasks import TASKS, validate_prompt

    values: dict[str, str] = {}
    known = {m["ref"] for m in registry.list_models()["models"]}
    for name, cfg in (data or {}).items():
        if name not in TASKS:
            return {}, f"Unbekannte Aufgabe: {name}"
        prefix = name.upper()
        if "enabled" in cfg:
            from app.ingest.tasks import ENABLED_KEY

            values[ENABLED_KEY[name]] = "true" if cfg["enabled"] else "false"
        if "model_ref" in cfg:
            ref = (cfg["model_ref"] or "").strip()
            if ref:
                try:
                    store.parse_ref(ref)
                except ValueError:
                    return {}, f"{name}: ungültiges Modell"
                if ref not in known:
                    return {}, f"{name}: das Modell existiert nicht (mehr)"
            values[f"{prefix}_MODEL_REF"] = ref
        if "temperature" in cfg:
            raw = cfg["temperature"]
            if raw in (None, ""):
                values[f"{prefix}_TEMPERATURE"] = ""
            else:
                try:
                    t = float(raw)
                except (TypeError, ValueError):
                    return {}, f"{name}: Temperatur muss eine Zahl sein"
                if not 0 <= t <= 2:
                    return {}, f"{name}: Temperatur muss zwischen 0 und 2 liegen"
                values[f"{prefix}_TEMPERATURE"] = str(t)
        if "max_tokens" in cfg:
            raw = cfg["max_tokens"]
            if raw in (None, ""):
                values[f"{prefix}_MAX_TOKENS"] = ""
            else:
                try:
                    n = int(raw)
                except (TypeError, ValueError):
                    return {}, f"{name}: Max-Tokens muss eine ganze Zahl sein"
                if n < 1:
                    return {}, f"{name}: Max-Tokens muss mindestens 1 sein"
                values[f"{prefix}_MAX_TOKENS"] = str(n)
        if "prompt" in cfg:
            error = validate_prompt(name, cfg["prompt"])
            if error:
                return {}, error
            values[f"{prefix}_PROMPT"] = cfg["prompt"].strip()
        if name == "tables" and "max_row_deviation" in cfg:
            try:
                dev = int(cfg["max_row_deviation"])
            except (TypeError, ValueError):
                return {}, "Zeilenabweichung muss eine ganze Zahl sein"
            if not 0 <= dev <= 100:
                return {}, "Zeilenabweichung: erlaubt sind 0 bis 100 Prozent"
            values["TABLES_MAX_ROW_DEVIATION"] = str(dev)
        if name == "context" and "excerpt_chars" in cfg:
            try:
                chars = int(cfg["excerpt_chars"])
            except (TypeError, ValueError):
                return {}, "Auszugslänge muss eine ganze Zahl sein"
            if not 200 <= chars <= 20000:
                return {}, "Auszugslänge: erlaubt sind 200 bis 20000 Zeichen"
            values["CONTEXT_EXCERPT_CHARS"] = str(chars)
    return values, None


def _embedding_overview() -> dict:
    from app.rag import models_runtime
    from app.rag import sparse as sparse_lang

    emb, rr = model_settings.embedding_spec(), model_settings.reranker_spec()
    stale = [k for k in list_knowledge() if model_settings.is_stale(k)]
    return {
        "embedding": {
            **emb.to_dict(),
            "status": models_runtime.status("embedding", emb),
        },
        "reranker": {**rr.to_dict(), "status": models_runtime.status("reranker", rr)},
        "ingest": _ingest_values(),
        "bm25_languages": sparse_lang.supported_languages(),
        "device": torch_device(),
        "tasks": _tasks_overview(),
        "stale": [
            {"id": k["id"], "name": k["name"], "error": k.get("last_reindex_error")}
            for k in stale
        ],
        "suggestions": MODEL_SUGGESTIONS,
    }


@bp.route("/admin/embedding", methods=["GET"])
@require_role("admin")
def embedding_overview():
    return jsonify(_embedding_overview())


@bp.route("/admin/embedding", methods=["PUT"])
@require_role("admin")
def embedding_save():
    """Speichert Modelle und Einlese-Einstellungen; neue Modelle werden im Hintergrund geladen.

    Ein anderes Embedding-Modell macht bestehende Knowledge "veraltet" (Neu indexieren); bis
    dahin werden sie weiter mit ihrem bisherigen Modell abgefragt.
    """
    from app.rag import model_settings, models_runtime

    data = request.get_json() or {}
    try:
        emb = _parse_spec(data["embedding"]) if "embedding" in data else None
        rr = _parse_spec(data["reranker"]) if "reranker" in data else None
    except (ValueError, TypeError) as e:
        return _error(str(e), 400)

    values: dict[str, str] = {}
    task_values, error = _validate_tasks(data.get("tasks") or {})
    if error:
        return _error(error, 400)
    values.update(task_values)
    ingest = data.get("ingest") or {}
    for field, raw in ingest.items():
        if field not in INGEST_FIELDS:
            return _error(f"Unbekannte Einstellung: {field}", 400)
        key, kind, bounds = INGEST_FIELDS[field]
        if kind is bool:
            values[key] = "true" if bool(raw) else "false"
        elif kind is str:
            from app.rag import sparse as sparse_lang

            if str(raw) not in sparse_lang.supported_languages():
                return _error(f"{field}: unbekannte Sprache", 400)
            values[key] = str(raw)
        else:
            try:
                number = int(raw)
            except (TypeError, ValueError):
                return _error(f"{field}: ganze Zahl erwartet", 400)
            if not bounds[0] <= number <= bounds[1]:
                return _error(f"{field}: erlaubt sind {bounds[0]} bis {bounds[1]}", 400)
            values[key] = str(number)
    size = int(values.get("INGEST_CHUNK_SIZE") or _ingest_values()["chunk_size"])
    overlap = int(
        values.get("INGEST_CHUNK_OVERLAP") or _ingest_values()["chunk_overlap"]
    )
    if overlap >= size:
        return _error("Die Überlappung muss kleiner als die Chunk-Größe sein", 400)

    db = get_db_service().get_session()
    try:
        SettingsService(db).set_batch(values)
    finally:
        db.close()
    model_settings.save_specs(emb, rr)
    if emb:
        models_runtime.load_in_background("embedding", emb)
    if rr:
        models_runtime.load_in_background("reranker", rr)
    return jsonify(_embedding_overview())


@bp.route("/admin/embedding/test", methods=["POST"])
@require_role("admin")
def embedding_test():
    """Laedt das Modell (beim ersten Mal mit Download) und meldet Dimension bzw. Fehler."""
    import time

    from app.rag import models_runtime

    data = request.get_json() or {}
    kind = data.get("kind")
    if kind not in ("embedding", "reranker"):
        return _error("kind muss embedding oder reranker sein", 400)
    try:
        spec = _parse_spec(data.get("spec"))
    except (ValueError, TypeError) as e:
        return _error(str(e), 400)
    started = time.time()
    try:
        if kind == "embedding":
            model = models_runtime.get_embedder(spec)
            result = {"dimension": model.dimension}
        else:
            scores = models_runtime.get_reranker(spec).score(
                "Wie hoch ist die Miete?",
                ["Die Miete beträgt 800 Euro.", "Das Wetter ist schön."],
            )
            result = {"ranking_ok": scores[0] > scores[1]}
    except models_runtime.ModelLoadError as e:
        return jsonify({"success": False, "error": str(e)}), 502
    return jsonify(
        {"success": True, "seconds": round(time.time() - started, 1), **result}
    )


@bp.route("/admin/embedding/remote-models", methods=["GET"])
@require_role("admin")
def embedding_remote_models():
    """Modell-IDs einer Connection (fuer die Auswahl bei Quelle "Endpoint")."""
    from app.rag import endpoints

    conn = store.get_connection(
        request.args.get("connection_id", type=int) or 0, with_key=True
    )
    if not conn:
        return _error("Connection nicht gefunden", 404)
    try:
        if conn.get("kind", "llm") == "llm":
            return jsonify(registry.fetch_remote_models(conn))
        wanted = request.args.get(
            "capability"
        )  # embedding | reranker: nur passende Modelle
        return jsonify(
            [
                m["id"]
                for m in endpoints.describe(conn)
                if not wanted or wanted in m["capabilities"]
            ]
        )
    except (registry.RemoteModelsError, endpoints.EndpointError) as e:
        return _error(str(e), 502)


class _FormSettings:
    """Stellt die Werte aus dem Formular wie Settings bereit (Test vor dem Speichern)."""

    def __init__(self, values: dict):
        self.values = values

    def get(self, key, default=None):
        value = self.values.get(key)
        return default if value is None else value

    def get_float(self, key, default=0.0):
        try:
            return float(self.values.get(key))
        except (TypeError, ValueError):
            return default


def _tools_overview() -> dict:
    from app.agent.tools.web_access import web_search_available, web_search_masked

    db = get_db_service().get_session()
    try:
        s = SettingsService(db)
        return {
            "stt": {
                "base_url": s.get("STT_BASEURL", "") or "",
                "model": s.get("STT_MODEL", "") or "",
                "mode": (s.get("STT_MODE", "auto") or "auto"),
                "has_key": bool(s.get("STT_API_KEY")),
            },
            "search": {
                "url": s.get("SEARCH_SEARXNG_URL", "") or "",
                "categories": s.get("SEARCH_SEARXNG_CATEGORIES", "general")
                or "general",
                "allow_unmasked": s.get_bool("SEARCH_ALLOW_UNMASKED", False),
            },
            "pii": {
                "url": s.get("PII_FILTER_URL", "") or "",
                "enabled": s.get_bool("PII_FILTER_ENABLED", True),
                "has_key": bool(s.get("PII_FILTER_API_KEY")),
                "timeout_s": s.get_float("PII_FILTER_TIMEOUT_S", 10.0),
                "max_chars": s.get_int("PII_FILTER_MAX_CHARS", 12000),
            },
            # Die Websuche ist je Modell zuschaltbar, sobald SearXNG und der PII-Filter laufen
            "web_search_available": web_search_available(),
            "web_search_masked": web_search_masked(),
        }
    finally:
        db.close()


def _clean_url(raw, label: str) -> str:
    url = store.normalize_base_url((raw or "").strip()) if raw else ""
    if url and not url.startswith(("http://", "https://")):
        raise ValueError(f"{label}: Adresse muss mit http:// oder https:// beginnen")
    if url:
        try:
            registry._reject_link_local(url)
        except registry.RemoteModelsError as e:
            raise ValueError(f"{label}: {e}") from None
    return url


@bp.route("/admin/tools", methods=["GET"])
@require_role("admin")
def tools_overview():
    return jsonify(_tools_overview())


@bp.route("/admin/tools", methods=["PUT"])
@require_role("admin")
def tools_save():
    """Speichert die Tool-Einstellungen. API-Keys werden nie ausgeliefert: ein leeres Feld
    behaelt den gespeicherten Key, ``clear_key`` loescht ihn."""

    data = request.get_json() or {}
    values: dict[str, str] = {}
    try:
        if "stt" in data:
            stt = data["stt"] or {}
            values["STT_BASEURL"] = _clean_url(stt.get("base_url"), "Spracheingabe")
            values["STT_MODEL"] = (stt.get("model") or "").strip()
            if "mode" in stt:
                from app.chat import stt as stt_config

                mode = str(stt.get("mode") or "auto").strip().lower()
                if mode not in stt_config.MODES:
                    raise ValueError(
                        "Spracheingabe: Betriebsart muss auto, stream oder upload sein"
                    )
                values["STT_MODE"] = mode
            if stt.get("clear_key"):
                values["STT_API_KEY"] = ""
            elif (stt.get("api_key") or "").strip():
                values["STT_API_KEY"] = stt["api_key"].strip()
        if "search" in data:
            search = data["search"] or {}
            values["SEARCH_SEARXNG_URL"] = _clean_url(search.get("url"), "Websuche")
            values["SEARCH_SEARXNG_CATEGORIES"] = (
                search.get("categories") or "general"
            ).strip()
            if "allow_unmasked" in search:
                values["SEARCH_ALLOW_UNMASKED"] = (
                    "true" if search["allow_unmasked"] else "false"
                )
        if "pii" in data:
            pii = data["pii"] or {}
            url = (pii.get("url") or "").strip()
            if url and not url.startswith(("http://", "https://")):
                raise ValueError(
                    "PII-Filter: Adresse muss mit http:// oder https:// beginnen"
                )
            if url:
                try:
                    registry._reject_link_local(url)
                except registry.RemoteModelsError as e:
                    raise ValueError(f"PII-Filter: {e}") from None
            values["PII_FILTER_URL"] = url
            values["PII_FILTER_ENABLED"] = (
                "true" if pii.get("enabled", True) else "false"
            )
            if "timeout_s" in pii:
                try:
                    timeout_s = float(pii["timeout_s"])
                except (TypeError, ValueError):
                    raise ValueError(
                        "PII-Filter: Zeitlimit muss eine Zahl sein"
                    ) from None
                if not 2 <= timeout_s <= 120:
                    raise ValueError(
                        "PII-Filter: Zeitlimit zwischen 2 und 120 Sekunden"
                    )
                values["PII_FILTER_TIMEOUT_S"] = f"{timeout_s:g}"
            if "max_chars" in pii:
                try:
                    chars = int(pii["max_chars"])
                except (TypeError, ValueError):
                    raise ValueError(
                        "PII-Filter: Höchstlänge muss eine ganze Zahl sein"
                    ) from None
                if not 2000 <= chars <= 100000:
                    raise ValueError(
                        "PII-Filter: Höchstlänge zwischen 2000 und 100000 Zeichen"
                    )
                values["PII_FILTER_MAX_CHARS"] = str(chars)
            if pii.get("clear_key"):
                values["PII_FILTER_API_KEY"] = ""
            elif (pii.get("api_key") or "").strip():
                values["PII_FILTER_API_KEY"] = pii["api_key"].strip()
    except ValueError as e:
        return _error(str(e), 400)

    if values:
        db = get_db_service().get_session()
        try:
            SettingsService(db).set_batch(values)
        finally:
            db.close()
        if any(k.startswith("STT_") for k in values):
            from app.chat import stt as stt_config

            stt_config.clear_probe_cache()
    # Ohne PII-Filter darf keine Websuche an Modellen haengen (sie steht dann nicht mehr bereit)
    return jsonify(_tools_overview())


@bp.route("/admin/tools/test", methods=["POST"])
@require_role("admin")
def tools_test():
    """Testet die eingegebenen Werte (auch ungespeichert). Ein leeres Key-Feld nutzt den
    gespeicherten Key nur fuer die gespeicherte Adresse (sonst liesse er sich ausleiten)."""
    from app.api.tool_checks import check_pii_filter, check_search, check_stt

    data = request.get_json() or {}
    kind = data.get("kind")
    values_in = data.get("values") or {}
    if kind not in ("stt", "search", "pii"):
        return _error("kind muss stt, search oder pii sein", 400)

    db = get_db_service().get_session()
    try:
        saved = SettingsService(db)
        if kind == "stt":
            url = (values_in.get("base_url") or "").strip()
            key = (values_in.get("api_key") or "").strip()
            if not key and store.normalize_base_url(url) == store.normalize_base_url(
                saved.get("STT_BASEURL", "") or ""
            ):
                key = saved.get("STT_API_KEY") or ""
            form = _FormSettings(
                {
                    "STT_BASEURL": url,
                    "STT_MODEL": values_in.get("model", ""),
                    "STT_API_KEY": key,
                }
            )
            test = check_stt
        elif kind == "search":
            form = _FormSettings(
                {"SEARCH_SEARXNG_URL": (values_in.get("url") or "").strip()}
            )
            test = check_search
        else:
            url = (values_in.get("url") or "").strip()
            key = (values_in.get("api_key") or "").strip()
            if not key and url == (saved.get("PII_FILTER_URL", "") or ""):
                key = saved.get("PII_FILTER_API_KEY") or ""
            form = _FormSettings(
                {
                    "PII_FILTER_URL": url,
                    "PII_FILTER_API_KEY": key,
                    "PII_FILTER_TIMEOUT_S": values_in.get("timeout_s"),
                }
            )
            test = check_pii_filter
    finally:
        db.close()

    for field, label in (("base_url", "stt"), ("url", "search"), ("url", "pii")):
        raw = values_in.get(field) if kind == label else None
        if raw:
            try:
                registry._reject_link_local(store.normalize_base_url(raw.strip()))
            except registry.RemoteModelsError as e:
                return jsonify({"success": False, "error": str(e)}), 400
    try:
        result = test(form)
    except Exception as e:
        return jsonify({"success": False, "error": f"Unerwarteter Fehler: {e}"}), 500
    return jsonify(result), 200 if result.get("success") else 502


@bp.route("/admin/auth", methods=["GET"])
@require_role("admin")
def auth_overview():
    from app.auth import external as auth_external

    cfg = auth_external.load_config()
    pub = auth_external.public_config(cfg)
    base = cfg["base_url"] or request.host_url.rstrip("/")
    pub["redirect_uri"] = f"{base}/api/auth/oidc/callback"
    return jsonify(pub)


@bp.route("/admin/auth", methods=["PUT"])
@require_role("admin")
def auth_save():
    """Speichert die Einstellungen. Leere Geheimnis-Felder behalten den gespeicherten Wert."""
    from app.auth import external as auth_external

    try:
        auth_external.save_config(request.get_json() or {})
    except auth_external.AuthConfigError as e:
        return _error(str(e), 400)
    return auth_overview()


@bp.route("/admin/auth/test", methods=["POST"])
@require_role("admin")
def auth_test():
    """Testet LDAP bzw. OIDC mit den eingegebenen Werten (ohne zu speichern, ohne Sitzung).

    Ein leeres Geheimnis-Feld nutzt das gespeicherte, aber nur bei unveraenderter Adresse."""
    from app.auth import external as auth_external
    from app.auth import ldap as auth_ldap
    from app.auth import oidc as auth_oidc

    data = request.get_json() or {}
    kind = data.get("kind")
    if kind not in ("ldap", "oidc"):
        return _error("kind muss ldap oder oidc sein", 400)
    saved = auth_external.load_config()
    current = saved[kind]
    incoming = dict(data.get("values") or {})
    target_key = "url" if kind == "ldap" else "issuer"
    same_target = (incoming.get(target_key) or "").strip() == current.get(target_key)
    merged = {**current}
    try:
        merged = auth_external._clean_block(
            kind,
            incoming,
            current
            if same_target
            else {**current, **{s: "" for s in auth_external.SECRET_FIELDS[kind]}},
        )
    except auth_external.AuthConfigError as e:
        return _error(str(e), 400)
    try:
        if kind == "ldap":
            result = auth_ldap.probe_ldap(
                merged,
                str(data.get("test_user") or ""),
                str(data.get("test_password") or ""),
            )
        else:
            result = auth_oidc.probe_oidc(
                merged, saved["base_url"] or request.host_url.rstrip("/")
            )
    except (auth_ldap.LdapError, auth_oidc.OidcError) as e:
        return jsonify({"success": False, "error": str(e)}), 502
    return jsonify({"success": True, **result})


@bp.route("/admin/compaction", methods=["GET"])
@require_role("admin")
def compaction_overview():
    from app.chat import compaction

    return jsonify(compaction.load_settings())


@bp.route("/admin/compaction", methods=["PUT"])
@require_role("admin")
def compaction_save():
    """Speichert die Compaction-Einstellungen (nur genannte Felder)."""
    from app.chat import compaction
    from app.llm import registry

    try:
        compaction.save_settings(request.get_json() or {})
    except compaction.CompactionConfigError as e:
        return _error(str(e), 400)
    registry.clear_caches()  # laufende Agenten tragen das alte Budget
    return jsonify(compaction.load_settings())


@bp.route("/admin/agent-behavior", methods=["GET"])
@require_role("admin")
def agent_behavior_overview():
    from app.agent import behavior

    return jsonify(behavior.load())


@bp.route("/admin/agent-behavior", methods=["PUT"])
@require_role("admin")
def agent_behavior_save():
    """Speichert die Standard-Anweisungen; ein leerer Text heisst: nur Basis-Prompt und Modell-Zusatz."""
    from app.agent import behavior
    from app.llm import registry

    try:
        behavior.save(request.get_json() or {})
    except behavior.BehaviorConfigError as e:
        return _error(str(e), 400)
    registry.clear_caches()  # laufende Agenten tragen den alten Prompt
    return jsonify(behavior.load())


def _tool_worker_view():
    from app.agent.toolworker import config as worker_config
    from app.agent.toolworker.client import get_client

    sql = get_client("sql").ping()
    return {
        **worker_config.view(),
        **get_client().ping(),
        "sql_reachable": sql["reachable"],
        "sql_version": sql["version"],
    }


@bp.route("/admin/tool-worker", methods=["GET"])
@require_role("admin")
def tool_worker_overview():
    return jsonify(_tool_worker_view())


@bp.route("/admin/tool-worker", methods=["PUT"])
@require_role("admin")
def tool_worker_save():
    """Speichert Adresse und Limits des Tool-Workers (nur genannte Felder)."""
    from app.agent.toolworker import config as worker_config

    try:
        worker_config.save(request.get_json() or {})
    except worker_config.ConfigError as e:
        return _error(str(e), 400)
    return jsonify(_tool_worker_view())
