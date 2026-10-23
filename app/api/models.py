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

"""LLM-Connections und Modell-Konfigurationen (Admin)."""

from logging import getLogger

from flask import Blueprint, jsonify, request

from app.api._common import _error
from app.auth.access import can_use_model
from app.auth.auth import auth_required_api, current_user, require_role
from app.auth.users import role_at_least
from app.llm import connections as store
from app.llm import registry

bp = Blueprint("models", __name__, url_prefix="/api")
logger = getLogger(__name__)


@bp.route("/connections", methods=["GET"])
@require_role("admin")
def connections_list():
    return jsonify(store.list_connections())


@bp.route("/connections", methods=["POST"])
@require_role("admin")
def connections_create():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    base_url = store.normalize_base_url(data.get("base_url") or "")
    if not name or not base_url:
        return _error("Name und Base-URL sind erforderlich", 400)
    if not base_url.startswith(("http://", "https://")):
        return _error("Base-URL muss mit http:// oder https:// beginnen", 400)

    try:
        registry._reject_link_local(base_url)
    except registry.RemoteModelsError as e:
        return _error(str(e), 400)

    try:
        created = store.create_connection(
            name,
            base_url,
            data.get("api_key") or "",
            bool(data.get("enabled", True)),
            kind=data.get("kind") or "llm",
        )
    except ValueError as e:
        return _error(str(e), 400)
    registry.clear_caches()
    return jsonify(created), 201


@bp.route("/connections/<int:connection_id>", methods=["PATCH"])
@require_role("admin")
def connections_update(connection_id):
    data = request.get_json() or {}
    base_url = data.get("base_url")
    if base_url and not store.normalize_base_url(base_url).startswith(
        ("http://", "https://")
    ):
        return _error("Base-URL muss mit http:// oder https:// beginnen", 400)
    if base_url:
        try:
            registry._reject_link_local(store.normalize_base_url(base_url))
        except registry.RemoteModelsError as e:
            return _error(str(e), 400)

    # Wird die URL geaendert, muss der Key neu eingegeben werden (siehe oben).
    current = store.get_connection(connection_id)
    if (
        current
        and base_url
        and store.normalize_base_url(base_url) != current["base_url"]
        and current["has_key"]
        and not (data.get("api_key") or "").strip()
    ):
        return _error("Bei geänderter Base-URL bitte den API-Key erneut eingeben.", 400)

    updated = store.update_connection(
        connection_id,
        name=data.get("name"),
        base_url=base_url,
        api_key=data.get("api_key"),
        enabled=data.get("enabled"),
        kind=data.get("kind"),
    )
    if not updated:
        return _error("Connection nicht gefunden", 404)
    registry.clear_caches()
    return jsonify(updated)


@bp.route("/connections/<int:connection_id>", methods=["DELETE"])
@require_role("admin")
def connections_delete(connection_id):
    if not store.delete_connection(connection_id):
        return _error("Connection nicht gefunden", 404)
    registry.clear_caches()
    return jsonify({"success": True})


@bp.route("/connections/test", methods=["POST"])
@require_role("admin")
def connections_test():
    """Testet Base-URL + Key, auch fuer noch ungespeicherte Eingaben. Ein leerer
    Key bei vorhandener ``id`` verwendet den gespeicherten."""
    data = request.get_json() or {}
    base_url = store.normalize_base_url(data.get("base_url") or "")
    if not base_url:
        return _error("Base-URL erforderlich", 400)

    api_key = (data.get("api_key") or "").strip()
    connection_id = data.get("id")
    if not api_key and connection_id:
        # Den gespeicherten Key nur an die gespeicherte URL senden, sonst liesse
        # er sich per geaenderter Base-URL an einen fremden Server ausleiten.
        stored = store.get_connection(int(connection_id), with_key=True)
        if stored and store.normalize_base_url(stored["base_url"]) == base_url:
            api_key = stored["api_key"]

    kind = data.get("kind") or "llm"
    if kind not in store.CONNECTION_KINDS:
        return _error("Unbekannter Typ", 400)
    connection = {
        "id": int(connection_id or 0),
        "kind": kind,
        "base_url": base_url,
        "api_key": api_key,
    }
    if kind == "llm":
        result = registry.test_connection(connection)
    else:  # TEI / Infinity: Erreichbarkeit und beide APIs wirklich pruefen
        from app.rag import endpoints

        result = endpoints.test_connection(connection)
    return jsonify(result), 200 if result["success"] else 502


@bp.route("/models", methods=["GET"])
@auth_required_api
def models_list():
    """Alle Modelle der aktiven Connections. ``visible=1`` filtert auf die im
    Chat sichtbaren (Modell-Dropdown), ``refresh=1`` umgeht den kurzen Cache."""
    is_admin = role_at_least(current_user()["role"], "admin")
    result = registry.list_models(force=is_admin and request.args.get("refresh") == "1")
    if request.args.get("visible") == "1" or not is_admin:
        result["models"] = [
            m for m in result["models"] if m["visible"] and m["available"]
        ]
    if not is_admin:
        user = current_user()
        result["models"] = [
            m for m in result["models"] if can_use_model(user, m["ref"])
        ]
        if result["default"] and not can_use_model(user, result["default"]):
            result["default"] = ""
    # Das Modell-Knowledge ist immer die Basis und im Chat nicht abwaehlbar: ``knowledge_base`` und
    # ``knowledge_locked`` bekommen auch Admins, sonst sehen sie es als normale, abwaehlbare Auswahl.
    from app.knowledge.service import list_knowledge

    names = {k["id"]: k["name"] for k in list_knowledge()}
    for m in result["models"]:
        cfg = m.get("config")
        ids = (cfg or {}).get("knowledge_ids") or []
        # Gesperrt ist die Auswahl nur, wenn der Admin zusaetzliches Knowledge nicht erlaubt hat.
        m["knowledge_locked"] = bool(ids) and not (cfg or {}).get(
            "allow_extra_knowledge", True
        )
        m["knowledge_base"] = [{"id": i, "name": names[i]} for i in ids if i in names]
        m["vision"] = bool((cfg or {}).get("vision"))
        if not is_admin:
            # Prompts, Parameter und Knowledge-Zuordnung sind Admin-Sache
            m.pop("config", None)
            m["customized"] = False
    if not is_admin:
        result["errors"] = []
    return jsonify(result)


@bp.route("/models/config", methods=["PUT"])
@require_role("admin")
def models_config_save():
    data = request.get_json() or {}
    ref = data.pop("ref", "")
    if data.get("web_search"):
        from app.agent.tools.web_access import web_search_available

        if not web_search_available():
            return _error(
                "Die Websuche braucht SearXNG (Administration › Tools) und entweder einen aktiven "
                "PII-Filter oder die ausdrückliche Freigabe „Auch ohne PII-Filter erlauben“. "
                "Suchanfragen verlassen das System.",
                400,
            )
    try:
        saved = store.save_model_config(ref, data)
    except ValueError as e:
        return _error(str(e), 400)
    if not saved:
        return _error("Modell oder Connection nicht gefunden", 404)
    return jsonify(saved)


@bp.route("/models/config", methods=["DELETE"])
@require_role("admin")
def models_config_delete():
    ref = request.args.get("ref", "")
    try:
        deleted = store.delete_model_config(ref)
    except ValueError as e:
        return _error(str(e), 400)
    # Ein nie angepasstes Basismodell hat nichts zurueckzusetzen: auch ok.
    return jsonify({"success": True, "deleted": deleted})


@bp.route("/models/clone", methods=["POST"])
@require_role("admin")
def models_clone():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    if not name:
        return _error("Name erforderlich", 400)
    try:
        clone = store.clone_model(data.get("ref", ""), name)
    except ValueError as e:
        return _error(str(e), 400)
    if not clone:
        return _error("Modell oder Connection nicht gefunden", 404)
    return jsonify(clone), 201


@bp.route("/models/default", methods=["PUT"])
@require_role("admin")
def models_set_default():
    ref = (request.get_json() or {}).get("ref", "")
    try:
        store.parse_ref(ref)
    except ValueError as e:
        return _error(str(e), 400)
    store.set_default_model(ref)
    return jsonify({"success": True, "default": ref})
