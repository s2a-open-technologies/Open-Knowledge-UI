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

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.errors import ApiError
from app.api.schemas.common import Loose, Success
from app.api.schemas.models import (
    CloneRequest,
    ConfigDeleted,
    Connection,
    ConnectionCreate,
    ConnectionTest,
    ConnectionUpdate,
    DefaultModel,
    DefaultSet,
    ModelConfig,
    ModelConfigSave,
    ModelList,
)
from app.auth.access import can_use_model
from app.auth.auth import Admin, CurrentUser
from app.auth.users import role_at_least
from app.llm import connections as store
from app.llm import registry

router = APIRouter(prefix="/api", tags=["models"])
logger = getLogger(__name__)


def _error(message: str, status: int) -> ApiError:
    return ApiError(status, message, success=False)


@router.get("/connections", response_model=list[Connection])
def connections_list(_user: Admin):
    return store.list_connections()


@router.post("/connections", response_model=Connection, status_code=201)
def connections_create(_user: Admin, body: ConnectionCreate | None = None):
    body = body or ConnectionCreate()
    name = (body.name or "").strip()
    base_url = store.normalize_base_url(body.base_url or "")
    if not name or not base_url:
        raise _error("Name und Base-URL sind erforderlich", 400)
    if not base_url.startswith(("http://", "https://")):
        raise _error("Base-URL muss mit http:// oder https:// beginnen", 400)

    try:
        registry._reject_link_local(base_url)
    except registry.RemoteModelsError as e:
        raise _error(str(e), 400)

    try:
        created = store.create_connection(
            name, base_url, body.api_key or "", body.enabled, kind=body.kind or "llm"
        )
    except ValueError as e:
        raise _error(str(e), 400)
    registry.clear_caches()
    return created


@router.patch("/connections/{connection_id:int}", response_model=Connection)
def connections_update(connection_id: int, _user: Admin, body: ConnectionUpdate | None = None):
    body = body or ConnectionUpdate()
    base_url = body.base_url
    if base_url and not store.normalize_base_url(base_url).startswith(("http://", "https://")):
        raise _error("Base-URL muss mit http:// oder https:// beginnen", 400)
    if base_url:
        try:
            registry._reject_link_local(store.normalize_base_url(base_url))
        except registry.RemoteModelsError as e:
            raise _error(str(e), 400)

    # Wird die URL geaendert, muss der Key neu eingegeben werden (siehe oben).
    current = store.get_connection(connection_id)
    if (
        current
        and base_url
        and store.normalize_base_url(base_url) != current["base_url"]
        and current["has_key"]
        and not (body.api_key or "").strip()
    ):
        raise _error("Bei geänderter Base-URL bitte den API-Key erneut eingeben.", 400)

    updated = store.update_connection(
        connection_id,
        name=body.name,
        base_url=base_url,
        api_key=body.api_key,
        enabled=body.enabled,
        kind=body.kind,
    )
    if not updated:
        raise _error("Connection nicht gefunden", 404)
    registry.clear_caches()
    return updated


@router.delete("/connections/{connection_id:int}", response_model=Success)
def connections_delete(connection_id: int, _user: Admin):
    if not store.delete_connection(connection_id):
        raise _error("Connection nicht gefunden", 404)
    registry.clear_caches()
    return Success()


@router.post("/connections/test", response_model=Loose)
def connections_test(_user: Admin, body: ConnectionTest | None = None):
    """Testet Base-URL + Key, auch fuer noch ungespeicherte Eingaben. Ein leerer
    Key bei vorhandener ``id`` verwendet den gespeicherten."""
    body = body or ConnectionTest()
    base_url = store.normalize_base_url(body.base_url or "")
    if not base_url:
        raise _error("Base-URL erforderlich", 400)

    api_key = (body.api_key or "").strip()
    connection_id = body.id
    if not api_key and connection_id:
        # Den gespeicherten Key nur an die gespeicherte URL senden, sonst liesse
        # er sich per geaenderter Base-URL an einen fremden Server ausleiten.
        stored = store.get_connection(int(connection_id), with_key=True)
        if stored and store.normalize_base_url(stored["base_url"]) == base_url:
            api_key = stored["api_key"]

    kind = body.kind or "llm"
    if kind not in store.CONNECTION_KINDS:
        raise _error("Unbekannter Typ", 400)
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
    return JSONResponse(result, status_code=200 if result["success"] else 502)


@router.get("/models", response_model=ModelList)
def models_list(user: CurrentUser, visible: str | None = None, refresh: str | None = None):
    """Alle Modelle der aktiven Connections. ``visible=1`` filtert auf die im
    Chat sichtbaren (Modell-Dropdown), ``refresh=1`` umgeht den kurzen Cache."""
    is_admin = role_at_least(user["role"], "admin")
    result = registry.list_models(force=is_admin and refresh == "1")
    if visible == "1" or not is_admin:
        result["models"] = [m for m in result["models"] if m["visible"] and m["available"]]
    if not is_admin:
        result["models"] = [m for m in result["models"] if can_use_model(user, m["ref"])]
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
        m["knowledge_locked"] = bool(ids) and not (cfg or {}).get("allow_extra_knowledge", True)
        m["knowledge_base"] = [{"id": i, "name": names[i]} for i in ids if i in names]
        m["vision"] = bool((cfg or {}).get("vision"))
        if not is_admin:
            # Prompts, Parameter und Knowledge-Zuordnung sind Admin-Sache
            m.pop("config", None)
            m["customized"] = False
    if not is_admin:
        result["errors"] = []
    return result


@router.put("/models/config", response_model=ModelConfig)
def models_config_save(_user: Admin, body: ModelConfigSave | None = None):
    data = body.model_dump(exclude_unset=True) if body else {}
    ref = data.pop("ref", "")
    if data.get("web_search"):
        from app.agent.tools.web_access import web_search_available

        if not web_search_available():
            raise _error(
                "Die Websuche braucht SearXNG (Administration › Tools) und entweder einen aktiven "
                "PII-Filter oder die ausdrückliche Freigabe „Auch ohne PII-Filter erlauben“. "
                "Suchanfragen verlassen das System.",
                400,
            )
    try:
        saved = store.save_model_config(ref, data)
    except ValueError as e:
        raise _error(str(e), 400)
    if not saved:
        raise _error("Modell oder Connection nicht gefunden", 404)
    return saved


@router.delete("/models/config", response_model=ConfigDeleted)
def models_config_delete(_user: Admin, ref: str = ""):
    try:
        deleted = store.delete_model_config(ref)
    except ValueError as e:
        raise _error(str(e), 400)
    # Ein nie angepasstes Basismodell hat nichts zurueckzusetzen: auch ok.
    return {"success": True, "deleted": deleted}


@router.post("/models/clone", response_model=ModelConfig, status_code=201)
def models_clone(_user: Admin, body: CloneRequest | None = None):
    body = body or CloneRequest()
    name = (body.name or "").strip()
    if not name:
        raise _error("Name erforderlich", 400)
    try:
        clone = store.clone_model(body.ref, name)
    except ValueError as e:
        raise _error(str(e), 400)
    if not clone:
        raise _error("Modell oder Connection nicht gefunden", 404)
    return clone


@router.put("/models/default", response_model=DefaultSet)
def models_set_default(_user: Admin, body: DefaultModel | None = None):
    ref = (body or DefaultModel()).ref
    try:
        store.parse_ref(ref)
    except ValueError as e:
        raise _error(str(e), 400)
    store.set_default_model(ref)
    return {"success": True, "default": ref}
