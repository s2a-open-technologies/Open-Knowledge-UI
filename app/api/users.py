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

"""JSON-API: Benutzer, Gruppen, Freigaben und eigenes Passwort."""

from logging import getLogger

from fastapi import APIRouter

from app.api.errors import ApiError
from app.api.schemas.auth import ChangePasswordRequest
from app.api.schemas.common import Success
from app.api.schemas.users import (
    Access,
    AccessUpdate,
    AdminUser,
    Group,
    GroupCreate,
    GroupMembers,
    GroupUpdate,
    ModelAccessUpdate,
    Principals,
    UserCreate,
    UserUpdate,
)
from app.auth import access, user_admin
from app.auth.auth import Admin, CurrentUser, Editor
from app.auth.users import set_password, verify_password
from app.chat.attachments import cleanup_session_attachments
from app.core import runtime
from app.llm import connections as models_store

router = APIRouter(prefix="/api", tags=["users"])

logger = getLogger(__name__)


# =============================================================================
# Eigenes Passwort
# =============================================================================


@router.post("/auth/change-password", response_model=Success)
def change_own_password(user: CurrentUser, body: ChangePasswordRequest | None = None):
    body = body or ChangePasswordRequest()
    if len(body.new_password) < user_admin.MIN_PASSWORD_LENGTH:
        raise ApiError(
            400,
            f"Passwort muss mindestens {user_admin.MIN_PASSWORD_LENGTH} Zeichen haben",
            success=False,
        )
    if not verify_password(user["id"], body.old_password):
        raise ApiError(403, "Altes Passwort ist falsch", success=False)
    set_password(user["id"], body.new_password)
    return Success()


# =============================================================================
# Benutzer (Admin)
# =============================================================================


@router.get("/users", response_model=list[AdminUser])
def users_list(_user: Admin):
    return user_admin.list_users()


@router.post("/users", response_model=AdminUser, status_code=201)
def users_create(_user: Admin, body: UserCreate | None = None):
    body = body or UserCreate()
    return user_admin.create_user(body.username, body.display_name, body.password, body.role)


@router.patch("/users/{user_id}", response_model=AdminUser)
def users_update(user_id: int, user: Admin, body: UserUpdate | None = None):
    data = body.model_dump(exclude_unset=True) if body else {}
    return user_admin.update_user(user_id, user["id"], data)


@router.delete("/users/{user_id}", response_model=Success)
def users_delete(user_id: int, user: Admin):
    session_ids = user_admin.delete_user(user_id, user["id"])
    for sid in session_ids:
        cleanup_session_attachments(sid, runtime.state.client)
    return Success()


# =============================================================================
# Gruppen (Admin)
# =============================================================================


@router.get("/groups", response_model=list[Group])
def groups_list(_user: Admin):
    return user_admin.list_groups()


@router.post("/groups", response_model=Group, status_code=201)
def groups_create(_user: Admin, body: GroupCreate | None = None):
    body = body or GroupCreate()
    return user_admin.create_group(body.name, body.description)


@router.patch("/groups/{group_id}", response_model=Group)
def groups_update(group_id: int, _user: Admin, body: GroupUpdate | None = None):
    return user_admin.update_group(group_id, body.model_dump(exclude_unset=True) if body else {})


@router.delete("/groups/{group_id}", response_model=Success)
def groups_delete(group_id: int, _user: Admin):
    user_admin.delete_group(group_id)
    return Success()


@router.put("/groups/{group_id}/members", response_model=Group)
def groups_set_members(group_id: int, _user: Admin, body: GroupMembers | None = None):
    return user_admin.set_group_members(group_id, (body or GroupMembers()).user_ids)


# =============================================================================
# Freigaben
# =============================================================================


@router.get("/principals", response_model=Principals)
def principals(_user: Editor):
    """Namen aller aktiven Nutzer und Gruppen fuer den Freigabe-Dialog."""
    return {
        "users": [
            {"id": u["id"], "name": u["display_name"], "username": u["username"]}
            for u in user_admin.list_users()
            if u["is_active"]
        ],
        "groups": [{"id": g["id"], "name": g["name"]} for g in user_admin.list_groups()],
    }


def _grants(body: AccessUpdate) -> list[dict]:
    return [g.model_dump() for g in body.grants]


@router.get("/knowledge/{knowledge_id}/access", response_model=Access)
def knowledge_access_get(knowledge_id: int, user: Editor):
    if not access.can_access_knowledge(user, knowledge_id, "write"):
        raise ApiError(404, "Knowledge nicht gefunden", success=False)
    return access.get_access("knowledge", knowledge_id)


@router.put("/knowledge/{knowledge_id}/access", response_model=Access)
def knowledge_access_set(knowledge_id: int, user: Editor, body: AccessUpdate | None = None):
    if not access.can_access_knowledge(user, knowledge_id, "write"):
        raise ApiError(404, "Knowledge nicht gefunden", success=False)
    body = body or AccessUpdate()
    try:
        access.set_access("knowledge", knowledge_id, body.is_public, _grants(body))
    except ValueError as e:
        raise ApiError(400, str(e), success=False)
    # Wer ein Knowledge verwaltet, soll sich beim Speichern nicht selbst
    # aussperren (z. B. Zeile in der Liste entfernt): Schreibrecht bleibt erhalten.
    if not access.can_access_knowledge(user, knowledge_id, "write"):
        access.add_grant("knowledge", knowledge_id, "user", user["id"], "write")
    return access.get_access("knowledge", knowledge_id)


@router.get("/models/access", response_model=Access)
def model_access_get(_user: Admin, ref: str = ""):
    try:
        rid = access.model_resource_id(ref)
    except ValueError as e:
        raise ApiError(400, str(e), success=False)
    if rid is None:
        # Nie angepasstes Basismodell: oeffentlich, ohne Grants.
        return {"is_public": True, "grants": []}
    return access.get_access("model", rid)


@router.put("/models/access", response_model=Access)
def model_access_set(_user: Admin, body: ModelAccessUpdate | None = None):
    body = body or ModelAccessUpdate()
    try:
        rid = access.model_resource_id(body.ref)
        if rid is None:
            # Freigabe braucht eine Konfigurationszeile; sie entsteht hier.
            saved = models_store.save_model_config(body.ref, {"is_public": body.is_public})
            if not saved:
                raise ApiError(404, "Modell oder Connection nicht gefunden", success=False)
            rid = saved["id"]
        access.set_access("model", rid, body.is_public, _grants(body))
    except ValueError as e:
        raise ApiError(400, str(e), success=False)
    return access.get_access("model", rid)
