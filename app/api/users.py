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

from flask import Blueprint, current_app, jsonify, request

from app.auth import access, user_admin
from app.auth.auth import auth_required_api, current_user, require_role
from app.auth.user_admin import AdminError
from app.auth.users import set_password, verify_password
from app.chat.attachments import cleanup_session_attachments
from app.llm import connections as models_store

users_bp = Blueprint("users_api", __name__, url_prefix="/api")

logger = getLogger(__name__)


def _error(message: str, status: int):
    return jsonify({"success": False, "error": message}), status


@users_bp.errorhandler(AdminError)
def _admin_error(e: AdminError):
    return _error(e.message, e.status)


# =============================================================================
# Eigenes Passwort
# =============================================================================


@users_bp.route("/auth/change-password", methods=["POST"])
@auth_required_api
def change_own_password():
    data = request.get_json() or {}
    user = current_user()
    new_password = data.get("new_password", "")
    if len(new_password) < user_admin.MIN_PASSWORD_LENGTH:
        return _error(
            f"Passwort muss mindestens {user_admin.MIN_PASSWORD_LENGTH} Zeichen haben",
            400,
        )
    if not verify_password(user["id"], data.get("old_password", "")):
        return _error("Altes Passwort ist falsch", 403)
    set_password(user["id"], new_password)
    return jsonify({"success": True})


# =============================================================================
# Benutzer (Admin)
# =============================================================================


@users_bp.route("/users", methods=["GET"])
@require_role("admin")
def users_list():
    return jsonify(user_admin.list_users())


@users_bp.route("/users", methods=["POST"])
@require_role("admin")
def users_create():
    data = request.get_json() or {}
    created = user_admin.create_user(
        data.get("username", ""),
        data.get("display_name", ""),
        data.get("password", ""),
        data.get("role", "user"),
    )
    return jsonify(created), 201


@users_bp.route("/users/<int:user_id>", methods=["PATCH"])
@require_role("admin")
def users_update(user_id):
    return jsonify(
        user_admin.update_user(user_id, current_user()["id"], request.get_json() or {})
    )


@users_bp.route("/users/<int:user_id>", methods=["DELETE"])
@require_role("admin")
def users_delete(user_id):
    session_ids = user_admin.delete_user(user_id, current_user()["id"])
    client = current_app.extensions.get("client")
    for sid in session_ids:
        cleanup_session_attachments(sid, client)
    return jsonify({"success": True})


# =============================================================================
# Gruppen (Admin)
# =============================================================================


@users_bp.route("/groups", methods=["GET"])
@require_role("admin")
def groups_list():
    return jsonify(user_admin.list_groups())


@users_bp.route("/groups", methods=["POST"])
@require_role("admin")
def groups_create():
    data = request.get_json() or {}
    created = user_admin.create_group(data.get("name", ""), data.get("description", ""))
    return jsonify(created), 201


@users_bp.route("/groups/<int:group_id>", methods=["PATCH"])
@require_role("admin")
def groups_update(group_id):
    return jsonify(user_admin.update_group(group_id, request.get_json() or {}))


@users_bp.route("/groups/<int:group_id>", methods=["DELETE"])
@require_role("admin")
def groups_delete(group_id):
    user_admin.delete_group(group_id)
    return jsonify({"success": True})


@users_bp.route("/groups/<int:group_id>/members", methods=["PUT"])
@require_role("admin")
def groups_set_members(group_id):
    data = request.get_json() or {}
    return jsonify(user_admin.set_group_members(group_id, data.get("user_ids", [])))


# =============================================================================
# Freigaben
# =============================================================================


@users_bp.route("/principals", methods=["GET"])
@require_role("editor")
def principals():
    """Namen aller aktiven Nutzer und Gruppen fuer den Freigabe-Dialog."""
    return jsonify(
        {
            "users": [
                {"id": u["id"], "name": u["display_name"], "username": u["username"]}
                for u in user_admin.list_users()
                if u["is_active"]
            ],
            "groups": [
                {"id": g["id"], "name": g["name"]} for g in user_admin.list_groups()
            ],
        }
    )


def _access_payload() -> tuple[bool, list[dict]]:
    data = request.get_json() or {}
    return bool(data.get("is_public", False)), data.get("grants") or []


@users_bp.route("/knowledge/<int:knowledge_id>/access", methods=["GET"])
@require_role("editor")
def knowledge_access_get(knowledge_id):
    if not access.can_access_knowledge(current_user(), knowledge_id, "write"):
        return _error("Knowledge nicht gefunden", 404)
    return jsonify(access.get_access("knowledge", knowledge_id))


@users_bp.route("/knowledge/<int:knowledge_id>/access", methods=["PUT"])
@require_role("editor")
def knowledge_access_set(knowledge_id):
    if not access.can_access_knowledge(current_user(), knowledge_id, "write"):
        return _error("Knowledge nicht gefunden", 404)
    is_public, grants = _access_payload()
    user = current_user()
    try:
        access.set_access("knowledge", knowledge_id, is_public, grants)
    except ValueError as e:
        return _error(str(e), 400)
    # Wer ein Knowledge verwaltet, soll sich beim Speichern nicht selbst
    # aussperren (z. B. Zeile in der Liste entfernt): Schreibrecht bleibt erhalten.
    if not access.can_access_knowledge(user, knowledge_id, "write"):
        access.add_grant("knowledge", knowledge_id, "user", user["id"], "write")
    return jsonify(access.get_access("knowledge", knowledge_id))


@users_bp.route("/models/access", methods=["GET"])
@require_role("admin")
def model_access_get():
    ref = request.args.get("ref", "")
    try:
        rid = access.model_resource_id(ref)
    except ValueError as e:
        return _error(str(e), 400)
    if rid is None:
        # Nie angepasstes Basismodell: oeffentlich, ohne Grants.
        return jsonify({"is_public": True, "grants": []})
    return jsonify(access.get_access("model", rid))


@users_bp.route("/models/access", methods=["PUT"])
@require_role("admin")
def model_access_set():
    ref = (request.get_json() or {}).get("ref", "")
    is_public, grants = _access_payload()
    try:
        rid = access.model_resource_id(ref)
        if rid is None:
            # Freigabe braucht eine Konfigurationszeile; sie entsteht hier.
            saved = models_store.save_model_config(ref, {"is_public": is_public})
            if not saved:
                return _error("Modell oder Connection nicht gefunden", 404)
            rid = saved["id"]
        access.set_access("model", rid, is_public, grants)
    except ValueError as e:
        return _error(str(e), 400)
    return jsonify(access.get_access("model", rid))
