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

"""JSON-API: Retrieval-Profile, Zuordnung am Knowledge und Pipeline-Test."""

from logging import getLogger

from flask import Blueprint, current_app, jsonify, request

from app.auth.access import accessible_knowledge_ids, can_access_knowledge
from app.auth.auth import current_user, require_role
from app.auth.users import role_at_least
from app.llm import registry
from app.rag import model_settings, models_runtime, pipeline, profiles
from app.rag import sparse as sparse_lang
from app.rag.helper import make_helper
from app.rag.knowledge import list_knowledge
from app.rag.profiles import ProfileError

rag_bp = Blueprint("rag_api", __name__, url_prefix="/api")

logger = getLogger(__name__)


def _error(message: str, status: int):
    return jsonify({"success": False, "error": message}), status


@rag_bp.errorhandler(ProfileError)
def _profile_error(e: ProfileError):
    return _error(e.message, e.status)


# =============================================================================
# Profile
# =============================================================================


@rag_bp.route("/retrieval/profiles", methods=["GET"])
@require_role("user")
def profiles_list():
    return jsonify(profiles.list_profiles())


@rag_bp.route("/retrieval/profiles", methods=["POST"])
@require_role("admin")
def profiles_create():
    data = request.get_json() or {}
    created = profiles.create_profile(
        data.get("name", ""), data.get("description", ""), data.get("config")
    )
    return jsonify(created), 201


@rag_bp.route("/retrieval/profiles/<int:profile_id>", methods=["PATCH"])
@require_role("admin")
def profiles_update(profile_id):
    data = request.get_json() or {}
    return jsonify(profiles.update_profile(profile_id, data))


@rag_bp.route("/retrieval/profiles/<int:profile_id>", methods=["DELETE"])
@require_role("admin")
def profiles_delete(profile_id):
    profiles.delete_profile(profile_id)
    return jsonify({"success": True})


@rag_bp.route("/retrieval/defaults", methods=["GET"])
@require_role("user")
def profile_defaults():
    """Startwerte fuer ein neues Profil (das Standardprofil) und der Standard-Prompt."""
    return jsonify(
        {"config": profiles.default_config(), "prompt": profiles.DEFAULT_QUERY_PROMPT}
    )


# =============================================================================
# Zuordnung am Knowledge
# =============================================================================


@rag_bp.route("/knowledge/<int:knowledge_id>/retrieval-profile", methods=["PUT"])
@require_role("user")
def knowledge_set_profile(knowledge_id):
    """Wechselt das Profil eines Knowledge (``profile_id: null`` = Standard).
    Wirkt sofort; die gespeicherten Vektoren bleiben unveraendert."""
    if not can_access_knowledge(current_user(), knowledge_id, "write"):
        return _error("Knowledge nicht gefunden", 404)
    profile_id = (request.get_json() or {}).get("profile_id")
    if profile_id is not None and not isinstance(profile_id, int):
        return _error("Ungültige Profil-ID", 400)
    profiles.set_knowledge_profile(knowledge_id, profile_id)
    return jsonify({"success": True, "profile_id": profile_id})


# =============================================================================
# Test mit Trace
# =============================================================================


@rag_bp.route("/retrieval/test", methods=["POST"])
@require_role("user")
def retrieval_test():
    """Fuehrt die Pipeline mit einem Profil aus und liefert Treffer plus Ablaufprotokoll.

    Body: ``question``, ``knowledge_ids`` (Liste; leer/fehlend = alle lesbaren),
    ``profile_id`` (gespeichertes Profil; ohne Angabe das Profil des jeweiligen
    Knowledge) oder ``config`` (Entwurf, nur Admin), optional ``model`` (Modell
    fuers Query-Enhancement; sonst das Standardmodell).
    """
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    if not question:
        return _error("Frage erforderlich", 400)

    user = current_user()
    readable = accessible_knowledge_ids(user, "read")
    wanted = data.get("knowledge_ids") or sorted(readable)
    targets = [
        k
        for k in list_knowledge()
        if k["id"] in readable and k["id"] in set(wanted) and k.get("kind") != "sql"
    ]
    if not targets:
        return _error("Kein lesbares Knowledge ausgewählt", 400)

    override = None
    if data.get("config") is not None:
        if not role_at_least(user["role"], "admin"):
            return _error("Entwürfe testen darf nur ein Admin", 403)
        override = profiles.validate_config(data["config"])
    elif data.get("profile_id") is not None:
        found = profiles.get_profile(int(data["profile_id"]))
        if not found:
            return _error("Profil nicht gefunden", 404)
        override = found["config"]

    # Helfer-LLM fuers Query-Enhancement
    from app.auth.access import can_use_model

    try:
        chat_model = registry.resolve_model(data.get("model"))
        if not can_use_model(user, chat_model.ref):
            chat_model = None
    except registry.ModelUnavailableError:
        chat_model = None
    helper = make_helper(chat_model, user)

    client = current_app.extensions.get("client")
    sparse = current_app.extensions.get("sparse_embeddings")
    if not (client and sparse):
        return _error("RAG-System wird noch initialisiert. Bitte kurz warten.", 503)

    results = []
    for k in targets:
        cfg = override or profiles.config_for_knowledge(k)
        trace = pipeline.Trace()
        try:
            dense = model_settings.embedder_for(k)
        except models_runtime.ModelLoadError as e:
            return _error(
                f"Embedding-Modell von „{k['name']}“ nicht verfügbar: {e}", 503
            )
        run_client, collection, layout = client, k["collection_name"], None
        if k.get("kind") == "external":
            from app.vector import sources

            src = sources.parse_config(k.get("source_config"))
            try:
                run_client = sources.connect(src)
            except sources.SourceError as e:
                return _error(f"„{k['name']}“: {e}", 502)
            collection, layout = src.collection, sources.layout_for(src)
        try:
            hits = pipeline.run(
                question,
                [collection],
                cfg,
                client=run_client,
                dense=dense,
                sparse=sparse_lang.resolve(
                    sparse,
                    sparse_lang.LEGACY_LANGUAGE
                    if k.get("kind") == "external"
                    else sparse_lang.knowledge_language(k),
                ),
                helper=helper,
                trace=trace,
                layout=layout,
            )
        except Exception as e:
            logger.exception("Retrieval-Test fehlgeschlagen")
            return _error(f"Pipeline-Fehler: {type(e).__name__}: {e}", 500)
        results.append(
            {
                "knowledge_id": k["id"],
                "knowledge": k["name"],
                "profile_id": None if override else k.get("retrieval_profile_id"),
                "documents": [
                    {
                        "filename": h.filename,
                        "score": round(h.score, 4),
                        "chunk_index": h.chunk_index,
                        "chunks": h.payload.get("_chunks"),
                        "text": h.text,
                    }
                    for h in hits
                ],
                "trace": trace.to_dict(),
            }
        )
    return jsonify(
        {
            "results": results,
            "helper_model": chat_model.display_name if chat_model else None,
        }
    )
