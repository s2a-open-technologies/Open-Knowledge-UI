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

from fastapi import APIRouter

from app.api.errors import ApiError
from app.api.schemas.common import Success
from app.api.schemas.retrieval import (
    Defaults,
    Profile,
    ProfileAssign,
    ProfileAssigned,
    ProfileCreate,
    ProfileUpdate,
    RetrievalTestRequest,
    RetrievalTestResponse,
)
from app.auth.access import accessible_knowledge_ids, can_access_knowledge
from app.auth.auth import Admin, CurrentUser
from app.auth.users import role_at_least
from app.core import runtime
from app.knowledge.service import list_knowledge
from app.llm import registry
from app.rag import model_settings, models_runtime, pipeline, profiles
from app.rag import sparse as sparse_lang
from app.rag.helper import make_helper

router = APIRouter(prefix="/api", tags=["retrieval"])

logger = getLogger(__name__)


def _error(message: str, status: int) -> ApiError:
    return ApiError(status, message, success=False)


# =============================================================================
# Profile
# =============================================================================


@router.get("/retrieval/profiles", response_model=list[Profile])
def profiles_list(_user: CurrentUser):
    return profiles.list_profiles()


@router.post("/retrieval/profiles", response_model=Profile, status_code=201)
def profiles_create(_user: Admin, body: ProfileCreate | None = None):
    body = body or ProfileCreate()
    return profiles.create_profile(body.name, body.description, body.config)


@router.patch("/retrieval/profiles/{profile_id:int}", response_model=Profile)
def profiles_update(profile_id: int, _user: Admin, body: ProfileUpdate | None = None):
    return profiles.update_profile(profile_id, body.model_dump(exclude_unset=True) if body else {})


@router.delete("/retrieval/profiles/{profile_id:int}", response_model=Success)
def profiles_delete(profile_id: int, _user: Admin):
    profiles.delete_profile(profile_id)
    return Success()


@router.get("/retrieval/defaults", response_model=Defaults)
def profile_defaults(_user: CurrentUser):
    """Startwerte fuer ein neues Profil (das Standardprofil) und der Standard-Prompt."""
    return {"config": profiles.default_config(), "prompt": profiles.DEFAULT_QUERY_PROMPT}


# =============================================================================
# Zuordnung am Knowledge
# =============================================================================


@router.put("/knowledge/{knowledge_id:int}/retrieval-profile", response_model=ProfileAssigned)
def knowledge_set_profile(knowledge_id: int, user: CurrentUser, body: ProfileAssign | None = None):
    """Wechselt das Profil eines Knowledge (``profile_id: null`` = Standard).
    Wirkt sofort; die gespeicherten Vektoren bleiben unveraendert."""
    if not can_access_knowledge(user, knowledge_id, "write"):
        raise _error("Knowledge nicht gefunden", 404)
    profile_id = body.profile_id if body else None
    if profile_id is not None and not isinstance(profile_id, int):
        raise _error("Ungültige Profil-ID", 400)
    profiles.set_knowledge_profile(knowledge_id, profile_id)
    return {"success": True, "profile_id": profile_id}


# =============================================================================
# Test mit Trace
# =============================================================================


@router.post("/retrieval/test", response_model=RetrievalTestResponse)
def retrieval_test(user: CurrentUser, body: RetrievalTestRequest | None = None):
    """Fuehrt die Pipeline mit einem Profil aus und liefert Treffer plus Ablaufprotokoll.

    Body: ``question``, ``knowledge_ids`` (Liste; leer/fehlend = alle lesbaren),
    ``profile_id`` (gespeichertes Profil; ohne Angabe das Profil des jeweiligen
    Knowledge) oder ``config`` (Entwurf, nur Admin), optional ``model`` (Modell
    fuers Query-Enhancement; sonst das Standardmodell).
    """
    data = (body or RetrievalTestRequest()).model_dump()
    question = (data.get("question") or "").strip()
    if not question:
        raise _error("Frage erforderlich", 400)

    readable = accessible_knowledge_ids(user, "read")
    wanted = data.get("knowledge_ids") or sorted(readable)
    targets = [
        k
        for k in list_knowledge()
        if k["id"] in readable and k["id"] in set(wanted) and k.get("kind") != "sql"
    ]
    if not targets:
        raise _error("Kein lesbares Knowledge ausgewählt", 400)

    override = None
    if data.get("config") is not None:
        if not role_at_least(user["role"], "admin"):
            raise _error("Entwürfe testen darf nur ein Admin", 403)
        override = profiles.validate_config(data["config"])
    elif data.get("profile_id") is not None:
        found = profiles.get_profile(int(data["profile_id"]))
        if not found:
            raise _error("Profil nicht gefunden", 404)
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

    client = runtime.state.client
    sparse = runtime.state.sparse_embeddings
    if not (client and sparse):
        raise _error("RAG-System wird noch initialisiert. Bitte kurz warten.", 503)

    results = []
    for k in targets:
        cfg = override or profiles.config_for_knowledge(k)
        trace = pipeline.Trace()
        try:
            dense = model_settings.embedder_for(k)
        except models_runtime.ModelLoadError as e:
            raise _error(
                f"Embedding-Modell von „{k['name']}“ nicht verfügbar: {e}", 503
            )
        run_client, collection, layout = client, k["collection_name"], None
        if k.get("kind") == "external":
            from app.knowledge import external as sources

            src = sources.parse_config(k.get("source_config"))
            try:
                run_client = sources.connect(src)
            except sources.SourceError as e:
                raise _error(f"„{k['name']}“: {e}", 502)
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
            raise _error(f"Pipeline-Fehler: {type(e).__name__}: {e}", 500)
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
    return {
        "results": results,
        "helper_model": chat_model.display_name if chat_model else None,
    }
