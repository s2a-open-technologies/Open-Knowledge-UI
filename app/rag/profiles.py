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
Retrieval-Profile: validierte Konfiguration der Abruf-Pipeline (siehe pipeline.py).

Ein Profil gehoert zu keinem Modell, sondern wird am Knowledge ausgewaehlt
(``knowledge.retrieval_profile_id``, NULL = Standardprofil). Der Wechsel wirkt
sofort, weil nur der Abruf betroffen ist und nicht die gespeicherten Vektoren.
"""

import json
from logging import getLogger
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from app.core.database_service import get_db_service
from app.core.models import Knowledge, RetrievalProfile
from app.core.settings_service import SettingsService

logger = getLogger(__name__)

# Das Reranking-Modell ist global (Admin: "Einlesen & Modelle"); im Profil steht nur "default".
# Messung (13 deutsche Fragen, Treffer@1 ohne Reranking 10): bge-reranker-base 12, jina-multilingual 11.
DEFAULT_RERANK_MODEL = "default"

DEFAULT_QUERY_PROMPT = (
    "Du bist ein Suchassistent für eine Dokumentensuche. Erzeuge {anzahl} unterschiedliche "
    "Suchfragen, die zusammen alle Aspekte der folgenden Nutzerfrage abdecken "
    "(andere Formulierungen, Teilfragen, Synonyme, Fachbegriffe). "
    "Antworte ausschließlich mit den Fragen, eine pro Zeile, ohne Nummerierung und ohne Erklärung.\n\n"
    "Nutzerfrage: {frage}"
)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchConfig(_Section):
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    fusion: Literal["rrf", "dbsf"] = "rrf"
    dense_weight: float = Field(1.0, ge=0, le=10)
    sparse_weight: float = Field(1.0, ge=0, le=10)
    # Kandidaten je Suchanfrage und Collection
    candidates: int = Field(30, ge=1, le=200)
    # Mindest-Cosine der Dense-Suche (None = aus)
    dense_min_score: float | None = Field(0.2, ge=-1, le=1)


class RerankConfig(_Section):
    enabled: bool = False
    model: str = DEFAULT_RERANK_MODEL
    # So viele Kandidaten gehen in den Cross-Encoder
    candidates: int = Field(30, ge=1, le=100)
    min_score: float | None = None

    @field_validator("model")
    @classmethod
    def _global_model(cls, _v):
        return DEFAULT_RERANK_MODEL  # alte Profile/Clients: Modellname wird ignoriert


class MmrConfig(_Section):
    enabled: bool = True
    lambda_mult: float = Field(0.5, ge=0, le=1)


class QueryEnhancementConfig(_Section):
    mode: Literal["off", "multi_query"] = "off"
    n_queries: int = Field(3, ge=1, le=8)
    include_original: bool = True
    # None = das gerade chattende Modell
    model_ref: str | None = None
    prompt: str = DEFAULT_QUERY_PROMPT
    temperature: float = Field(0.3, ge=0, le=2)
    timeout_s: int = Field(30, ge=3, le=300)

    @model_validator(mode="after")
    def _prompt_needs_question(self):
        if self.mode != "off" and "{frage}" not in self.prompt:
            raise ValueError("Der Prompt muss den Platzhalter {frage} enthalten")
        return self


class ProfileConfig(_Section):
    search: SearchConfig = SearchConfig()
    rerank: RerankConfig = RerankConfig()
    mmr: MmrConfig = MmrConfig()
    query_enhancement: QueryEnhancementConfig = QueryEnhancementConfig()
    final_k: int = Field(5, ge=1, le=30)
    # Benachbarte Chunks je Treffer (braucht chunk_index in der Collection)
    neighbors: int = Field(0, ge=0, le=3)


class ProfileError(Exception):
    """Fachlicher Fehler mit HTTP-Status fuer die API."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def validate_config(raw: dict | None) -> dict:
    """Validiert und ergaenzt Defaults.

    Raises:
        ProfileError: mit lesbarer Meldung bei ungueltigen Werten.
    """
    try:
        return ProfileConfig.model_validate(raw or {}).model_dump()
    except ValidationError as e:
        first = e.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        msg = first["msg"].removeprefix("Value error, ")
        raise ProfileError(f"{where}: {msg}" if where else msg) from None


def _row(p: RetrievalProfile) -> dict:
    return {
        "id": p.id,
        "name": p.name,
        "description": p.description or "",
        "is_default": bool(p.is_default),
        "config": validate_config(json.loads(p.config)),
    }


# --------------------------------------------------------------------------
# Seed
# --------------------------------------------------------------------------


def _fast_config(settings: SettingsService) -> dict:
    """Bisheriges Verhalten (Hybrid + MMR, ohne Reranking) aus den RETRIEVER_*-Settings."""
    return validate_config(
        {
            "search": {
                "candidates": settings.get_int("RETRIEVER_FETCH_K", 30),
                "dense_min_score": settings.get_float("RETRIEVER_SCORE_THRESHOLD", 0.2),
            },
            "mmr": {
                "enabled": True,
                "lambda_mult": settings.get_float("RETRIEVER_LAMBDA", 0.5),
            },
            "final_k": settings.get_int("RETRIEVER_K", 5),
        }
    )


# Empfohlene Strategie: Stichwort- und Bedeutungssuche kombinieren, die besten Kandidaten
# mit einem Cross-Encoder neu sortieren, Nachbar-Chunks fuer vollstaendige Absaetze.
RECOMMENDED_CONFIG = {
    "search": {"candidates": 30, "dense_min_score": None},
    "rerank": {"enabled": True, "candidates": 30},
    "mmr": {"enabled": False},
    "final_k": 5,
    "neighbors": 1,
}

THOROUGH_CONFIG = {
    **RECOMMENDED_CONFIG,
    "query_enhancement": {"mode": "multi_query", "n_queries": 3},
    "final_k": 6,
}


def ensure_default_profiles(db_session) -> None:
    """Legt die mitgelieferten Profile an, wenn es noch keine gibt (idempotent).

    "Standard" ist die empfohlene Strategie und gilt, solange nichts anderes
    gewaehlt wird: Nutzer muessen nichts entscheiden. "Gruendlich" und
    "Schnell" sind Alternativen fuer Sonderfaelle.
    """
    if db_session.query(RetrievalProfile).count() > 0:
        return
    settings = SettingsService(db_session)
    seeds = [
        (
            "Standard",
            "Empfohlen: kombiniert Stichwort- und Bedeutungssuche und sortiert die Treffer neu.",
            RECOMMENDED_CONFIG,
            True,
        ),
        (
            "Gründlich",
            (
                "Wie Standard, zusätzlich formuliert ein LLM weitere Suchfragen. "
                "Hilft bei vagen Fragen, dauert aber länger."
            ),
            THOROUGH_CONFIG,
            False,
        ),
        (
            "Schnell",
            "Ohne Neusortierung, am schnellsten (bisheriges Verhalten).",
            _fast_config(settings),
            False,
        ),
    ]
    for name, description, config, is_default in seeds:
        db_session.add(
            RetrievalProfile(
                name=name,
                description=description,
                config=json.dumps(validate_config(config)),
                is_default=is_default,
            )
        )


# --------------------------------------------------------------------------
# CRUD
# --------------------------------------------------------------------------


def list_profiles() -> list[dict]:
    db = get_db_service().get_session()
    try:
        rows = db.query(RetrievalProfile).order_by(
            RetrievalProfile.is_default.desc(), RetrievalProfile.name
        )
        return [_row(p) for p in rows]
    finally:
        db.close()


def get_profile(profile_id: int | None) -> dict | None:
    """Profil zur ID; ``None`` als ID liefert das Standardprofil."""
    db = get_db_service().get_session()
    try:
        q = db.query(RetrievalProfile)
        p = (
            q.filter_by(is_default=True).first()
            if profile_id is None
            else q.filter_by(id=profile_id).first()
        )
        return _row(p) if p else None
    finally:
        db.close()


def default_config() -> dict:
    """Konfiguration des Standardprofils (Fallback: Pydantic-Defaults)."""
    p = get_profile(None)
    return p["config"] if p else validate_config({})


def config_for_knowledge(knowledge: dict) -> dict:
    """Wirksame Konfiguration eines Knowledge: sein Profil, sonst das Standardprofil."""
    pid = knowledge.get("retrieval_profile_id")
    p = get_profile(pid) if pid else None
    return p["config"] if p else default_config()


def _check_name(db, name: str, exclude_id: int | None = None) -> str:
    name = (name or "").strip()
    if not name or len(name) > 80:
        raise ProfileError("Name: 1–80 Zeichen")
    q = db.query(RetrievalProfile).filter(RetrievalProfile.name == name)
    if exclude_id is not None:
        q = q.filter(RetrievalProfile.id != exclude_id)
    if q.first():
        raise ProfileError("Ein Profil mit diesem Namen existiert bereits", 409)
    return name


def create_profile(name: str, description: str, config: dict | None) -> dict:
    cfg = validate_config(config)
    db = get_db_service().get_session()
    try:
        p = RetrievalProfile(
            name=_check_name(db, name),
            description=(description or "").strip(),
            config=json.dumps(cfg),
            is_default=False,
        )
        db.add(p)
        db.commit()
        return _row(p)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def update_profile(profile_id: int, data: dict) -> dict:
    db = get_db_service().get_session()
    try:
        p = db.query(RetrievalProfile).filter_by(id=profile_id).first()
        if not p:
            raise ProfileError("Profil nicht gefunden", 404)
        if "name" in data:
            p.name = _check_name(db, data["name"], exclude_id=profile_id)
        if "description" in data:
            p.description = (data["description"] or "").strip()
        if "config" in data:
            p.config = json.dumps(validate_config(data["config"]))
        db.commit()
        return _row(p)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_profile(profile_id: int) -> None:
    """Loescht ein Profil; Knowledge mit diesem Profil faellt auf den Standard zurueck."""
    db = get_db_service().get_session()
    try:
        p = db.query(RetrievalProfile).filter_by(id=profile_id).first()
        if not p:
            raise ProfileError("Profil nicht gefunden", 404)
        if p.is_default:
            raise ProfileError("Das Standardprofil kann nicht gelöscht werden", 409)
        db.query(Knowledge).filter_by(retrieval_profile_id=profile_id).update(
            {Knowledge.retrieval_profile_id: None}
        )
        db.delete(p)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def set_knowledge_profile(knowledge_id: int, profile_id: int | None) -> bool:
    """Weist einem Knowledge ein Profil zu (``None`` = Standard).

    Raises:
        ProfileError: wenn das Profil nicht existiert.
    """
    db = get_db_service().get_session()
    try:
        k = db.query(Knowledge).filter_by(id=knowledge_id).first()
        if not k:
            return False
        if (
            profile_id is not None
            and not db.query(RetrievalProfile).filter_by(id=profile_id).first()
        ):
            raise ProfileError("Profil nicht gefunden", 404)
        k.retrieval_profile_id = profile_id
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
