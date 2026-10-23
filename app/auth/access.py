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
Zugriff auf Knowledge und Modelle (Freigaben).

Eine Ressource ist fuer einen Nutzer zugreifbar, wenn die Ressource oeffentlich ist
(nur lesen/nutzen) oder ein Grant fuer ihn oder eine seiner Gruppen existiert.
``write`` schliesst ``read`` ein. Admins sehen Knowledge nicht automatisch: auch fuer sie gilt
nur, was oeffentlich oder ihnen freigegeben ist (Modelle nutzen und konfigurieren duerfen sie weiter alle).

Ressourcentypen: ``knowledge`` (Schluessel: knowledge.id) und ``model``
(Schluessel: model_configs.id; ein Basismodell ohne Konfiguration ist oeffentlich).
Die Rollenpruefung (``editor`` fuer Freigaben, ``admin`` fuer Fremdquellen) sitzt an den Endpunkten, nicht hier.
Ausnahme Knowledge: Ein Editor darf alles bearbeiten, was einer seiner Gruppen freigegeben ist (auch mit
Lese-Freigabe). Der Rolle "Benutzer" bleibt es beim vergebenen Level; eigenes Knowledge verwaltet er ueber
das Schreibrecht, das der Ersteller automatisch bekommt.
"""

from logging import getLogger

from app.core.database_service import get_db_service
from app.core.models import (
    Group,
    GroupMember,
    Knowledge,
    ModelConfig,
    ResourceGrant,
    User,
)
from app.llm import connections as models_store

logger = getLogger(__name__)

RESOURCE_TYPES = ("knowledge", "model")
PRINCIPAL_TYPES = ("user", "group")
LEVELS = ("read", "write")


def _is_admin(user: dict) -> bool:
    return user.get("role") == "admin"


def _group_ids(db, user_id: int) -> set[int]:
    return {m.group_id for m in db.query(GroupMember).filter_by(user_id=user_id)}


def _granted(db, user: dict, rtype: str, rid: int, level: str) -> bool:
    groups = _group_ids(db, user["id"])
    wanted = ("write",) if level == "write" else LEVELS
    # Editoren duerfen alles bearbeiten, was einer ihrer Gruppen freigegeben ist (auch bei Lese-Freigabe)
    group_wanted = (
        LEVELS
        if (level == "write" and rtype == "knowledge" and user.get("role") == "editor")
        else wanted
    )
    for g in db.query(ResourceGrant).filter_by(resource_type=rtype, resource_id=rid):
        if (
            g.principal_type == "user"
            and g.principal_id == user["id"]
            and g.access in wanted
        ):
            return True
        if (
            g.principal_type == "group"
            and g.principal_id in groups
            and g.access in group_wanted
        ):
            return True
    return False


# --------------------------------------------------------------------------
# Knowledge
# --------------------------------------------------------------------------


def can_access_knowledge(user: dict, knowledge_id: int, level: str = "read") -> bool:
    db = get_db_service().get_session()
    try:
        k = db.query(Knowledge).filter_by(id=knowledge_id).first()
        if k is None:
            return False
        if level == "read" and k.is_public:
            return True
        return _granted(db, user, "knowledge", knowledge_id, level)
    finally:
        db.close()


def accessible_knowledge_ids(user: dict, level: str = "read") -> set[int]:
    db = get_db_service().get_session()
    try:
        ids = set()
        for k in db.query(Knowledge):
            if (level == "read" and k.is_public) or _granted(
                db, user, "knowledge", k.id, level
            ):
                ids.add(k.id)
        return ids
    finally:
        db.close()


# --------------------------------------------------------------------------
# Modelle
# --------------------------------------------------------------------------


def can_use_model(user: dict, ref: str) -> bool:
    """Darf der Nutzer das Modell verwenden?

    Raises:
        ValueError: bei ungueltiger Referenz.
    """
    kind, key, model = models_store.parse_ref(ref)
    if _is_admin(user):
        return True
    db = get_db_service().get_session()
    try:
        q = db.query(ModelConfig)
        if kind == "clone":
            cfg = q.filter_by(id=key, is_clone=True).first()
            if cfg is None:
                return False
        else:
            cfg = q.filter_by(
                connection_id=key, base_model=model, is_clone=False
            ).first()
            if cfg is None:
                return True  # nie angepasstes Basismodell: oeffentlich
        return bool(cfg.is_public) or _granted(db, user, "model", cfg.id, "read")
    finally:
        db.close()


def model_resource_id(ref: str) -> int | None:
    """Schluessel der Ressource (model_configs.id) oder ``None`` ohne Konfiguration."""
    cfg = models_store.get_model_config(ref)
    return cfg["id"] if cfg else None


# --------------------------------------------------------------------------
# Grants verwalten
# --------------------------------------------------------------------------


def _principal_name(db, ptype: str, pid: int) -> str | None:
    if ptype == "user":
        u = db.query(User).filter_by(id=pid).first()
        return (u.display_name or u.username) if u else None
    g = db.query(Group).filter_by(id=pid).first()
    return g.name if g else None


def get_access(rtype: str, rid: int) -> dict | None:
    """``{"is_public": bool, "grants": [...]}`` einer Ressource oder ``None``."""
    db = get_db_service().get_session()
    try:
        model = Knowledge if rtype == "knowledge" else ModelConfig
        res = db.query(model).filter_by(id=rid).first()
        if res is None:
            return None
        grants = []
        for g in db.query(ResourceGrant).filter_by(
            resource_type=rtype, resource_id=rid
        ):
            name = _principal_name(db, g.principal_type, g.principal_id)
            if name is None:
                continue  # verwaister Grant (Principal geloescht)
            grants.append(
                {
                    "principal_type": g.principal_type,
                    "principal_id": g.principal_id,
                    "name": name,
                    "access": g.access,
                }
            )
        return {"is_public": bool(res.is_public), "grants": grants}
    finally:
        db.close()


def set_access(rtype: str, rid: int, is_public: bool, grants: list[dict]) -> bool:
    """Ersetzt Oeffentlich-Flag und alle Grants einer Ressource.

    Raises:
        ValueError: bei unbekanntem Typ/Principal oder ungueltiger Zugriffsstufe.
    """
    if rtype not in RESOURCE_TYPES:
        raise ValueError(f"Unbekannter Ressourcentyp: {rtype}")

    db = get_db_service().get_session()
    try:
        model = Knowledge if rtype == "knowledge" else ModelConfig
        res = db.query(model).filter_by(id=rid).first()
        if res is None:
            return False

        clean: dict[tuple[str, int], str] = {}
        for g in grants or []:
            ptype, access = g.get("principal_type"), g.get("access", "read")
            try:
                pid = int(g.get("principal_id"))
            except (TypeError, ValueError):
                raise ValueError("Ungültige Principal-ID") from None
            if ptype not in PRINCIPAL_TYPES or access not in LEVELS:
                raise ValueError("Ungültige Freigabe")
            if _principal_name(db, ptype, pid) is None:
                raise ValueError("Nutzer bzw. Gruppe existiert nicht")
            # write gewinnt, wenn ein Principal doppelt vorkommt
            if clean.get((ptype, pid)) != "write":
                clean[(ptype, pid)] = access

        res.is_public = bool(is_public)
        db.query(ResourceGrant).filter_by(resource_type=rtype, resource_id=rid).delete()
        for (ptype, pid), access in clean.items():
            db.add(
                ResourceGrant(
                    resource_type=rtype,
                    resource_id=rid,
                    principal_type=ptype,
                    principal_id=pid,
                    access=access,
                )
            )
        db.commit()

        from app.auth import audit

        audit.record(
            audit.ACCESS_SET,
            object_type=rtype,
            object_id=rid,
            detail={"oeffentlich": is_public, "freigaben": len(clean)},
        )
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def add_grant(rtype: str, rid: int, ptype: str, pid: int, access: str) -> None:
    """Fuegt einen Grant hinzu bzw. erhoeht die Stufe (idempotent)."""
    db = get_db_service().get_session()
    try:
        g = (
            db.query(ResourceGrant)
            .filter_by(
                resource_type=rtype,
                resource_id=rid,
                principal_type=ptype,
                principal_id=pid,
            )
            .first()
        )
        if g is None:
            db.add(
                ResourceGrant(
                    resource_type=rtype,
                    resource_id=rid,
                    principal_type=ptype,
                    principal_id=pid,
                    access=access,
                )
            )
        elif access == "write":
            g.access = "write"
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_grants_for_resource(rtype: str, rid: int) -> None:
    db = get_db_service().get_session()
    try:
        db.query(ResourceGrant).filter_by(resource_type=rtype, resource_id=rid).delete()
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_grants_for_principal(ptype: str, pid: int) -> None:
    db = get_db_service().get_session()
    try:
        db.query(ResourceGrant).filter_by(
            principal_type=ptype, principal_id=pid
        ).delete()
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def user_group_names(user_id: int) -> list[str]:
    """Namen der lokalen Gruppen eines Benutzers (fuer Zugriffsfilter externer Quellen)."""
    db = get_db_service().get_session()
    try:
        ids = _group_ids(db, user_id)
        if not ids:
            return []
        return sorted(g.name for g in db.query(Group).filter(Group.id.in_(ids)))
    finally:
        db.close()
