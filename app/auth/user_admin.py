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

"""Verwaltung von Benutzern und Gruppen (Admin)."""

import re
from logging import getLogger

from sqlalchemy import func
from werkzeug.security import generate_password_hash

from app.auth.users import ROLES
from app.core.database_service import get_db_service
from app.core.models import ChatSession, Group, GroupMember, Message, User

logger = getLogger(__name__)

MIN_PASSWORD_LENGTH = 4
_USERNAME_RE = re.compile(r"^[A-Za-z0-9._@-]{3,80}$")


class AdminError(Exception):
    """Fachlicher Fehler mit HTTP-Status fuer die API."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def _user_row(u: User, group_ids: list[int]) -> dict:
    return {
        "id": u.id,
        "username": u.username,
        "display_name": u.display_name or u.username,
        "role": u.role,
        "is_active": bool(u.is_active),
        "group_ids": sorted(group_ids),
        "auth_source": u.auth_source or "local",
        "created_at": u.created_at.isoformat() if u.created_at else None,
    }


def _validate_password(password: str) -> str:
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise AdminError(
            f"Passwort muss mindestens {MIN_PASSWORD_LENGTH} Zeichen haben"
        )
    return password


def _active_admin_count(db, excluding_id: int | None = None) -> int:
    q = db.query(User).filter_by(role="admin", is_active=True)
    if excluding_id is not None:
        q = q.filter(User.id != excluding_id)
    return q.count()


# --------------------------------------------------------------------------
# Benutzer
# --------------------------------------------------------------------------


def list_users() -> list[dict]:
    db = get_db_service().get_session()
    try:
        groups: dict[int, list[int]] = {}
        for m in db.query(GroupMember):
            groups.setdefault(m.user_id, []).append(m.group_id)
        return [
            _user_row(u, groups.get(u.id, [])) for u in db.query(User).order_by(User.id)
        ]
    finally:
        db.close()


def create_user(username: str, display_name: str, password: str, role: str) -> dict:
    username = (username or "").strip()
    if not _USERNAME_RE.match(username):
        raise AdminError(
            "Benutzername: 3–80 Zeichen aus Buchstaben, Ziffern und . _ @ -"
        )
    if role not in ROLES:
        raise AdminError("Ungültige Rolle")
    _validate_password(password)

    db = get_db_service().get_session()
    try:
        if db.query(User).filter(func.lower(User.username) == username.lower()).first():
            raise AdminError("Benutzername ist bereits vergeben", 409)
        u = User(
            username=username,
            display_name=(display_name or "").strip() or username,
            password_hash=generate_password_hash(password),
            role=role,
            is_active=True,
        )
        db.add(u)
        db.commit()
        from app.auth import audit

        audit.record(
            audit.USER_CREATE,
            actor_user_id=u.id,
            object_type="user",
            object_id=u.id,
            detail={"rolle": role},
        )
        return _user_row(u, [])
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def update_user(user_id: int, actor_id: int, data: dict) -> dict:
    """Aktualisiert Name, Rolle, Aktiv-Status oder Passwort.

    Schutz vor Aussperren: Rolle und Aktiv-Status des eigenen Kontos sowie die
    letzte aktive Admin-Rolle sind nicht aenderbar.
    """
    db = get_db_service().get_session()
    try:
        u = db.query(User).filter_by(id=user_id).first()
        if not u:
            raise AdminError("Benutzer nicht gefunden", 404)

        new_role = data.get("role", u.role)
        new_active = bool(data.get("is_active", u.is_active))
        if new_role not in ROLES:
            raise AdminError("Ungültige Rolle")

        loses_admin = (
            u.role == "admin"
            and u.is_active
            and (new_role != "admin" or not new_active)
        )
        if user_id == actor_id and (new_role != u.role or new_active != u.is_active):
            raise AdminError(
                "Eigene Rolle und eigenen Status kannst du nicht ändern", 409
            )
        if loses_admin and _active_admin_count(db, excluding_id=user_id) == 0:
            raise AdminError("Der letzte aktive Admin kann nicht entfernt werden", 409)

        changes = {}
        if new_role != u.role:
            changes["rolle_alt"] = u.role
        if new_active != u.is_active:
            changes["aktiv_alt"] = u.is_active
        u.role, u.is_active = new_role, new_active
        if "display_name" in data:
            u.display_name = (data["display_name"] or "").strip() or u.username
        if data.get("password"):
            if (u.auth_source or "local") != "local":
                raise AdminError(
                    "Das Passwort externer Konten verwaltet der Anmeldedienst", 400
                )
            u.password_hash = generate_password_hash(
                _validate_password(data["password"])
            )
            changes["passwort"] = True
        db.commit()
        from app.auth import audit

        audit.record(
            audit.USER_UPDATE,
            actor_user_id=actor_id,
            actor_name=None,
            object_type="user",
            object_id=user_id,
            detail=changes,
        )
        groups = [m.group_id for m in db.query(GroupMember).filter_by(user_id=user_id)]
        return _user_row(u, groups)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_user(user_id: int, actor_id: int) -> list[str]:
    """Loescht den Benutzer samt Chats, Gruppenmitgliedschaften und Grants.

    Returns:
        Die IDs der geloeschten Chat-Sessions (fuer das Aufraeumen der Anhaenge).
    """
    db = get_db_service().get_session()
    try:
        u = db.query(User).filter_by(id=user_id).first()
        if not u:
            raise AdminError("Benutzer nicht gefunden", 404)
        if user_id == actor_id:
            raise AdminError("Du kannst dich nicht selbst löschen", 409)
        if u.role == "admin" and u.is_active and _active_admin_count(db, user_id) == 0:
            raise AdminError("Der letzte aktive Admin kann nicht gelöscht werden", 409)

        session_ids = [s.id for s in db.query(ChatSession).filter_by(user_id=user_id)]
        if session_ids:
            db.query(Message).filter(Message.session_id.in_(session_ids)).delete(
                synchronize_session=False
            )
            db.query(ChatSession).filter(ChatSession.id.in_(session_ids)).delete(
                synchronize_session=False
            )
        db.query(GroupMember).filter_by(user_id=user_id).delete()
        db.delete(u)
        db.commit()
        from app.auth import audit

        audit.record(
            audit.USER_DELETE,
            actor_user_id=actor_id,
            object_type="user",
            object_id=user_id,
            detail={"name": u.username},
        )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    from app.auth.access import delete_grants_for_principal

    delete_grants_for_principal("user", user_id)
    return session_ids


# --------------------------------------------------------------------------
# Gruppen
# --------------------------------------------------------------------------


def _group_row(g: Group, member_ids: list[int]) -> dict:
    return {
        "id": g.id,
        "name": g.name,
        "description": g.description or "",
        "member_ids": sorted(member_ids),
    }


def list_groups() -> list[dict]:
    db = get_db_service().get_session()
    try:
        members: dict[int, list[int]] = {}
        for m in db.query(GroupMember):
            members.setdefault(m.group_id, []).append(m.user_id)
        return [
            _group_row(g, members.get(g.id, []))
            for g in db.query(Group).order_by(func.lower(Group.name))
        ]
    finally:
        db.close()


def _check_group_name(db, name: str, exclude_id: int | None = None) -> str:
    name = (name or "").strip()
    if not name or len(name) > 80:
        raise AdminError("Gruppenname: 1–80 Zeichen")
    q = db.query(Group).filter(func.lower(Group.name) == name.lower())
    if exclude_id is not None:
        q = q.filter(Group.id != exclude_id)
    if q.first():
        raise AdminError("Gruppenname ist bereits vergeben", 409)
    return name


def create_group(name: str, description: str = "") -> dict:
    db = get_db_service().get_session()
    try:
        g = Group(
            name=_check_group_name(db, name), description=(description or "").strip()
        )
        db.add(g)
        db.commit()
        from app.auth import audit

        audit.record(
            audit.GROUP_CREATE,
            object_type="group",
            object_id=g.id,
            detail={"name": g.name},
        )
        return _group_row(g, [])
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def update_group(group_id: int, data: dict) -> dict:
    db = get_db_service().get_session()
    try:
        g = db.query(Group).filter_by(id=group_id).first()
        if not g:
            raise AdminError("Gruppe nicht gefunden", 404)
        if "name" in data:
            g.name = _check_group_name(db, data["name"], exclude_id=group_id)
        if "description" in data:
            g.description = (data["description"] or "").strip()
        db.commit()
        from app.auth import audit

        audit.record(
            audit.GROUP_UPDATE,
            object_type="group",
            object_id=g.id,
            detail={"name": g.name},
        )
        members = [
            m.user_id for m in db.query(GroupMember).filter_by(group_id=group_id)
        ]
        return _group_row(g, members)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def set_group_members(group_id: int, user_ids: list[int]) -> dict:
    db = get_db_service().get_session()
    try:
        g = db.query(Group).filter_by(id=group_id).first()
        if not g:
            raise AdminError("Gruppe nicht gefunden", 404)
        try:
            wanted = {int(i) for i in user_ids or []}
        except (TypeError, ValueError):
            raise AdminError("Ungültige Benutzer-IDs") from None
        existing = {u.id for u in db.query(User).filter(User.id.in_(wanted))}
        if existing != wanted:
            raise AdminError("Mindestens ein Benutzer existiert nicht")
        db.query(GroupMember).filter_by(group_id=group_id).delete()
        for uid in sorted(wanted):
            db.add(GroupMember(group_id=group_id, user_id=uid))
        db.commit()
        from app.auth import audit

        audit.record(
            audit.GROUP_MEMBERS_SET,
            object_type="group",
            object_id=group_id,
            detail={"mitglieder": sorted(wanted)},
        )
        return _group_row(g, sorted(wanted))
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_group(group_id: int) -> None:
    db = get_db_service().get_session()
    try:
        g = db.query(Group).filter_by(id=group_id).first()
        if not g:
            raise AdminError("Gruppe nicht gefunden", 404)
        db.query(GroupMember).filter_by(group_id=group_id).delete()
        db.delete(g)
        db.commit()
        from app.auth import audit

        audit.record(
            audit.GROUP_DELETE,
            object_type="group",
            object_id=group_id,
            detail={"name": g.name},
        )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    from app.auth.access import delete_grants_for_principal

    delete_grants_for_principal("group", group_id)
