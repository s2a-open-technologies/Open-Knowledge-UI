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
Audit-Log: sicherheitsrelevante Ereignisse nachweisen koennen.

``record`` ist **best-effort** und darf niemals den eigentlichen Vorgang brechen:
eine eigene kurze DB-Sitzung, Fehler nur in das Log. Aufgerufen wird von zentralen
Stellen: Anmeldung (``app/auth/users.authenticate``), externe Anmeldungen
(``auth_external.login_external_user``), Abmeldung, Kontoverwaltung
(``user_admin``), Freigaben (``access.set_access``), Passwortwechsel und
Datenexport (``api_routes``).

``detail`` wird als JSON-Text gespeichert (kleine, flache Struktur); Namen und
Rollen liegen schon im Klartext in der Tabelle, Passwoerter werden nie gespeichert.
"""

import json
from logging import getLogger

from flask import has_request_context, request

from app.core.database_service import get_db_service
from app.core.models import AuditLog

log = getLogger(__name__)

# Aktionen der zentralen Hook-Stellen
LOGIN_OK = "login_ok"
LOGIN_FAILED = "login_failed"
LOGIN_THROTTLED = "login_throttled"
LOGIN_EXTERNAL_OK = "login_external_ok"
LOGIN_EXTERNAL_FAILED = "login_external_failed"
LOGOUT = "logout"
PASSWORD_CHANGE = "password_change"
DATA_EXPORT = "data_export"
USER_CREATE = "user_create"
USER_UPDATE = "user_update"
USER_DELETE = "user_delete"
GROUP_CREATE = "group_create"
GROUP_UPDATE = "group_update"
GROUP_DELETE = "group_delete"
GROUP_MEMBERS_SET = "group_members_set"
ACCESS_SET = "access_set"


def _request_ip():
    if not has_request_context():
        return None
    return request.remote_addr


def record(
    action: str,
    *,
    actor_user_id: int | None = None,
    actor_name: str | None = None,
    object_type: str | None = None,
    object_id=None,
    detail=None,
    db=None,
) -> None:
    """Ereignis nachweisen; scheitert hoestens im Log, nie im Aufrufer.

    Mit ``db`` wird der Eintrag in die übergebene Sitzung eingefügt (nur add, kein
    Commit — der Aufrufer committet sein Vorgangs-Update atomar mit dem Protokoll);
    ohne ``db`` geht der Eintrag über eine eigene kurze Sitzung.
    """
    if has_request_context() and (actor_user_id is None or actor_name is None):
        try:
            from app.auth.auth import current_user

            user = current_user()
            if user:
                if actor_user_id is None:
                    actor_user_id, actor_name = user["id"], user["username"]
                elif actor_name is None and actor_user_id == user["id"]:
                    actor_name = user["username"]
        except Exception:  # noqa: BLE001, S110 — Audit darf nichts blockieren; Name ist optional
            pass

    if isinstance(detail, (dict, list)):
        detail = json.dumps(detail, ensure_ascii=False)
    if isinstance(detail, str) and len(detail) > 2000:
        detail = detail[:2000]

    row = AuditLog(
        actor_user_id=actor_user_id,
        actor_name=actor_name,
        ip=_request_ip(),
        action=action,
        object_type=object_type,
        object_id=str(object_id) if object_id is not None else None,
        detail=detail,
    )

    if db is not None:
        db.add(row)
        return

    try:
        session = get_db_service().get_session()
        try:
            session.add(row)
            session.commit()
        finally:
            session.close()
    except Exception:
        log.exception("Audit-Eintrag fehlgeschlagen (Aktion %r)", action)


def list_events(
    limit: int = 100, offset: int = 0, action: str | None = None
) -> list[dict]:
    """Neueste Ereignisse fuer die Admin-Protokoll-Ansicht."""
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    db = get_db_service().get_session()
    try:
        q = db.query(AuditLog).order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
        if action:
            q = q.filter(AuditLog.action == action)
        rows = q.limit(limit).offset(offset).all()
        return [
            {
                "id": row.id,
                "created_at": row.created_at.isoformat(timespec="seconds")
                if row.created_at
                else None,
                "actor_user_id": row.actor_user_id,
                "actor_name": row.actor_name,
                "ip": row.ip,
                "action": row.action,
                "object_type": row.object_type,
                "object_id": row.object_id,
                "detail": row.detail,
            }
            for row in rows
        ]
    finally:
        db.close()
