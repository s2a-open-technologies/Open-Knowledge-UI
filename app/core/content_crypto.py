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

"""Chat-Inhalte in der Datenbank mit dem Schluessel des Nutzers verschluesseln (ORM-Ereignisse).

Verschluesselt werden ``sessions.title``, ``sessions.summary`` (Compaction) und
``messages.content``. Der Code ausserhalb sieht weiter Klartext: beim Laden wird entschluesselt
(ohne das Objekt als geaendert zu markieren), vor dem Schreiben verschluesselt und danach wieder
der Klartext ins Objekt gelegt. Chats ohne Besitzer (Altbestand vor dem Nutzermodell) bleiben im
Klartext. Bulk-Updates (``query.update``) umgehen die Ereignisse und sind fuer diese Spalten tabu.
"""

import threading

from sqlalchemy import event, inspect, select
from sqlalchemy.orm.attributes import set_committed_value

from app.core import keys
from app.core.models import ChatSession, Message

_lock = threading.Lock()
_session_owner: dict[str, int | None] = {}  # session_id -> user_id (aendert sich nie)

SESSION_FIELDS = ("title", "summary")
MESSAGE_FIELDS = ("content",)
_STASH = "_okc_plain"


def reset() -> None:
    with _lock:
        _session_owner.clear()


def _owner_of_session(session_id: str, conn=None) -> int | None:
    with _lock:
        if session_id in _session_owner:
            return _session_owner[session_id]
    t = ChatSession.__table__

    def fetch(c):
        return c.execute(select(t.c.user_id).where(t.c.id == session_id)).scalar()

    owner = keys._run(conn, fetch)
    if owner is not None:
        with _lock:
            _session_owner[session_id] = owner
    return owner


def _owner(target, conn=None) -> int | None:
    if isinstance(target, ChatSession):
        return target.user_id
    return _owner_of_session(target.session_id, conn)


def _fields(target):
    return SESSION_FIELDS if isinstance(target, ChatSession) else MESSAGE_FIELDS


def _open(target, conn=None) -> None:
    fields = [f for f in _fields(target) if keys.is_sealed_text(target.__dict__.get(f))]
    if not fields:
        return
    owner = _owner(target, conn)
    for field in fields:
        set_committed_value(
            target, field, keys.unseal_for(keys.USER, owner, target.__dict__[field], conn)
        )


def _on_load(target, context):
    conn = None
    try:
        conn = context.session.connection()
    except Exception:  # noqa: BLE001 — ohne Session-Verbindung eine eigene nehmen
        conn = None
    _open(target, conn)


def _on_refresh(target, context, attrs):
    _on_load(target, context)


def _before_write(mapper, connection, target):
    owner = _owner(target, connection)
    if owner is None:
        return
    state = inspect(target)
    plain = {}
    for field in _fields(target):
        value = target.__dict__.get(field)
        if value is None or value == "" or keys.is_sealed_text(value):
            continue
        if state.persistent and not state.attrs[field].history.has_changes():
            continue
        plain[field] = value
        setattr(target, field, keys.seal_for(keys.USER, owner, value, connection))
    if plain:
        target.__dict__[_STASH] = plain


def _after_write(mapper, connection, target):
    plain = target.__dict__.pop(_STASH, None)
    for field, value in (plain or {}).items():
        set_committed_value(target, field, value)


_installed = False


def install() -> None:
    """Ereignisse einmal registrieren (beim Import von app.core.models)."""
    global _installed
    if _installed:
        return
    for model in (ChatSession, Message):
        event.listen(model, "load", _on_load)
        event.listen(model, "refresh", _on_refresh)
        event.listen(model, "before_insert", _before_write)
        event.listen(model, "before_update", _before_write)
        event.listen(model, "after_insert", _after_write)
        event.listen(model, "after_update", _after_write)
    _installed = True
