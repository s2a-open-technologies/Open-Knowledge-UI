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

"""Aufbewahrungsfrist fuer Chats (Datensparsamkeit, Art. 5 Abs. 1 lit. e DSGVO).

``CHAT_RETENTION_DAYS`` = 0 (Standard) bewahrt Chats unbegrenzt auf. Ein Wert > 0 loescht
Chats, die so viele Tage nicht mehr benutzt wurden (``sessions.updated_at``), samt
Nachrichten, Anhaengen (Dateien, Markdown, Vektoren) und Zusammenfassung. Der Lauf startet
kurz nach dem Start und dann taeglich; er ist idempotent, mehrere Prozesse schaden nicht.
"""

import asyncio
from datetime import datetime, timedelta
from logging import getLogger

from app.core.database_service import get_db_service
from app.core.models import ChatSession
from app.core.settings_service import SettingsService

logger = getLogger(__name__)

KEY = "CHAT_RETENTION_DAYS"
MAX_DAYS = 3650
INTERVAL_S = 24 * 3600
FIRST_RUN_DELAY_S = 60

SETTINGS = [
    (
        KEY,
        "0",
        "Chats nach so vielen Tagen ohne Nutzung loeschen, samt Anhaengen (0 = nie)",
    ),
]


class RetentionConfigError(ValueError):
    """Ungueltige Aufbewahrungsfrist."""


def load_settings() -> dict:
    session = get_db_service().get_session()
    try:
        return {"chat_retention_days": max(0, SettingsService(session).get_int(KEY, 0))}
    finally:
        session.close()


def save_settings(data: dict) -> None:
    if "chat_retention_days" not in data:
        return
    try:
        days = int(data["chat_retention_days"])
    except (TypeError, ValueError):
        raise RetentionConfigError("Aufbewahrungsfrist: ganze Zahl erwartet") from None
    if not 0 <= days <= MAX_DAYS:
        raise RetentionConfigError(
            f"Aufbewahrungsfrist: Wert zwischen 0 und {MAX_DAYS} Tagen erwartet"
        )
    session = get_db_service().get_session()
    try:
        SettingsService(session).set_batch({KEY: str(days)})
    finally:
        session.close()


def purge_expired(client, now: datetime | None = None) -> int:
    """Loescht abgelaufene Chats; liefert die Anzahl. Ohne Frist (0) passiert nichts."""
    from app.chat.attachments import cleanup_session_attachments

    days = load_settings()["chat_retention_days"]
    if days <= 0:
        return 0
    cutoff = (now or datetime.utcnow()) - timedelta(days=days)
    session = get_db_service().get_session()
    try:
        expired = [
            row.id
            for row in session.query(ChatSession.id).filter(
                ChatSession.updated_at < cutoff
            )
        ]
    finally:
        session.close()
    db = get_db_service()
    for session_id in expired:
        try:
            # Erst Anhaenge (Vektoren, Dateien, Zeilen), dann der Chat selbst
            cleanup_session_attachments(session_id, client)
            db.delete_session(session_id)
        except Exception:
            logger.exception("Abgelaufener Chat konnte nicht geloescht werden")
    if expired:
        logger.info(f"Aufbewahrungsfrist ({days} Tage): {len(expired)} Chat(s) geloescht")
    return len(expired)


async def run_periodically(get_client) -> None:
    """Hintergrund-Aufgabe fuer die Lebensdauer der App."""
    await asyncio.sleep(FIRST_RUN_DELAY_S)
    while True:
        try:
            await asyncio.to_thread(purge_expired, get_client())
        except Exception:
            logger.exception("Aufbewahrungsfrist: Lauf fehlgeschlagen")
        await asyncio.sleep(INTERVAL_S)
