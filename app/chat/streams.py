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

"""Laufende Chat-Streams dieses Prozesses: ``stream_id`` -> Task, zum sofortigen Abbruch.

Die ``stream_id`` erzeugt der Browser, damit auch der allererste Chat (noch ohne Session) stoppbar
ist. Abbrechen heisst: den asyncio-Task canceln. Der Agent und der LLM-Stream enden sofort; ein
bereits laufendes synchrones Werkzeug (Pandas/SQL im Tool-Worker) rechnet bis zu seinem Timeout
weiter, sein Ergebnis wird verworfen. Prozessuebergreifend verteilt ``app/core/redis_bus.py`` den
Stopp.
"""

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class _Stream:
    task: asyncio.Task
    user_id: int
    connection: object | None
    loop: asyncio.AbstractEventLoop


class StreamRegistry:
    def __init__(self):
        self._streams: dict[str, _Stream] = {}

    def add(self, stream_id: str, task: asyncio.Task, user_id: int, connection=None) -> bool:
        """Stream eintragen; False, wenn die ID schon laeuft."""
        if stream_id in self._streams:
            return False
        self._streams[stream_id] = _Stream(task, user_id, connection, task.get_loop())
        task.add_done_callback(lambda _t: self._streams.pop(stream_id, None))
        return True

    def owner(self, stream_id: str) -> int | None:
        stream = self._streams.get(stream_id)
        return stream.user_id if stream else None

    def cancel(self, stream_id: str, user_id: int | None = None) -> bool:
        """Task abbrechen. Mit ``user_id`` nur den eigenen Stream (fremde IDs: False, wie unbekannt)."""
        stream = self._streams.get(stream_id)
        if stream is None or (user_id is not None and stream.user_id != user_id):
            return False
        if not stream.task.done():
            # threadsicher: der Stopp kann aus einem anderen Thread/Loop kommen (Threadpool, Redis)
            stream.loop.call_soon_threadsafe(stream.task.cancel)
        return True

    def cancel_connection(self, connection) -> int:
        """Alle Streams einer WebSocket-Verbindung abbrechen (Socket geschlossen)."""
        ids = [sid for sid, s in self._streams.items() if s.connection is connection]
        for sid in ids:
            self.cancel(sid)
        return len(ids)

    def active(self) -> int:
        return len(self._streams)


registry = StreamRegistry()
