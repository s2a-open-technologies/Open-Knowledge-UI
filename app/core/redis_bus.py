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

"""Chat-Stopp ueber Prozessgrenzen: Redis Pub/Sub plus kurzlebiger Schluessel als Backstop.

``REDIS_URL`` kommt nur aus der Umgebung (Infrastruktur wie ``QDRANT_*``). Ohne Redis oder wenn
es nicht erreichbar ist, funktioniert der Stopp im eigenen Prozess weiter (``app/chat/streams.py``);
nur ein Stream auf einem anderen Worker ist dann nicht erreichbar. Der Start bricht nie ab.

* :func:`publish_cancel` setzt ``chat:cancel:<stream_id>`` (TTL 10 min) und sendet auf
  ``chat:cancel``. Der Schluessel faengt den Fall ab, dass der Stopp ankommt, bevor der Stream
  auf dem anderen Worker registriert ist (:func:`cancel_requested` beim Start des Streams).
* :func:`start` abonniert den Kanal und bricht passende lokale Streams ab.
"""

import asyncio
import json
import logging
import os

from app.chat.streams import registry

logger = logging.getLogger(__name__)

CHANNEL = "chat:cancel"
KEY_PREFIX = "chat:cancel:"
KEY_TTL_S = 600

_client = None
_listener: asyncio.Task | None = None
_status = "off"  # off | ok | unavailable


def url() -> str:
    return (os.getenv("REDIS_URL") or "").strip()


def status() -> str:
    return _status


def set_client(client) -> None:
    """Client direkt setzen (Tests mit fakeredis)."""
    global _client, _status
    _client = client
    _status = "ok" if client is not None else "off"


async def start() -> None:
    global _client, _listener, _status
    if _client is None:
        if not url():
            logger.info("REDIS_URL nicht gesetzt: Chat-Stopp nur im eigenen Prozess")
            return
        import redis.asyncio as redis

        _client = redis.from_url(url(), decode_responses=True)
    try:
        await _client.ping()
        _status = "ok"
    except Exception as e:
        _status = "unavailable"
        logger.warning(f"Redis nicht erreichbar ({e}): Chat-Stopp nur im eigenen Prozess")
    _listener = asyncio.create_task(_listen(), name="redis-chat-cancel")


async def stop() -> None:
    global _listener, _client, _status
    if _listener is not None:
        _listener.cancel()
        try:
            await _listener
        except (asyncio.CancelledError, Exception):
            pass
        _listener = None
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:
            pass
    _client = None
    _status = "off"


def handle_message(data: str) -> bool:
    """Eine Kanal-Nachricht auswerten und den lokalen Stream abbrechen (falls er hier laeuft)."""
    try:
        msg = json.loads(data)
        return registry.cancel(str(msg["stream_id"]), int(msg["user_id"]))
    except (ValueError, KeyError, TypeError):
        return False


async def _listen() -> None:
    """Kanal abonnieren; bei Verbindungsfehlern mit wachsender Pause neu verbinden."""
    global _status
    delay = 1.0
    while True:
        try:
            pubsub = _client.pubsub()
            await pubsub.subscribe(CHANNEL)
            _status, delay = "ok", 1.0
            async for message in pubsub.listen():
                if message.get("type") == "message":
                    handle_message(message.get("data"))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if _status != "unavailable":
                logger.warning(f"Redis-Abo unterbrochen ({e}); neuer Versuch")
            _status = "unavailable"
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)


async def publish_cancel(stream_id: str, user_id: int) -> bool:
    """Stopp an alle Worker melden; False ohne (erreichbares) Redis."""
    if _client is None:
        return False
    try:
        await _client.set(KEY_PREFIX + stream_id, str(user_id), ex=KEY_TTL_S)
        await _client.publish(CHANNEL, json.dumps({"stream_id": stream_id, "user_id": user_id}))
        return True
    except Exception as e:
        logger.warning(f"Chat-Stopp ueber Redis fehlgeschlagen: {e}")
        return False


async def cancel_requested(stream_id: str, user_id: int) -> bool:
    """Wurde dieser Stream schon gestoppt, bevor er hier lief?"""
    if _client is None:
        return False
    try:
        owner = await _client.get(KEY_PREFIX + stream_id)
        return owner is not None and str(owner) == str(user_id)
    except Exception:
        return False
