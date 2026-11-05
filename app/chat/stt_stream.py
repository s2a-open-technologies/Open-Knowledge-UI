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
WebSocket-Proxy: Browser <-> Parakeet-Streaming-ASR-Server (asynchron).

Pro Browser-Verbindung (``/ws/stt-stream``, ``app/api/chat.py``) wird eine eigene Verbindung zum
internen Parakeet-Service (``/v1/audio/stream``, siehe parakeet/main.py) geoeffnet und in beide
Richtungen durchgereicht:
  * Binäre Float32-PCM-Frames vom Browser -> Parakeet,
  * JSON-Nachrichten (``partial``/``committed``/``final``/``error``) von
    Parakeet -> Browser,
  * Text-Frame ``{"event": "stop"}`` vom Browser -> Parakeet, danach wird auf
    die abschließende ``final``-Nachricht gewartet, bevor beide Seiten
    geschlossen werden.
"""

import asyncio
import json
import logging
import urllib.parse

from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocket

from app.core.database_service import get_db_service
from app.core.settings_service import SettingsService

logger = logging.getLogger(__name__)

_STOP_JOIN_TIMEOUT_S = 15.0


def _build_stream_url(base_url: str) -> str:
    """Wandelt STT_BASEURL (z.B. http://parakeet:5001/v1) in die Parakeet-WS-URL um.

    Gleiche Umwandlung wie im openDox-`ParakeetStreamBackend`-Client:
    Schema auf ws/wss umschreiben, Streaming-Pfad anhängen.
    """
    parsed = urllib.parse.urlparse(base_url.rstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path or ""
    if not path.endswith("/audio/stream"):
        path = path.rstrip("/") + "/audio/stream"
    return urllib.parse.urlunparse((scheme, parsed.netloc, path, "", "", ""))


async def _connect_upstream(url: str, api_key: str):
    """Verbindung zum Streaming-Dienst; ein gesetzter API-Key geht als Bearer-Header mit."""
    from websockets.asyncio.client import connect

    return await connect(
        url,
        open_timeout=10,
        additional_headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
    )


def _settings() -> tuple[str, str]:
    db = get_db_service()
    settings = SettingsService(db.get_session())
    base_url = settings.get("STT_BASEURL", "http://parakeet:5001/v1")
    return base_url, settings.get("STT_API_KEY", "") or ""


def _is_final_or_error(msg) -> bool:
    if not isinstance(msg, str):
        return False
    try:
        parsed = json.loads(msg)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(parsed, dict) and ("final" in parsed or "error" in parsed)


async def relay(ws: WebSocket) -> None:
    """Relay fuer eine bereits angenommene Browser-Verbindung (Auth/Origin prueft der Router)."""
    base_url, api_key = await run_in_threadpool(_settings)
    try:
        upstream = await _connect_upstream(_build_stream_url(base_url), api_key)
    except Exception as e:
        logger.warning("STT-Stream: Verbindung zu Parakeet fehlgeschlagen: %s", e)
        try:
            await ws.send_text(json.dumps({"error": f"STT-Server nicht erreichbar: {e}"}))
        except Exception:
            pass
        return

    async def from_upstream():
        """Liest Nachrichten von Parakeet und reicht sie an den Browser weiter."""
        try:
            async for msg in upstream:
                if isinstance(msg, bytes):
                    await ws.send_bytes(msg)
                else:
                    await ws.send_text(msg)
                if _is_final_or_error(msg):
                    break
        except Exception as e:
            logger.warning("STT-Stream: Upstream-Relay-Fehler: %s", e)

    reader = asyncio.create_task(from_upstream(), name="stt-upstream")
    try:
        while not reader.done():
            receive = asyncio.create_task(ws.receive())
            done, _ = await asyncio.wait({receive, reader}, return_when=asyncio.FIRST_COMPLETED)
            if receive not in done:
                receive.cancel()
                break
            message = receive.result()
            if message["type"] == "websocket.disconnect":
                break
            try:
                if message.get("bytes") is not None:
                    await upstream.send(message["bytes"])
                elif message.get("text") is not None:
                    text = message["text"]
                    await upstream.send(text)
                    if "stop" in text.lower():
                        try:
                            await asyncio.wait_for(asyncio.shield(reader), _STOP_JOIN_TIMEOUT_S)
                        except TimeoutError:
                            pass
                        break
            except Exception as e:
                logger.warning("STT-Stream: Weiterleitung an Parakeet fehlgeschlagen: %s", e)
                break
    finally:
        reader.cancel()
        try:
            await upstream.close()
        except Exception:
            pass
