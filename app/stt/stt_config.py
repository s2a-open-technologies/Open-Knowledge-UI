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
Spracheingabe: Einstellungen, Betriebsart und Transkription.

Zwei Wege zum selben Ziel (Text im Chatfeld):

* ``stream``: Live-Streaming ueber das Parakeet-Protokoll (``<base>/audio/stream``), der Text
  erscheint waehrend des Sprechens.
* ``upload``: Aufnehmen und beim Stoppen an den OpenAI-Standard ``<base>/audio/transcriptions``
  schicken. Das koennen Whisper, vLLM, speaches, OpenAI, Groq u. a.

``STT_MODE=auto`` (Standard) waehlt selbst: nimmt ``<base>/audio/stream`` einen WebSocket an,
wird gestreamt, sonst hochgeladen. Das Ergebnis wird je Adresse kurz gemerkt.
"""

import io
import math
import struct
import threading
import time
import urllib.parse
import wave
from dataclasses import dataclass
from logging import getLogger

from app.core.database_service import get_db_service
from app.core.settings_service import SettingsService

logger = getLogger(__name__)

MODES = ("auto", "stream", "upload")
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # Grenze der OpenAI-Schnittstelle
PROBE_TIMEOUT_S = 3.0
PROBE_TTL_S = 300.0

_probe_cache: dict[str, tuple[float, bool]] = {}
_probe_lock = threading.Lock()


@dataclass(frozen=True)
class SttConfig:
    base_url: str
    model: str
    api_key: str
    mode: str  # Einstellung: auto | stream | upload


def load() -> SttConfig:
    session = get_db_service().get_session()
    try:
        s = SettingsService(session)
        mode = (s.get("STT_MODE", "auto") or "auto").strip().lower()
        return SttConfig(
            base_url=(s.get("STT_BASEURL", "") or "").strip().rstrip("/"),
            model=(s.get("STT_MODEL", "") or "").strip(),
            api_key=s.get("STT_API_KEY", "") or "",
            mode=mode if mode in MODES else "auto",
        )
    finally:
        session.close()


def stream_url(base_url: str) -> str:
    """``http://host:5001/v1`` -> ``ws://host:5001/v1/audio/stream``."""
    parsed = urllib.parse.urlparse(base_url.rstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path or ""
    if not path.endswith("/audio/stream"):
        path = path.rstrip("/") + "/audio/stream"
    return urllib.parse.urlunparse((scheme, parsed.netloc, path, "", "", ""))


def auth_headers(api_key: str) -> list[str]:
    return [f"Authorization: Bearer {api_key}"] if api_key else []


def probe_stream(base_url: str, api_key: str = "") -> bool:
    """Nimmt ``<base>/audio/stream`` einen WebSocket an? (kurzer Verbindungsversuch)"""
    import websocket

    try:
        ws = websocket.create_connection(
            stream_url(base_url), timeout=PROBE_TIMEOUT_S, header=auth_headers(api_key)
        )
    except Exception:
        return False
    try:
        ws.close()
    except Exception:
        pass
    return True


def streaming_available(cfg: SttConfig) -> bool:
    """Gemerktes Ergebnis von ``probe_stream`` je Adresse (5 Minuten)."""
    key = cfg.base_url
    now = time.monotonic()
    with _probe_lock:
        hit = _probe_cache.get(key)
        if hit and now - hit[0] < PROBE_TTL_S:
            return hit[1]
    result = probe_stream(cfg.base_url, cfg.api_key)
    with _probe_lock:
        _probe_cache[key] = (now, result)
    return result


def clear_probe_cache() -> None:
    with _probe_lock:
        _probe_cache.clear()


def resolve_mode(cfg: SttConfig) -> str:
    """``stream`` oder ``upload`` (aufgeloeste Betriebsart)."""
    if cfg.mode in ("stream", "upload"):
        return cfg.mode
    return "stream" if streaming_available(cfg) else "upload"


def client_config() -> dict:
    """Was der Browser wissen darf: ob und wie die Spracheingabe laeuft (keine Geheimnisse)."""
    cfg = load()
    if not cfg.base_url:
        return {"available": False, "mode": "upload"}
    return {"available": True, "mode": resolve_mode(cfg)}


class TranscriptionError(Exception):
    """Transkription fehlgeschlagen (lesbare Meldung ohne interne Details)."""


def transcribe(
    cfg: SttConfig,
    audio: bytes,
    filename: str = "recording.webm",
    timeout: float = 120.0,
) -> str:
    """Schickt Audio an ``<base>/audio/transcriptions`` (OpenAI-Standard) und liefert den Text."""
    import openai

    if not cfg.base_url:
        raise TranscriptionError("Die Spracheingabe ist nicht eingerichtet")
    from app.llm.registry import RemoteModelsError, _reject_link_local

    try:
        _reject_link_local(cfg.base_url)
    except RemoteModelsError as e:
        raise TranscriptionError(str(e)) from None
    file_obj = io.BytesIO(audio)
    file_obj.name = filename or "recording.webm"
    kwargs = {"file": file_obj}
    if cfg.model:
        kwargs["model"] = cfg.model
    else:
        kwargs["model"] = (
            "whisper-1"  # Pflichtfeld des Standards, viele Server ignorieren es
        )
    client = openai.OpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key or "none",
        timeout=timeout,
        max_retries=0,
    )
    try:
        result = client.audio.transcriptions.create(**kwargs)
    except openai.APIStatusError as e:
        logger.warning("STT: HTTP %s von %s", e.status_code, cfg.base_url)
        raise TranscriptionError(
            f"Der Spracherkennungs-Dienst antwortete mit HTTP {e.status_code}"
        ) from None
    except openai.APIConnectionError:
        logger.warning("STT: %s nicht erreichbar", cfg.base_url)
        raise TranscriptionError(
            "Der Spracherkennungs-Dienst ist nicht erreichbar"
        ) from None
    except Exception as e:
        logger.warning("STT: Transkription fehlgeschlagen: %s", type(e).__name__)
        raise TranscriptionError("Die Transkription ist fehlgeschlagen") from None
    return (getattr(result, "text", "") or "").strip()


def silent_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    """Kurze Stille als WAV (16 kHz, mono) fuer den Verbindungstest."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * math.ceil(seconds * rate))
    return buf.getvalue()
