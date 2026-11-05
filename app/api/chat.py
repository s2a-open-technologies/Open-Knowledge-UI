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

"""Chat: WebSocket-Stream (``/ws/chat``), Abbruch, Sitzungen, Verlauf, Anhaenge und Spracheingabe.

Protokoll auf ``/ws/chat`` (JSON-Frames):

* Client -> Server: ``{"type": "message", "stream_id": "<uuid>", "message": ..., "session_id": ...,
  "knowledge_ids": [...], "attachment_ids": [...], "model": ...}`` und
  ``{"type": "stop", "stream_id": "<uuid>"}``.
* Server -> Client: ``{"type": "ready"}``, dann die Events des Chat-Kerns (``token``, ``reasoning``,
  ``tool_start``, ... ``done``/``error``, siehe ``app/chat/service.py``) und bei Abbruch
  ``{"type": "cancelled"}``; jeder Frame traegt die ``stream_id``.

Die ``stream_id`` erzeugt der Browser; damit ist auch der allererste Chat sofort stoppbar. Pro
Verbindung laeuft hoechstens ein Stream; schliesst der Socket, werden seine Streams abgebrochen.
"""

import asyncio
import contextvars
from contextlib import aclosing
from logging import getLogger

from fastapi import APIRouter, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.api._common import Upload
from app.api.errors import ApiError, validation_message
from app.api.schemas.chat import (
    AttachmentStarted,
    AttachmentStatus,
    ChatMessage,
    History,
    SessionDetail,
    SessionOut,
    StopRequest,
    SttConfig,
    Transcript,
)
from app.api.schemas.common import Success
from app.auth.auth import CurrentUser, current_user
from app.chat import service
from app.chat.streams import registry
from app.core import redis_bus, request_context, runtime
from app.core.csrf import same_origin
from app.core.database_service import get_db_service

router = APIRouter(tags=["chat"])
logger = getLogger(__name__)


def _iso(value) -> str | None:
    return value.isoformat() if value else None


def _session_dict(sess) -> dict:
    return {
        "id": sess.id,
        "title": sess.title,
        "created_at": _iso(sess.created_at),
        "updated_at": _iso(sess.updated_at),
    }


def _messages(session_id: str) -> list[dict]:
    return [
        {"role": msg.role, "content": msg.content, "created_at": _iso(msg.created_at)}
        for msg in get_db_service().get_messages(session_id)
    ]


@router.get("/api/sessions", response_model=list[SessionOut])
def get_sessions(user: CurrentUser):
    """List all sessions"""
    return [_session_dict(s) for s in get_db_service().list_sessions(user_id=user["id"])]


@router.post("/api/sessions", response_model=SessionOut)
def new_session(user: CurrentUser):
    """Create a new session"""
    return _session_dict(get_db_service().create_session(user_id=user["id"]))


@router.get("/api/sessions/{session_id}", response_model=SessionDetail)
def get_session_messages(session_id: str, user: CurrentUser):
    """Get session with messages"""
    sess = get_db_service().get_chat_session(session_id, user_id=user["id"])
    if not sess:
        raise ApiError(404, "Session nicht gefunden")
    return {**_session_dict(sess), "messages": _messages(session_id)}


@router.delete("/api/sessions/{session_id}", response_model=Success)
def delete_session_route(session_id: str, user: CurrentUser):
    """Delete a session"""
    if not get_db_service().delete_session(session_id, user_id=user["id"]):
        raise ApiError(404, "Session nicht gefunden")

    from app.chat.attachments import cleanup_session_attachments

    cleanup_session_attachments(session_id, runtime.state.client)
    return Success()


@router.get("/history/{session_id}", response_model=History)
def history(session_id: str, user: CurrentUser):
    if not get_db_service().get_chat_session(session_id, user_id=user["id"]):
        raise ApiError(404, "Session nicht gefunden")
    return {"history": _messages(session_id)}


# =============================================================================
# Chat-Stream (WebSocket) und Stopp
# =============================================================================


def _origin_ok(ws: WebSocket) -> bool:
    # WebSocket-Verbindungen unterliegen keiner CORS-Pruefung: nur Seiten derselben Adresse zulassen
    return same_origin(
        ws.headers.get("origin"), ws.headers.get("x-forwarded-host") or ws.headers.get("host")
    )


async def _authorized(ws: WebSocket) -> dict | None:
    """Angemeldeter Nutzer (Session-Cookie) von derselben Adresse, sonst Verbindung ablehnen."""
    user = await run_in_threadpool(current_user, ws)
    if user is None or not _origin_ok(ws):
        await ws.close(code=1008)
        return None
    return user


async def _prepare(user: dict, payload: dict) -> "service.ChatContext":
    """``prepare_chat`` im Threadpool. Wird der Stream dabei abgebrochen, laeuft der Thread zu Ende;
    eine dabei neu angelegte, leere Session wird danach wieder entfernt."""
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(
        None, contextvars.copy_context().run, service.prepare_chat, user, payload
    )
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:

        def _cleanup(done):
            if not done.cancelled() and done.exception() is None:
                try:
                    service.discard_if_empty(done.result())
                except Exception as e:
                    logger.warning(f"Leere Session nach Abbruch nicht entfernt: {e}")

        future.add_done_callback(_cleanup)
        raise


async def _run_stream(ws: WebSocket, user: dict, payload: dict, stream_id: str) -> None:
    async def send(event: dict) -> None:
        await ws.send_json({**event, "stream_id": stream_id})

    tokens = request_context.begin(None)
    try:
        await send({"type": "ready"})
        if await redis_bus.cancel_requested(stream_id, user["id"]):
            raise asyncio.CancelledError
        try:
            ctx = await _prepare(user, payload)
        except service.ChatError as e:
            await send({"error": str(e)})
            return
        async with aclosing(service.stream_chat(ctx)) as events:
            async for event in events:
                await send(event)
    except asyncio.CancelledError:
        try:
            await send({"type": "cancelled"})
        except Exception:
            pass
        raise
    except WebSocketDisconnect:
        pass
    except Exception as e:  # nie die Verbindung mit einem stillen Fehler stehen lassen
        logger.exception("Chat-Stream fehlgeschlagen")
        try:
            await send({"error": str(e)})
        except Exception:
            pass
    finally:
        request_context.end(tokens)


@router.websocket("/ws/chat")
async def chat_socket(ws: WebSocket):
    user = await _authorized(ws)
    if user is None:
        return
    await ws.accept()
    active: asyncio.Task | None = None
    try:
        while True:
            frame = await ws.receive_json()
            kind = frame.get("type") if isinstance(frame, dict) else None
            if kind == "message":
                try:
                    msg = ChatMessage.model_validate(frame)
                except ValidationError as e:
                    await ws.send_json(
                        {
                            "stream_id": frame.get("stream_id"),
                            "error": validation_message(e.errors()),
                        }
                    )
                    continue
                if active is not None and not active.done():
                    await ws.send_json(
                        {"stream_id": msg.stream_id, "error": "Es läuft bereits eine Antwort."}
                    )
                    continue
                task = asyncio.create_task(
                    _run_stream(ws, user, msg.model_dump(), msg.stream_id),
                    name=f"chat-{msg.stream_id[:8]}",
                )
                if not registry.add(msg.stream_id, task, user["id"], ws):
                    task.cancel()
                    await ws.send_json(
                        {"stream_id": msg.stream_id, "error": "Diese Stream-ID läuft bereits."}
                    )
                    continue
                active = task
            elif kind == "stop":
                stream_id = str(frame.get("stream_id") or "")
                if stream_id and not registry.cancel(stream_id, user["id"]):
                    await redis_bus.publish_cancel(stream_id, user["id"])
            else:
                await ws.send_json({"error": "Unbekannte Nachricht"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    except ValueError:  # kein JSON
        try:
            await ws.close(code=1003)
        except Exception:
            pass
    finally:
        registry.cancel_connection(ws)


@router.post("/chat/stop", response_model=Success)
async def stop_chat(user: CurrentUser, body: StopRequest | None = None):
    """Bricht einen laufenden Chat-Stream ab (Fallback, wenn der Socket nicht offen ist).

    Laeuft der Stream in diesem Prozess, endet er sofort; sonst geht der Stopp ueber Redis an alle
    Worker. Fremde oder unbekannte ``stream_id`` werden still ignoriert."""
    stream_id = (body.stream_id if body else "").strip()
    if stream_id and not registry.cancel(stream_id, user["id"]):
        await redis_bus.publish_cancel(stream_id, user["id"])
    return Success()


# =============================================================================
# Spracheingabe
# =============================================================================


@router.get("/api/stt/config", response_model=SttConfig)
def stt_config_route(_user: CurrentUser):
    """Ob und wie die Spracheingabe laeuft (``stream`` live, ``upload`` aufnehmen und senden)."""
    from app.chat import stt as stt_config

    return stt_config.client_config()


@router.post("/api/stt", response_model=Transcript)
def stt(_user: CurrentUser, audio: UploadFile | None = File(default=None)):
    """Wandelt eine Aufnahme in Text um (OpenAI-kompatibler Standard ``/audio/transcriptions``)."""
    from app.chat import stt as stt_config

    if audio is None or not audio.filename:
        raise ApiError(400, "Keine Audiodatei")
    data = audio.file.read(stt_config.MAX_AUDIO_BYTES + 1)
    if not data:
        raise ApiError(400, "Leere Audiodatei")
    if len(data) > stt_config.MAX_AUDIO_BYTES:
        raise ApiError(413, "Die Aufnahme ist zu groß (höchstens 25 MB)")

    cfg = stt_config.load()
    try:
        text = stt_config.transcribe(cfg, data, audio.filename)
    except stt_config.TranscriptionError as e:
        raise ApiError(502, str(e))
    return {"text": text}


@router.websocket("/ws/stt-stream")
async def stt_stream(ws: WebSocket):
    """Relay: Live-PCM-Audio vom Browser an Parakeet, Transkripte zurückstreamen."""
    from app.chat.stt_stream import relay

    if await _authorized(ws) is None:
        return
    await ws.accept()
    try:
        await relay(ws)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# =============================================================================
# Anhaenge
# =============================================================================


@router.post("/api/chat/attachments", response_model=AttachmentStarted, status_code=202)
def upload_chat_attachment(
    user: CurrentUser,
    file: UploadFile | None = File(default=None),
    session_id: str | None = Form(default=None),
):
    from app import resources_ready
    from app.chat.attachments import is_image, is_supported, start_attachment_ingestion

    if file is None or not file.filename:
        raise ApiError(400, "Keine Datei", success=False)
    if not is_supported(file.filename):
        raise ApiError(
            415,
            "Dateityp im Chat nicht unterstützt "
            "(PDF, DOCX, DOC, ODT, MD, Bilder). CSV/Excel bitte über Knowledge hochladen.",
            success=False,
        )
    if is_image(file.filename):
        from app.ingest.image_text import enabled

        if not enabled():
            raise ApiError(
                415,
                "Bilder im Chat sind abgeschaltet (Administration › Einlesen & Modelle).",
                success=False,
            )

    sparse = runtime.state.sparse_embeddings
    if not resources_ready() or not sparse:
        raise ApiError(503, "RAG-System wird noch initialisiert. Bitte kurz warten.", success=False)

    db = get_db_service()
    session_id = session_id or None
    if session_id and not db.get_chat_session(session_id, user_id=user["id"]):
        raise ApiError(404, "Ungültige Session", success=False)
    if not session_id:
        session_id = db.create_session(user_id=user["id"]).id

    attachment = start_attachment_ingestion(Upload(file), session_id, sparse, user["id"])
    return {"success": True, "session_id": session_id, **attachment}


@router.get("/api/chat/attachments/{attachment_id}", response_model=AttachmentStatus)
def chat_attachment_status(attachment_id: str, user: CurrentUser):
    db = get_db_service()
    att = db.get_attachment(attachment_id)
    # Fremde Anhaenge sind nicht auffindbar (404 statt 403).
    if not att or not db.get_chat_session(att.session_id, user_id=user["id"]):
        raise ApiError(404, "Anhang nicht gefunden")
    return {
        "id": att.id,
        "session_id": att.session_id,
        "filename": att.filename,
        "status": att.status,
        "error": att.error,
    }
