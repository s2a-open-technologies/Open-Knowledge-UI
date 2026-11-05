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

"""Modelle fuer ``app/api/chat.py`` (REST-Teil; der Chat selbst laeuft ueber ``/ws/chat``)."""

from typing import Any

from pydantic import BaseModel


class SessionOut(BaseModel):
    id: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: str | None = None


class SessionDetail(SessionOut):
    messages: list[MessageOut]


class History(BaseModel):
    history: list[MessageOut]


class ChatMessage(BaseModel):
    """Frame ``{"type": "message", ...}`` auf ``/ws/chat``."""

    stream_id: str
    message: str = ""
    session_id: str | None = None
    # None = alle Knowledge-Bereiche; Liste = nur diese (leer = keiner)
    knowledge_ids: list[Any] | None = None
    attachment_ids: list[Any] = []
    model: str | None = None


class StopRequest(BaseModel):
    stream_id: str = ""


class SttConfig(BaseModel):
    available: bool
    mode: str


class Transcript(BaseModel):
    text: str


class AttachmentStarted(BaseModel):
    success: bool
    session_id: str
    id: str
    filename: str
    status: str


class AttachmentStatus(BaseModel):
    id: str
    session_id: str
    filename: str
    status: str
    error: str | None = None
