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

"""Modelle fuer ``app/api/users.py`` (Benutzer, Gruppen, Freigaben)."""

from pydantic import BaseModel

from app.api.schemas.auth import ChangePasswordRequest  # noqa: F401 (gleiche Form)


class AdminUser(BaseModel):
    id: int
    username: str
    display_name: str
    role: str
    is_active: bool
    group_ids: list[int] = []
    auth_source: str
    created_at: str | None = None


class UserCreate(BaseModel):
    username: str = ""
    display_name: str = ""
    password: str = ""
    role: str = "user"


class UserUpdate(BaseModel):
    """Nur gesetzte Felder werden geaendert."""

    display_name: str | None = None
    role: str | None = None
    is_active: bool | None = None
    password: str | None = None


class Group(BaseModel):
    id: int
    name: str
    description: str = ""
    member_ids: list[int] = []


class GroupCreate(BaseModel):
    name: str = ""
    description: str = ""


class GroupUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class GroupMembers(BaseModel):
    user_ids: list[int] = []


class PrincipalUser(BaseModel):
    id: int
    name: str
    username: str


class PrincipalGroup(BaseModel):
    id: int
    name: str


class Principals(BaseModel):
    users: list[PrincipalUser]
    groups: list[PrincipalGroup]


class Grant(BaseModel):
    principal_type: str
    principal_id: int
    access: str = "read"
    name: str | None = None


class GrantIn(BaseModel):
    principal_type: str = ""
    principal_id: int | str | None = None
    access: str = "read"


class Access(BaseModel):
    is_public: bool
    grants: list[Grant] = []


class AccessUpdate(BaseModel):
    is_public: bool = False
    grants: list[GrantIn] = []


class ModelAccessUpdate(AccessUpdate):
    ref: str = ""
