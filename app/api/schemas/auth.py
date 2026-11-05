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

"""Modelle fuer ``app/api/auth.py``."""

from pydantic import BaseModel

from app.api.schemas.common import UserOut


class Version(BaseModel):
    version: str


class Me(BaseModel):
    authenticated: bool
    user: UserOut


class LoginRequest(BaseModel):
    username: str = ""
    password: str = ""


class LoginResponse(BaseModel):
    success: bool
    user: UserOut


class OidcProvider(BaseModel):
    enabled: bool
    name: str


class Providers(BaseModel):
    ldap: bool
    oidc: OidcProvider


class ChangePasswordRequest(BaseModel):
    old_password: str = ""
    new_password: str = ""
