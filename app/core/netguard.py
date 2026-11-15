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

"""Schutz vor Zielen, die ein Admin nicht eintragen darf (SSRF auf Cloud-Metadaten)."""

import ipaddress
import socket
from urllib.parse import urlparse

LINK_LOCAL_MESSAGE = (
    "Link-Local-Adressen (z. B. 169.254.x.x) sind als Ziel nicht erlaubt"
)


class BlockedTargetError(ValueError):
    """Das Ziel ist gesperrt (Standard-Fehlertyp von :func:`reject_link_local`)."""


def reject_host_link_local(host: str, error: type[Exception] = BlockedTargetError) -> None:
    """Sperrt Link-Local-Hosts (169.254.0.0/16, fe80::/10), v. a. die Cloud-Metadaten-Adresse,
    auch nach DNS-Aufloesung. Private LAN-Adressen bleiben erlaubt: lokale Server (vLLM, Ollama,
    Datenbanken) sind der Normalfall.

    Raises:
        ``error`` (Standard :class:`BlockedTargetError`) bei gesperrtem Ziel.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return  # nicht aufloesbar: der Verbindungsversuch meldet den Fehler verstaendlich
    for info in infos:
        if ipaddress.ip_address(info[4][0].split("%")[0]).is_link_local:
            raise error(LINK_LOCAL_MESSAGE)


def reject_link_local(url: str, error: type[Exception] = BlockedTargetError) -> None:
    """Wie :func:`reject_host_link_local`, aber fuer eine URL (``ldap://`` und ``ldaps://``
    muessen vorher auf http(s) umgeschrieben werden)."""
    host = urlparse(url).hostname
    if not host:
        raise error("Ungültige Base-URL")
    reject_host_link_local(host, error)
