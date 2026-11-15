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


# Metadaten-Dienste ausserhalb von 169.254.0.0/16: AWS (IPv6) und Alibaba Cloud
_EXTRA_BLOCKED = [
    ipaddress.ip_network(n) for n in ("fd00:ec2::254/128", "100.100.100.200/32")
]
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


def _is_blocked(address: str) -> bool:
    """Link-Local oder bekannte Metadaten-Adresse, auch als IPv4-gemappte/NAT64-IPv6-Form."""
    ip = ipaddress.ip_address(address.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        elif ip in _NAT64:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip.is_link_local or any(ip in net for net in _EXTRA_BLOCKED)


def reject_host_link_local(host: str, error: type[Exception] = BlockedTargetError) -> None:
    """Sperrt Link-Local- und Metadaten-Hosts (169.254.0.0/16, fe80::/10, ...), v. a. die
    Cloud-Metadaten-Adresse, auch nach DNS-Aufloesung. Private LAN-Adressen bleiben erlaubt:
    lokale Server (vLLM, Ollama, Datenbanken) sind der Normalfall.

    Grenze: Die Pruefung gilt zum Zeitpunkt des Aufrufs. Ein Name, der danach auf eine
    gesperrte Adresse umgebogen wird (DNS-Rebinding), wird hier nicht erkannt.

    Raises:
        ``error`` (Standard :class:`BlockedTargetError`) bei gesperrtem Ziel.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return  # nicht aufloesbar: der Verbindungsversuch meldet den Fehler verstaendlich
    if any(_is_blocked(info[4][0]) for info in infos):
        raise error(LINK_LOCAL_MESSAGE)


def reject_link_local(url: str, error: type[Exception] = BlockedTargetError) -> None:
    """Wie :func:`reject_host_link_local`, aber fuer eine URL (``ldap://`` und ``ldaps://``
    muessen vorher auf http(s) umgeschrieben werden)."""
    host = urlparse(url).hostname
    if not host:
        raise error("Ungültige Base-URL")
    reject_host_link_local(host, error)
