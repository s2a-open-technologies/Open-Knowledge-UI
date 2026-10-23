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
Websuche und PII-Filter aus den aktuellen Einstellungen.

Die Websuche (SearXNG) ist ein Werkzeug, das pro Modell zugeschaltet wird (Modelle >
Konfigurieren). Sie steht zur Verfuegung, wenn SearXNG eingerichtet ist UND entweder ein aktiver
PII-Filter (Suchanfragen verlassen das System und werden vorher maskiert) oder der Admin die
Suche ausdruecklich ohne Maskierung erlaubt hat (``SEARCH_ALLOW_UNMASKED``, Standard aus).
Alles wird bei jedem Aufruf frisch aus den Einstellungen gelesen.
"""

from logging import getLogger

from app.core.database_service import get_db_service
from app.core.settings_service import SettingsService

logger = getLogger(__name__)

_clients: dict[tuple, object] = {}


def _settings() -> dict:
    session = get_db_service().get_session()
    try:
        s = SettingsService(session)
        return {
            "searxng_url": (s.get("SEARCH_SEARXNG_URL") or "").strip(),
            "categories": s.get("SEARCH_SEARXNG_CATEGORIES", "general") or "general",
            "pii_url": (s.get("PII_FILTER_URL") or "").strip(),
            "pii_key": s.get("PII_FILTER_API_KEY") or "",
            "pii_enabled": s.get_bool("PII_FILTER_ENABLED", True),
            "pii_timeout": max(
                2.0, min(120.0, s.get_float("PII_FILTER_TIMEOUT_S", 10.0))
            ),
            "pii_max_chars": max(2000, s.get_int("PII_FILTER_MAX_CHARS", 12000)),
            "allow_unmasked": s.get_bool("SEARCH_ALLOW_UNMASKED", False),
        }
    finally:
        session.close()


def pii_client():
    """PII-Filter-Client oder ``None``, wenn nicht eingerichtet bzw. abgeschaltet."""
    cfg = _settings()
    if not (cfg["pii_url"] and cfg["pii_enabled"]):
        return None
    key = (cfg["pii_url"], cfg["pii_key"], cfg["pii_timeout"], cfg["pii_max_chars"])
    if key not in _clients:
        from app.utils.pii_filter_client import PIIFilterClient

        _clients[key] = PIIFilterClient(
            full_url=cfg["pii_url"],
            api_key=cfg["pii_key"] or None,
            timeout=cfg["pii_timeout"],
            max_chars=cfg["pii_max_chars"],
        )
    return _clients[key]


def _masking_active(cfg: dict) -> bool:
    return bool(cfg["pii_url"] and cfg["pii_enabled"])


def web_search_available() -> bool:
    """SearXNG ist eingerichtet und die Suche ist maskiert oder ausdruecklich unmaskiert erlaubt."""
    cfg = _settings()
    return bool(cfg["searxng_url"] and (_masking_active(cfg) or cfg["allow_unmasked"]))


def web_search_masked() -> bool:
    """Werden Suchanfragen vor dem Versand durch den PII-Filter maskiert?"""
    cfg = _settings()
    return bool(cfg["searxng_url"] and _masking_active(cfg))


def build_web_search_tool():
    """Das Websuche-Werkzeug oder ``None``, wenn SearXNG fehlt bzw. weder Maskierung noch Freigabe da ist."""
    cfg = _settings()
    if not (cfg["searxng_url"] and (_masking_active(cfg) or cfg["allow_unmasked"])):
        return None
    from app.tools.web_search_tool import create_web_search_tool

    return create_web_search_tool(
        searxng_url=cfg["searxng_url"],
        default_categories=cfg["categories"],
        pii_filter_client=pii_client(),  # None ohne aktiven PII-Filter: Anfrage geht unmaskiert raus
    )
