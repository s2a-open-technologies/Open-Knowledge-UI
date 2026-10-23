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

"""HTTP-Client zum Tool-Worker. Meldet Ausfaelle ohne interne Adressen."""

import threading

import requests

from app.toolworker import config

NETWORK_MARGIN_S = 15


class ToolWorkerUnavailable(Exception):
    """Der Worker ist nicht erreichbar oder lehnt die Zugangsdaten ab (Text ist fuer Nutzer gedacht)."""


def _http_transport(base_url: str, token: str):
    def send(method: str, path: str, payload: dict | None, timeout: float):
        r = requests.request(
            method,
            base_url + path,
            json=payload,
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=timeout,
        )
        try:
            body = r.json()
        except ValueError:
            body = {}
        return r.status_code, body

    return send


class ToolWorkerClient:
    def __init__(self, url: str, token: str, transport=None):
        self._send = transport or _http_transport(url.rstrip("/"), token)

    def _call(self, method, path, payload, timeout):
        try:
            return self._send(method, path, payload, timeout)
        except Exception:  # noqa: BLE001 - Verbindung, Timeout, DNS: nie Details nach aussen
            raise ToolWorkerUnavailable(
                "Der Tool-Worker ist nicht erreichbar."
            ) from None

    def run_pandas(
        self,
        file: str,
        code: str,
        *,
        timeout_s: float,
        memory_mb: int,
        max_output_chars: int,
    ) -> str:
        status, body = self._call(
            "POST",
            "/v1/pandas",
            {
                "file": file,
                "code": code,
                "limits": {
                    "timeout_s": timeout_s,
                    "memory_mb": memory_mb,
                    "max_output_chars": max_output_chars,
                },
            },
            timeout_s + NETWORK_MARGIN_S,
        )
        if status == 401:
            raise ToolWorkerUnavailable(
                "Der Tool-Worker lehnt die Zugangsdaten ab (TOOL_WORKER_TOKEN prüfen)."
            )
        if status == 404:
            return f"FEHLER: Datei '{file}' ist für die Auswertung nicht verfügbar."
        if status != 200 or not isinstance(body, dict):
            raise ToolWorkerUnavailable("Der Tool-Worker meldet einen Fehler.")
        return (
            body.get("output")
            if body.get("ok")
            else (body.get("error") or "FEHLER: unbekannt")
        )

    def sql(self, action: str, payload: dict, timeout: float) -> dict:
        status, body = self._call("POST", f"/v1/sql/{action}", payload, timeout)
        if status == 401:
            raise ToolWorkerUnavailable(
                "Der SQL-Dienst lehnt die Zugangsdaten ab (TOOL_WORKER_TOKEN prüfen)."
            )
        if status != 200 or not isinstance(body, dict):
            raise ToolWorkerUnavailable("Der SQL-Dienst ist nicht verfügbar.")
        return body

    def ping(self) -> dict:
        try:
            status, body = self._send("GET", "/health", None, 3)
        except Exception:  # noqa: BLE001
            return {"reachable": False, "version": "", "error": "nicht erreichbar"}
        if status == 200 and isinstance(body, dict):
            return {
                "reachable": True,
                "version": str(body.get("version", "")),
                "error": "",
            }
        return {"reachable": False, "version": "", "error": f"HTTP {status}"}


_overrides: dict[str, ToolWorkerClient] = {}
_lock = threading.Lock()


def set_override(client: ToolWorkerClient | None, role: str = "pandas") -> None:
    with _lock:
        if client is None:
            _overrides.pop(role, None)
        else:
            _overrides[role] = client


def get_client(role: str = "pandas") -> ToolWorkerClient:
    with _lock:
        if role in _overrides:
            return _overrides[role]
    url = config.load_sql().url if role == "sql" else config.load().url
    return ToolWorkerClient(url, config.token())
