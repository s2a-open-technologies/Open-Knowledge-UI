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

"""Verbindungstests fuer Admin › Tools: Spracheingabe, Websuche, PII-Filter."""

import time

import requests

from app.core.settings_service import SettingsService


def check_stt(settings: SettingsService) -> dict:
    """Spracheingabe testen: echte Transkription einer Sekunde Stille (beweist Adresse, Modell
    und Key) und, ob zusaetzlich Live-Streaming (``/audio/stream``) angeboten wird."""
    from app.chat import stt as stt_config

    cfg = stt_config.SttConfig(
        base_url=(settings.get("STT_BASEURL") or "").strip().rstrip("/"),
        model=(settings.get("STT_MODEL") or "").strip(),
        api_key=settings.get("STT_API_KEY") or "",
        mode="auto",
    )
    if not cfg.base_url:
        return {"success": False, "error": "Es ist keine Adresse eingetragen"}

    upload_ok, upload_error = True, ""
    try:
        stt_config.transcribe(cfg, stt_config.silent_wav(), "test.wav", timeout=30)
    except stt_config.TranscriptionError as e:
        upload_ok, upload_error = False, str(e)
    stream_ok = stt_config.probe_stream(cfg.base_url, cfg.api_key)

    if not upload_ok and not stream_ok:
        return {"success": False, "error": upload_error or "Der Dienst antwortet nicht"}
    parts = []
    parts.append(
        "Aufnehmen & senden: ok" if upload_ok else f"Aufnehmen & senden: {upload_error}"
    )
    parts.append(
        "Live-Streaming: verfügbar" if stream_ok else "Live-Streaming: nicht angeboten"
    )
    mode = "stream" if stream_ok else "upload"
    return {
        "success": upload_ok or stream_ok,
        "message": f"Automatisch gewählt: {'Live-Streaming' if mode == 'stream' else 'Aufnehmen & senden'}",
        "details": " · ".join(parts),
    }


def check_search(settings: SettingsService) -> dict:
    """SearXNG Metasuchmaschine testen."""
    url = settings.get("SEARCH_SEARXNG_URL") or ""
    if not url:
        return {"success": False, "error": "SEARCH_SEARXNG_URL ist nicht gesetzt"}

    # health-Endpoint oder /search mit leerer Query probieren
    candidates = [
        url.rstrip("/") + "/healthz",
        url.rstrip("/") + "/search?q=test&format=json",
    ]

    last_error = ""
    for test_url in candidates:
        try:
            resp = requests.get(test_url, timeout=10)
            if resp.status_code == 200:
                # Bei /search prüfen ob JSON zurückkommt
                if test_url.endswith("format=json"):
                    try:
                        data = resp.json()
                        results = data.get("results", [])
                        return {
                            "success": True,
                            "message": (
                                f"SearXNG erreichbar. "
                                f"{len(results)} Test-Ergebnis(se) zurückgegeben."
                            ),
                        }
                    except ValueError:
                        last_error = "Antwort ist kein valides JSON"
                        continue
                return {"success": True, "message": "SearXNG erreichbar."}
            else:
                last_error = f"HTTP {resp.status_code} von {test_url}"
        except requests.exceptions.Timeout:
            last_error = f"Timeout (10s) bei {test_url}"
        except requests.exceptions.RequestException as e:
            last_error = f"Verbindung fehlgeschlagen: {e}"

    return {"success": False, "error": last_error or "SearXNG nicht erreichbar"}


PII_LONG_CHARS = (
    8000  # Laenge des zweiten Testtexts (ein langer Verlauf, ein grosses Ergebnis)
)


PII_TEST_TEXT = (
    "Bitte schreibe Max Mustermann (max.mustermann@example.com) eine kurze Antwort."
)


def check_pii_filter(settings: SettingsService) -> dict:
    """PII-Filter testen: echte Maskierung eines Beispieltexts mit erfundenen Daten und Pruefung
    des Vertrags (docs/sanitizer-api.md): Text veraendert, Mapping erkannt, Demaskierung stimmt."""
    from app.chat.pii_client import PIIFilterClient, mapping_pairs, parse_response
    from app.chat.pii_session import timeout_for

    url = settings.get("PII_FILTER_URL") or ""
    api_key = settings.get("PII_FILTER_API_KEY") or ""

    if not url:
        return {"success": False, "error": "Es ist keine Adresse eingetragen"}

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    started = time.monotonic()
    try:
        resp = requests.post(
            url, json={"text": PII_TEST_TEXT}, headers=headers, timeout=10
        )
        short_ms = int((time.monotonic() - started) * 1000)
    except requests.exceptions.Timeout:
        return {"success": False, "error": f"Timeout (10s) bei {url}"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Verbindung fehlgeschlagen: {e}"}

    if resp.status_code in (401, 403):
        return {
            "success": False,
            "error": f"API-Key ungültig oder fehlt (HTTP {resp.status_code})",
            "details": resp.text[:300],
        }
    if resp.status_code != 200:
        return {
            "success": False,
            "error": f"HTTP {resp.status_code} von {url}",
            "details": resp.text[:300],
        }
    try:
        data = resp.json()
    except ValueError:
        return {
            "success": False,
            "error": "Die Antwort ist kein JSON (siehe Anforderungen an den Dienst)",
        }
    if not isinstance(data, dict):
        return {
            "success": False,
            "error": "Die Antwort muss ein JSON-Objekt mit „filtered“ sein",
        }

    _original, filtered, mapping = parse_response(data, PII_TEST_TEXT)
    pairs = mapping_pairs(mapping)
    facts = []
    problems = []
    if filtered == PII_TEST_TEXT:
        problems.append(
            "Der Beispieltext kam unverändert zurück: der Dienst hat nichts erkannt oder das Feld „filtered“ fehlt"
        )
    else:
        facts.append("Text wird maskiert")
    if pairs:
        facts.append(f"{len(pairs)} Wert(e) im Mapping")
        if PIIFilterClient("-").demask(filtered, mapping) == PII_TEST_TEXT:
            facts.append("Demaskierung stimmt")
        else:
            problems.append(
                "Die Demaskierung ergibt nicht den Originaltext (Platzhalter nicht eindeutig oder Mapping unvollständig)"
            )
    elif filtered != PII_TEST_TEXT:
        problems.append(
            "Kein Mapping in der Antwort: Antworten des Modells können nicht demaskiert werden (die Websuche funktioniert trotzdem)"
        )
    # Ein kurzer Satz sagt wenig: ein langer Text zeigt, ob der Dienst auch bei grossen Verlaeufen und Ergebnissen
    # rechtzeitig antwortet (dort entstanden die Zeitueberschreitungen, obwohl der Kurztest bestand).
    facts.append(f"kurzer Text {short_ms} ms")
    timeout_s = settings.get_float("PII_FILTER_TIMEOUT_S", 10.0)
    try:
        long_text = (PII_TEST_TEXT + " ") * (
            PII_LONG_CHARS // (len(PII_TEST_TEXT) + 1) + 1
        )
        started = time.monotonic()
        long_resp = requests.post(
            url, json={"text": long_text[:PII_LONG_CHARS]}, headers=headers, timeout=120
        )
        long_ms = int((time.monotonic() - started) * 1000)
        allowed = timeout_for(PII_LONG_CHARS, timeout_s)
        if long_resp.status_code != 200:
            problems.append(
                f"Langer Text ({PII_LONG_CHARS} Zeichen): HTTP {long_resp.status_code}"
            )
        elif long_ms / 1000 > allowed:
            problems.append(
                f"Langer Text ({PII_LONG_CHARS} Zeichen) brauchte {long_ms} ms, erlaubt sind {allowed:g} s: "
                "Zeitlimit erhöhen oder den Dienst entlasten"
            )
        else:
            facts.append(f"langer Text ({PII_LONG_CHARS} Zeichen) {long_ms} ms")
    except requests.exceptions.RequestException as e:
        problems.append(
            f"Langer Text ({PII_LONG_CHARS} Zeichen) scheiterte: {type(e).__name__}"
        )
    result = {"details": " · ".join(facts + problems)}
    if filtered == PII_TEST_TEXT:
        return {"success": False, "error": problems[0], **result}
    return {
        "success": True,
        "message": "PII-Filter erreichbar"
        + (" mit Hinweisen" if problems else " und vertragsgemäß"),
        **result,
    }
