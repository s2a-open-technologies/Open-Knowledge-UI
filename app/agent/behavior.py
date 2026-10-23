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
Standard-Anweisungen fuer alle Modelle (Verhalten, Rolle, Ton).

Der feste Basis-Prompt (``DocumentAgent.SYSTEM_PROMPT``) enthaelt nur, was Werkzeugaufrufe und die
Quellenanzeige sicher macht. Alles Uebrige, also Rolle, Sprache, Stil und wie Uebersichts- oder
Zusammenfassungsfragen beantwortet werden, steht hier in der Datenbank (``settings``) und wird auf
der Seite Modelle eingestellt. Pro Modell kommt der Zusatz (Brand Voice) dazu. Hier stehen nur die Samen.
"""

from app.core.database_service import get_db_service
from app.core.models import Setting
from app.core.settings_service import SettingsService

KEY = "AGENT_BEHAVIOR_PROMPT"
MAX_CHARS = 8000

DEFAULT_BEHAVIOR = (
    "Du bist ein sorgfältiger Dokumentenanalyst. Du wertest Daten aus PDF-, Word-, Excel- und CSV-Dokumenten "
    "sachlich und genau aus und antwortest immer auf Deutsch.\n\n"
    '- **Zusammenfassungen**: Soll das Gespräch zusammengefasst werden ("unser Gespräch", "dieser Chat"), '
    "nutze NUR den Gesprächsverlauf, keine Dokumentsuche und keine neuen Quellen. Soll ein Dokument oder das Wissen "
    "zusammengefasst werden, starte direkt mit der Suche, ohne Vorbemerkungen. Ist unklar, was gemeint ist, "
    "frage in einem Satz nach.\n"
    '- **Gezielte Fragen** ("Ist X da?", "Gibt es etwas zu Y?"): Antworte knapp mit Ja oder Nein und nenne nur die '
    "passenden Fundstellen. Zähle nie den übrigen Inhalt auf.\n"
    '- **Übersichtsfragen**: Nur wenn der Nutzer ausdrücklich eine Übersicht oder Liste verlangt ("was gibt es alles", '
    '"Aufstellung", "Überblick"), nenne alle verfügbaren Quellen.\n'
    "- **Ablegen**: Bittet der Nutzer darum, etwas abzulegen oder zu speichern, lege genau den gewünschten Inhalt ab "
    "(Gespräch ODER Dokumentanalyse, nicht vermischen).\n"
    "- **Stil**: Formell, knapp und ohne Floskeln. Beschreibe nicht, was du tun wirst, sondern tue es."
)

DESCRIPTION = "Standard-Anweisungen fuer alle Modelle (Rolle, Sprache, Stil, Verhalten bei Uebersichts- und Zusammenfassungsfragen)"
SETTINGS = [(KEY, DEFAULT_BEHAVIOR, DESCRIPTION)]


class BehaviorConfigError(ValueError):
    pass


def load() -> dict:
    """Aktueller Text und Standard (frisch aus der Datenbank, Samen als Rueckfall)."""
    session = get_db_service().get_session()
    try:
        stored = session.query(Setting).filter_by(key=KEY).first()
        text = (SettingsService(session).get(KEY) or "").strip()
        default = (
            (stored.default_value if stored else "") or ""
        ).strip() or DEFAULT_BEHAVIOR
        return {"prompt": text, "default_prompt": default}
    finally:
        session.close()


def current() -> str:
    """Text fuer den Agenten; ein geleertes Feld heisst bewusst: keine Standard-Anweisungen."""
    return load()["prompt"]


def save(data: dict) -> None:
    text = str(data.get("prompt") or "").strip()
    if len(text) > MAX_CHARS:
        raise BehaviorConfigError(
            f"Der Text ist zu lang (höchstens {MAX_CHARS} Zeichen)"
        )
    session = get_db_service().get_session()
    try:
        SettingsService(session).set_batch({KEY: text})
    finally:
        session.close()
