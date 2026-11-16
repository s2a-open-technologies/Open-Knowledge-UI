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
Settings Service für zentrale Konfiguration
Liest aus DB mit Fallback auf Environment
"""

import os
from typing import Any

from sqlalchemy.orm import Session as SQLAlchemySession

from app.core import crypto
from app.core.models import Setting


class SettingsService:
    """Zentraler Service für Settings Management"""

    # Verschluesselt in der Datenbank (app/crypto.py). Ohne den Passwort-Hash: er ist schon ein Hash.
    ENCRYPTED_KEYS_EXCLUDED = {"APP_PASSWORD_HASH"}

    SENSITIVE_KEYS = {
        "API_KEY",
        "QDRANT_API_KEY",
        "STT_API_KEY",
        "APP_PASSWORD_HASH",
        "SECRET_KEY",
        "PII_FILTER_API_KEY",
    }

    # Nur aus Umgebung/Standard gelesen, nie aus der Datenbank: ein alter Datenbankwert
    # ("localhost") wuerde sonst die Angaben aus docker-compose uebersteuern. Die
    # OCR-Engine ist Maschinen-Infrastruktur (Binaries auf der Maschine, nicht in der DB).
    ENV_ONLY_KEYS = {
        "DATA_DIR",
        "SUMMARIES_DIR",
        "QDRANT_HOST",
        "QDRANT_PORT",
        "QDRANT_API_KEY",
        "OCR_ENGINE",
        "OCR_LANGUAGES",
        "OCR_TESSERACT_CMD",
    }

    def __init__(self, db: SQLAlchemySession):
        self.db = db

    @classmethod
    def encrypted_keys(cls) -> set:
        return cls.SENSITIVE_KEYS - cls.ENCRYPTED_KEYS_EXCLUDED

    @classmethod
    def _plain(cls, key: str, value):
        return crypto.decrypt(value) if key in cls.encrypted_keys() else value

    @classmethod
    def _stored(cls, key: str, value):
        return crypto.encrypt(value) if key in cls.encrypted_keys() else value

    def get(self, key: str, default: Any = None) -> str | None:
        """
        Setting-Wert holen (Reihenfolge: DB → Env → Default)
        """
        # 1. Aus DB lesen (Infrastruktur-Werte nicht, siehe ENV_ONLY_KEYS)
        if key not in self.ENV_ONLY_KEYS:
            setting = self.db.query(Setting).filter_by(key=key).first()
            if setting and setting.value is not None:
                return self._plain(key, setting.value)

        # 2. Fallback auf Environment
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value

        # 3. Default
        return default

    def get_int(self, key: str, default: int = 0) -> int:
        """Setting als Integer"""
        val = self.get(key, default)
        if val is None:
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Setting als Float"""
        val = self.get(key, default)
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Setting als Boolean"""
        val = self.get(key, str(default).lower())
        if val is None:
            return default
        return str(val).lower() in ("true", "1", "yes", "on", "enabled")

    def set(self, key: str, value: str) -> None:
        """Setting in DB speichern"""
        setting = self.db.query(Setting).filter_by(key=key).first()
        if setting:
            setting.value = self._stored(key, value)
        else:
            # Neue Setting anlegen (Kategorie und Beschreibung stehen nur fuer die Standardwerte
            # in DatabaseService._init_settings)
            setting = Setting(
                key=key,
                value=self._stored(key, value),
                default_value=self._stored(key, os.getenv(key, "")),
                category="other",
                is_sensitive=key in self.SENSITIVE_KEYS,
                description="",
            )
            self.db.add(setting)

        self.db.commit()

    def set_batch(self, settings_dict: dict[str, str]) -> None:
        """Mehrere Settings auf einmal speichern"""
        for key, value in settings_dict.items():
            self.set(key, value)


class LiveSettingsService:
    """Wie ``SettingsService``, aber jeder Aufruf nutzt eine frische, kurze Session.

    Eine dauerhaft gehaltene Session cacht geladene Settings und liefert nach Aenderungen
    anderer Sessions (Admin-Seite) veraltete Werte.
    """

    def __getattr__(self, name):
        method = getattr(SettingsService, name)
        if not callable(method):
            raise AttributeError(name)

        def call(*args, **kwargs):
            from app.core.database_service import get_db_service

            with get_db_service().session_scope(commit=False) as session:
                return getattr(SettingsService(session), name)(*args, **kwargs)

        return call
