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

    CATEGORIES = {
        "llm": [
            "CHAT_MODEL",
            "CHAT_BASEURL",
            "CHAT_TEMPERATURE",
            "CHAT_TOP_P",
            "CHAT_TOP_K",
            "CHAT_MAX_TOKENS",
            "CHAT_NUM_CTX",
            "CHAT_THINKING",
            "VISION_MODEL",
            "VISION_BASEURL",
            "API_KEY",
        ],
        "stt": [
            "STT_MODEL",
            "STT_BASEURL",
            "STT_API_KEY",
            "STT_MODE",
        ],
        "embedding": [
            "EMBEDDING_SOURCE",
            "EMBEDDING_MODEL",
            "EMBEDDING_ENDPOINT",
            "EMBEDDING_DIMENSION",
        ],
        "retriever": [
            "RETRIEVER_K",
            "RETRIEVER_FETCH_K",
            "RETRIEVER_LAMBDA",
            "RETRIEVER_SCORE_THRESHOLD",
        ],
        "storage": ["DATA_DIR", "SUMMARIES_DIR"],
        "qdrant": ["QDRANT_HOST", "QDRANT_PORT", "QDRANT_API_KEY"],
        "auth": [
            "APP_PASSWORD_HASH",
            "SECRET_KEY",
        ],
        "search": [
            "SEARCH_SEARXNG_URL",
            "SEARCH_SEARXNG_CATEGORIES",
            "SEARCH_SEARXNG_ENABLED",
        ],
        "tools": [
            "TOOL_WORKER_URL",
            "PANDAS_TIMEOUT_S",
            "PANDAS_MAX_MEMORY_MB",
            "PANDAS_MAX_OUTPUT_CHARS",
            "SQL_WORKER_URL",
            "SQL_ROW_LIMIT",
            "SQL_TIMEOUT_S",
            "SQL_MAX_CELL_CHARS",
            "SQL_PROMPT",
            "SQL_VALUES_MAX_DISTINCT",
            "SQL_VALUES_MAX_RESULTS",
            "SQL_VALUES_HINT",
        ],
        "pii_filter": [
            "PII_FILTER_URL",
            "PII_FILTER_API_KEY",
            "PII_FILTER_ENABLED",
            "PII_FILTER_TIMEOUT_S",
            "PII_FILTER_MAX_CHARS",
        ],
        "ingest": [
            "INGEST_CHUNK_SIZE",
            "INGEST_CHUNK_OVERLAP",
            "INGEST_CONTEXT_ENRICHMENT",
            "INGEST_KEEP_ORIGINAL",
            "INGEST_INDEX_ENABLED",
            "BM25_LANGUAGE",
            "BM25_LANGUAGE",
            "INGEST_VISION_ENABLED",
            "INGEST_TABLE_VISION",
            "INGEST_CHAT_IMAGES",
        ],
        "ui": [
            "UI_THEME",
        ],
    }

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

    # Veraltete Settings, die nicht mehr im Admin-Panel erscheinen und nicht aus
    # der env re-importiert werden. Bestehende DB-Zeilen werden beim Init geloescht.
    # APP_PASSWORD_HASH wird separat vom Passwort-Formular verwaltet.
    DEPRECATED_KEYS = {"MODEL", "BASEURL", "TEMPERATURE", "TOP_P", "CHAT_DB_PATH"}
    # RETRIEVER_*: nur noch Startwerte fuer das Standard-Retrieval-Profil (Admin > Retrieval).
    UI_EXCLUDED_KEYS = {
        "APP_PASSWORD_HASH",
        "DEFAULT_MODEL",
        "RETRIEVER_K",
        "RETRIEVER_FETCH_K",
        "RETRIEVER_LAMBDA",
        "RETRIEVER_SCORE_THRESHOLD",
        # Eigene Admin-Seite "Einlesen & Modelle"
        "EMBEDDING_SOURCE",
        "EMBEDDING_MODEL",
        "EMBEDDING_ENDPOINT",
        "EMBEDDING_DIMENSION",
        "EMBEDDING_SPEC",
        "RERANKER_SPEC",
        "INGEST_CHUNK_SIZE",
        "INGEST_CHUNK_OVERLAP",
        "INGEST_CONTEXT_ENRICHMENT",
        "INGEST_KEEP_ORIGINAL",
        "INGEST_INDEX_ENABLED",
        "INGEST_VISION_ENABLED",
        "VISION_MODEL_REF",
        "VISION_PROMPT",
        "CONTEXT_PROMPT",
        "INGEST_TABLE_VISION",
        "TABLES_PROMPT",
        "TABLES_MODEL_REF",
        "TABLES_TEMPERATURE",
        "TABLES_MAX_TOKENS",
        "TABLES_MAX_ROW_DEVIATION",
        "INGEST_CHAT_IMAGES",
        "IMAGES_PROMPT",
        "IMAGES_MODEL_REF",
        "IMAGES_TEMPERATURE",
        "IMAGES_MAX_TOKENS",
        "CONTEXT_EXCERPT_CHARS",
        "VISION_TEMPERATURE",
        "VISION_MAX_TOKENS",
        "CONTEXT_MODEL_REF",
        "CONTEXT_TEMPERATURE",
        "CONTEXT_MAX_TOKENS",
        # Chat- und Vision-Modelle laufen ueber Verbindungen und Modelle (Administration);
        # Sampling-Werte stellt man pro Modell ein.
        "CHAT_MODEL",
        "CHAT_BASEURL",
        "CHAT_TEMPERATURE",
        "CHAT_TOP_P",
        "CHAT_TOP_K",
        "CHAT_MAX_TOKENS",
        "CHAT_NUM_CTX",
        "CHAT_THINKING",
        "VISION_MODEL",
        "VISION_BASEURL",
        "API_KEY",
        # Infrastruktur kommt aus der Umgebung (docker-compose / .env), nicht aus der Oberflaeche.
        "DATA_DIR",
        "SUMMARIES_DIR",
        "QDRANT_HOST",
        "QDRANT_PORT",
        "QDRANT_API_KEY",
        # Eigene Karte unter Administration > Tools.
        "TOOL_WORKER_URL",
        "PANDAS_TIMEOUT_S",
        "PANDAS_MAX_MEMORY_MB",
        "PANDAS_MAX_OUTPUT_CHARS",
        "SQL_WORKER_URL",
        "SQL_ROW_LIMIT",
        "SQL_TIMEOUT_S",
        "SQL_MAX_CELL_CHARS",
        "SQL_PROMPT",
        "SQL_VALUES_MAX_DISTINCT",
        "SQL_VALUES_MAX_RESULTS",
        "SQL_VALUES_HINT",
        # Die Websuche wird pro Modell zugeschaltet (Modelle > Konfigurieren).
        "SEARCH_SEARXNG_ENABLED",
        "SECRET_KEY",
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

    DESCRIPTIONS = {
        "CHAT_MODEL": "Modell für Chat-Interaktionen (z.B. qwen3:8b, buddy)",
        "CHAT_BASEURL": "Ollama/OpenAI-Compatible API URL",
        "CHAT_TEMPERATURE": "Kreativität des Modells (0.0 = deterministisch, 1.0 = kreativ)",
        "CHAT_TOP_P": "Nucleus Sampling (0.0-1.0)",
        "CHAT_TOP_K": "Top-K Sampling (Anzahl der höchstwahrscheinlichen Token)",
        "CHAT_MAX_TOKENS": "Maximale Token-Anzahl pro Antwort (leer = unbegrenzt)",
        "CHAT_NUM_CTX": "Kontextfenster-Größe (Ollama num_ctx)",
        "CHAT_THINKING": "Thinking/Reasoning-Effort (none/low/medium/high/max) - nur für Reasoning-Modelle",
        "VISION_MODEL": "Vision-Modell für Bildverarbeitung in PDFs",
        "VISION_BASEURL": "Vision API URL (falls anders als Chat)",
        "API_KEY": "API Key für Authentifizierung",
        "STT_MODEL": "Speech-to-Text Modell (z.B. nemo-parakeet-tdt-0.6b-v3)",
        "STT_BASEURL": "STT API URL (z.B. OpenAI-compatible Whisper Endpoint)",
        "STT_API_KEY": "API Key für STT Endpoint",
        "STT_MODE": "Betriebsart der Spracheingabe: auto, stream (live) oder upload (aufnehmen und senden)",
        "EMBEDDING_SOURCE": 'Quelle: "local" (ONNX/fastembed) oder "endpoint"',
        "EMBEDDING_MODEL": "Embedding Modell Name",
        "EMBEDDING_ENDPOINT": "URL für externen Embedding-Service",
        "EMBEDDING_DIMENSION": "Vektor-Dimension (muss zum Modell passen)",
        "RETRIEVER_K": "Anzahl Ergebnisse für RAG",
        "RETRIEVER_FETCH_K": "Anzahl Dokumente zum Fetchen (vor MMR)",
        "RETRIEVER_LAMBDA": "MMR Balance: 0.0 = divers, 1.0 = relevant",
        "RETRIEVER_SCORE_THRESHOLD": "Mindest-Ähnlichkeitsscore",
        "DATA_DIR": "Verzeichnis für Excel/CSV Dateien",
        "SECRET_KEY": "Flask Secret Key (wird bei leerem Wert automatisch erzeugt)",
        "SUMMARIES_DIR": "Verzeichnis für gespeicherte Analysen",
        "QDRANT_HOST": "Qdrant Server Host",
        "QDRANT_PORT": "Qdrant Server Port",
        "SEARCH_SEARXNG_URL": "SearXNG Metasuchmaschine URL (z.B. http://localhost:8080)",
        "SEARCH_SEARXNG_CATEGORIES": "Durchsuchbare Kategorien (Komma-getrennt, z.B. general,news,science)",
        "SEARCH_SEARXNG_ENABLED": "Websuche aktivieren (true/false)",
        "PII_FILTER_URL": "Vollständige PII-Filter Endpoint URL (z.B. http://localhost:9500/api/v1/sanitize)",
        "PII_FILTER_API_KEY": "API Key für PII-Filter Service",
        "PII_FILTER_ENABLED": "PII-Filter aktivieren (true/false)",
        "PII_FILTER_TIMEOUT_S": "Zeitlimit eines PII-Filter-Aufrufs in Sekunden (waechst mit der Textlaenge)",
        "PII_FILTER_MAX_CHARS": "Hoechste Textlaenge je PII-Filter-Aufruf; laengere Texte gehen in Stuecken",
        "UI_THEME": "Oberflaeche: light, dark oder system (folgt OS)",
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
            # Neue Setting anlegen
            category = self._get_category(key)
            is_sensitive = key in self.SENSITIVE_KEYS
            description = self.DESCRIPTIONS.get(key, "")
            default_value = os.getenv(key, "")

            setting = Setting(
                key=key,
                value=self._stored(key, value),
                default_value=self._stored(key, default_value),
                category=category,
                is_sensitive=is_sensitive,
                description=description,
            )
            self.db.add(setting)

        self.db.commit()

    def set_batch(self, settings_dict: dict[str, str]) -> None:
        """Mehrere Settings auf einmal speichern"""
        for key, value in settings_dict.items():
            self.set(key, value)

    def get_all_by_category(self) -> dict[str, list[Setting]]:
        """Alle Settings gruppiert nach Kategorie (für Settings-Panel).
        Veraltete Settings (DEPRECATED_KEYS) und UI-exkludierte Keys
        (z.B. APP_PASSWORD_HASH) werden ausgeblendet."""
        settings = self.db.query(Setting).order_by(Setting.key).all()
        result = {cat: [] for cat in self.CATEGORIES}
        result["other"] = []  # Für Settings ohne Kategorie

        for setting in settings:
            if setting.key in self.DEPRECATED_KEYS:
                continue
            if setting.key in self.UI_EXCLUDED_KEYS:
                continue
            cat = setting.category if setting.category in result else "other"
            result[cat].append(setting)

        return result

    def reset_to_default(self, key: str) -> bool:
        """Ein Setting auf Default zurücksetzen"""
        setting = self.db.query(Setting).filter_by(key=key).first()
        if setting:
            setting.value = setting.default_value
            self.db.commit()
            return True
        return False

    def reset_category(self, category: str) -> int:
        """Alle Settings einer Kategorie zurücksetzen"""
        settings = self.db.query(Setting).filter_by(category=category).all()
        count = 0
        for setting in settings:
            setting.value = setting.default_value
            count += 1
        if count > 0:
            self.db.commit()
        return count

    def get_category_display_name(self, category: str) -> str:
        """Deutsche Anzeigenamen für Kategorien"""
        names = {
            "llm": "LLM (veraltet)",
            "stt": "Spracheingabe (STT)",
            "embedding": "Embedding Einstellungen",
            "retriever": "Retriever Einstellungen",
            "storage": "Speicher Einstellungen",
            "qdrant": "Qdrant Einstellungen",
            "auth": "Authentifizierung",
            "search": "Suchmaschine (SearXNG)",
            "pii_filter": "PII-Filter Service",
            "ui": "Oberfläche",
            "other": "Sonstige Einstellungen",
        }
        return names.get(category, category.capitalize())

    def get_missing_from_env(self) -> list[str]:
        """Settings die in .env existieren aber nicht in DB.
        Veraltete Keys werden ignoriert."""
        existing_keys = {s.key for s in self.db.query(Setting.key).all()}
        env_keys = set(os.environ.keys()) - self.DEPRECATED_KEYS
        return sorted(env_keys - existing_keys)

    def import_from_env(self, overwrite: bool = False) -> int:
        """Alle aktuellen .env-Werte in DB importieren.
        Veraltete Keys (DEPRECATED_KEYS) werden nicht importiert."""
        count = 0
        for key in os.environ:
            if key.startswith("_"):  # Interne Vars überspringen
                continue
            if key in self.DEPRECATED_KEYS:
                continue

            existing = self.db.query(Setting).filter_by(key=key).first()
            if existing and not overwrite:
                continue

            value = os.getenv(key)
            if existing:
                existing.value = self._stored(key, value)
            else:
                category = self._get_category(key)
                is_sensitive = key in self.SENSITIVE_KEYS
                description = self.DESCRIPTIONS.get(key, "")

                setting = Setting(
                    key=key,
                    value=self._stored(key, value),
                    default_value=self._stored(key, value),
                    category=category,
                    is_sensitive=is_sensitive,
                    description=description,
                )
                self.db.add(setting)
            count += 1

        if count > 0:
            self.db.commit()
        return count

    def _get_category(self, key: str) -> str:
        """Kategorie für einen Key bestimmen"""
        for cat, keys in self.CATEGORIES.items():
            if key in keys:
                return cat
        return "other"

    # === Convenience Methods für häufige Settings ===
    def get_data_dir(self) -> str:
        """Daten-Verzeichnis"""
        return self.get("DATA_DIR", "./data")

    def get_summaries_dir(self) -> str:
        """Summaries-Verzeichnis"""
        return self.get("SUMMARIES_DIR", "./data/summaries")

    def get_embedding_dimension(self) -> int:
        """Embedding Dimension"""
        return self.get_int("EMBEDDING_DIMENSION", 384)


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

            session = get_db_service().get_session()
            try:
                return getattr(SettingsService(session), name)(*args, **kwargs)
            finally:
                session.close()

        return call
