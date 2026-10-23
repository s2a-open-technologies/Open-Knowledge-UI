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
Database Service mit SQLAlchemy (Single-User)
"""

import os
import uuid
from datetime import datetime

from flask import after_this_request, g, has_request_context
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as SQLAlchemySession
from sqlalchemy.orm import sessionmaker
from werkzeug.security import generate_password_hash

from app.core.models import (
    Base,
    ChatAttachment,
    ChatSession,
    Message,
    Setting,
    UserDocument,
    UserSummaryFile,
)


def _close_request_sessions(response):
    for session in g.pop("_opened_sessions", []):
        try:
            session.close()  # idempotent; bereits geschlossene Sessions schaden nicht
        except Exception:
            pass
    return response


class DatabaseService:
    """Zentraler Database Service für SQLAlchemy (Single-User)"""

    def __init__(self, db_path: str = "./data/app.db"):
        """``db_path``: SQLite-Dateipfad (Tests, lokale Entwicklung) oder eine SQLAlchemy-URL
        (``postgresql+psycopg://...``). Betrieb ist Postgres; Standard ist ``DATABASE_URL``."""
        if "://" in db_path:
            url = db_path
        else:
            url = f"sqlite:///{db_path}"
        if url.startswith("sqlite"):
            self.engine = create_engine(
                url, echo=False, connect_args={"check_same_thread": False}
            )
            # Nur SQLite legt Tabellen selbst an; Postgres kommt ausschliesslich ueber Alembic
            # (sonst widerspricht der Tabellenstand dem Migrationsstand).
            Base.metadata.create_all(self.engine)
        else:
            self.engine = create_engine(
                url, echo=False, pool_pre_ping=True, pool_size=10, max_overflow=10
            )
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self) -> SQLAlchemySession:
        """Neue Session erstellen.

        Innerhalb eines Requests wird sie nach der Antwort automatisch geschlossen, auch
        wenn der Aufrufer es vergisst (z. B. ``SettingsService(db.get_session())``).
        Sonst bliebe die Verbindung belegt: SQLite sperrt, der Postgres-Pool laeuft voll.
        """
        session = self.Session()
        if has_request_context():
            opened = g.setdefault("_opened_sessions", [])
            if not opened:
                after_this_request(_close_request_sessions)
            opened.append(session)
        return session

    def init_default_data(self):
        """Erst-Setup: Settings und App-Passwort initialisieren"""
        db = self.get_session()
        try:
            from app.core.models import Setting

            # Frische Installation? (vor dem Anlegen der Standard-Settings pruefen)
            fresh = db.query(Setting).filter_by(key="EMBEDDING_MODEL").first() is None
            self._init_settings(db)
            from app.rag.model_settings import ensure_model_settings

            ensure_model_settings(db, fresh)
            self._init_app_password(db)
            from app.rag.knowledge import ensure_default_knowledge

            ensure_default_knowledge(db)
            from app.llm.connections import ensure_default_connection

            ensure_default_connection(db)
            from app.auth.users import ensure_admin_user

            ensure_admin_user(db)
            from app.rag.profiles import ensure_default_profiles

            ensure_default_profiles(db)
            db.commit()
            # Zugangsdaten im Klartext (Altbestand) verschluesseln und den Schluessel pruefen
            from app.core import crypto
            from app.core.settings_service import SettingsService

            crypto.encrypt_existing(self.engine, SettingsService.encrypted_keys())
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _init_settings(self, db: SQLAlchemySession):
        """Settings mit .env-Werten initialisieren"""
        from app.core.settings_service import SettingsService
        from app.ingest import prompts as ingest_prompts

        all_settings = {
            "CHAT_MODEL": ("llm", "qwen3:8b", "Modell für Chat-Interaktionen"),
            "CHAT_BASEURL": (
                "llm",
                "http://localhost:11434/v1",
                "Ollama/OpenAI API URL",
            ),
            "CHAT_TEMPERATURE": ("llm", "0.1", "Temperatur für Chat (0.0-1.0)"),
            "CHAT_TOP_P": ("llm", "0.2", "Top-P Sampling"),
            "CHAT_TOP_K": ("llm", "40", "Top-K Sampling"),
            "CHAT_MAX_TOKENS": ("llm", "", "Max. Token-Anzahl (leer = unbegrenzt)"),
            "CHAT_NUM_CTX": ("llm", "4096", "Kontextfenster (Ollama num_ctx)"),
            "CHAT_THINKING": (
                "llm",
                "none",
                "Thinking-Effort (none/low/medium/high/max)",
            ),
            "VISION_MODEL": ("llm", "qwen3-vl:8b", "Vision-Modell für Bilder"),
            "VISION_BASEURL": ("llm", "http://localhost:11434/v1", "Vision API URL"),
            "API_KEY": ("llm", "ollama", "API Key", True),
            "STT_MODEL": ("stt", "nemo-parakeet-tdt-0.6b-v3", "STT Modell"),
            "STT_BASEURL": ("stt", "http://parakeet:5001/v1", "STT API URL"),
            "STT_API_KEY": ("stt", "ollama", "STT API Key", True),
            "STT_MODE": (
                "stt",
                "auto",
                "Betriebsart der Spracheingabe: auto, stream (live, Parakeet) oder upload (aufnehmen und senden, OpenAI-Standard)",
            ),
            "EMBEDDING_SOURCE": ("embedding", "local", "Quelle: local oder endpoint"),
            "EMBEDDING_MODEL": (
                "embedding",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "Embedding Modell",
            ),
            "EMBEDDING_ENDPOINT": (
                "embedding",
                "http://localhost:8080/v1",
                "Embedding Endpoint URL",
            ),
            "EMBEDDING_DIMENSION": ("embedding", "384", "Vektor-Dimension"),
            "RETRIEVER_K": ("retriever", "5", "Anzahl Ergebnisse"),
            "RETRIEVER_FETCH_K": ("retriever", "30", "Anzahl zum Fetchen"),
            "RETRIEVER_LAMBDA": ("retriever", "0.5", "MMR Lambda (0.0-1.0)"),
            "RETRIEVER_SCORE_THRESHOLD": ("retriever", "0.2", "Mindest-Score"),
            "SECRET_KEY": (
                "auth",
                "",
                "Flask Secret Key (wird bei leerem Wert automatisch erzeugt)",
            ),
            "DATA_DIR": ("storage", "./data", "Daten-Verzeichnis"),
            "SUMMARIES_DIR": ("storage", "./data/summaries", "Summaries-Verzeichnis"),
            "QDRANT_HOST": ("qdrant", "localhost", "Qdrant Host"),
            "QDRANT_PORT": ("qdrant", "6333", "Qdrant Port"),
            "INGEST_CHUNK_SIZE": ("ingest", "1000", "Chunk-Groesse in Zeichen"),
            "INGEST_CHUNK_OVERLAP": (
                "ingest",
                "150",
                "Ueberlappung beim Zerlegen langer Absaetze",
            ),
            "INGEST_CONTEXT_ENRICHMENT": (
                "ingest",
                "true",
                "LLM-Kontextanreicherung pro Chunk (langsamer, aber mit Einordnung)",
            ),
            "INGEST_KEEP_ORIGINAL": (
                "ingest",
                "true",
                "Originaldatei nach dem Einlesen archivieren (aus: nur das Markdown bleibt)",
            ),
            "INGEST_VISION_ENABLED": (
                "ingest",
                "true",
                "Bilder in Dokumenten per Vision-Modell beschreiben (aus: Bilder werden uebersprungen)",
            ),
            "BM25_LANGUAGE": (
                "ingest",
                "german",
                "Sprache der Stichwortsuche (BM25: Stemmer und Stoppwoerter); gilt fuer neue und neu indexierte Knowledge",
            ),
            "INGEST_INDEX_ENABLED": (
                "ingest",
                "true",
                "Inhaltsverzeichnis (index.md) pro Knowledge erzeugen und dem Modell geben",
            ),
            "UI_THEME": ("ui", "system", "Oberflaeche: light, dark oder system"),
            "VISION_PROMPT": (
                "ingest",
                ingest_prompts.VISION_PROMPT,
                "Anweisung fuer die Beschreibung von Bildern in Dokumenten",
            ),
            "CONTEXT_PROMPT": (
                "ingest",
                ingest_prompts.CONTEXT_PROMPT,
                "Anweisung fuer die Kontext-Anreicherung (Platzhalter {dokument} und {abschnitt})",
            ),
            "CONTEXT_EXCERPT_CHARS": (
                "ingest",
                str(ingest_prompts.CONTEXT_EXCERPT_CHARS),
                "Zeichen des Dokuments, die der Kontext-Anreicherung als Auszug dienen",
            ),
            "INGEST_TABLE_VISION": (
                "ingest",
                "true",
                "Tabellen aus PDFs zusaetzlich per Vision-Modell pruefen und korrigieren (wirkt nur mit Bildbeschreibung)",
            ),
            "TABLES_PROMPT": (
                "ingest",
                ingest_prompts.TABLES_PROMPT,
                "Anweisung fuer die Pruefung von Tabellen per Vision-Modell",
            ),
            "TABLES_MODEL_REF": (
                "ingest",
                "",
                "Modell der Tabellen-Pruefung (leer = Modell der Bildbeschreibung)",
            ),
            "TABLES_TEMPERATURE": ("ingest", "0", "Temperatur der Tabellen-Pruefung"),
            "TABLES_MAX_TOKENS": (
                "ingest",
                "",
                "Maximale Antwortlaenge der Tabellen-Pruefung (leer = Wert des Modells)",
            ),
            "TABLES_MAX_ROW_DEVIATION": (
                "ingest",
                str(ingest_prompts.TABLES_MAX_ROW_DEVIATION),
                "Prozent, um die die Zeilenzahl der geprueften Tabelle vom Entwurf abweichen darf, bevor sie verworfen wird",
            ),
            "INGEST_CHAT_IMAGES": (
                "ingest",
                "true",
                "Bilder im Chat annehmen: das Bildmodell liest sie, der Text wird wie ein Anhang durchsuchbar",
            ),
            "IMAGES_PROMPT": (
                "ingest",
                ingest_prompts.IMAGES_PROMPT,
                "Anweisung fuer das Lesen von Bildern im Chat",
            ),
            "IMAGES_MODEL_REF": (
                "ingest",
                "",
                "Modell fuer Bilder im Chat (leer = Modell der Bildbeschreibung)",
            ),
            "IMAGES_TEMPERATURE": ("ingest", "0", "Temperatur fuer Bilder im Chat"),
            "IMAGES_MAX_TOKENS": (
                "ingest",
                "",
                "Maximale Antwortlaenge fuer Bilder im Chat (leer = Wert des Modells)",
            ),
            "VISION_TEMPERATURE": (
                "ingest",
                "",
                "Temperatur der Bildbeschreibung (leer = Wert des Modells)",
            ),
            "VISION_MAX_TOKENS": (
                "ingest",
                "",
                "Maximale Antwortlaenge der Bildbeschreibung (leer = Wert des Modells)",
            ),
            "CONTEXT_MODEL_REF": (
                "ingest",
                "",
                "Modell der Kontext-Anreicherung (leer = Standardmodell)",
            ),
            "CONTEXT_TEMPERATURE": (
                "ingest",
                "",
                "Temperatur der Kontext-Anreicherung (leer = Wert des Modells)",
            ),
            "CONTEXT_MAX_TOKENS": (
                "ingest",
                "",
                "Maximale Antwortlaenge der Kontext-Anreicherung (leer = Wert des Modells)",
            ),
            "AUTH_LDAP": (
                "auth",
                "{}",
                "LDAP/AD-Anmeldung (JSON, Verwaltung unter Administration > Authentifizierung)",
                True,
            ),
            "AUTH_OIDC": (
                "auth",
                "{}",
                "OIDC-Anmeldung (JSON, Verwaltung unter Administration > Authentifizierung)",
                True,
            ),
            "AUTH_DEFAULT_ROLE": ("auth", "user", "Rolle neuer externer Konten"),
            "AUTH_AUTO_CREATE": (
                "auth",
                "true",
                "Externe Konten beim ersten Login automatisch anlegen",
            ),
            "APP_BASE_URL": (
                "auth",
                "",
                "Oeffentliche Adresse der App (fuer den OIDC-Rueckruf)",
            ),
            "SEARCH_SEARXNG_URL": (
                "search",
                "",
                "SearXNG-URL (z. B. http://localhost:8081). Die Websuche wird pro Modell zugeschaltet.",
            ),
            "SEARCH_SEARXNG_CATEGORIES": (
                "search",
                "general",
                "Durchsuchbare Kategorien (Komma-getrennt, z. B. general,news,science)",
            ),
            "PII_FILTER_URL": (
                "pii_filter",
                "",
                "Vollstaendige PII-Filter-Endpoint-URL (z. B. http://localhost:8082/api/v1/sanitize)",
            ),
            "PII_FILTER_API_KEY": ("pii_filter", "", "API-Key des PII-Filters", True),
            "PII_FILTER_ENABLED": (
                "pii_filter",
                "true",
                "PII-Filter aktivieren (true/false)",
            ),
            "PII_FILTER_TIMEOUT_S": (
                "pii_filter",
                "10",
                "Zeitlimit eines PII-Filter-Aufrufs in Sekunden (waechst mit der Textlaenge)",
            ),
            "PII_FILTER_MAX_CHARS": (
                "pii_filter",
                "12000",
                "Hoechste Textlaenge je PII-Filter-Aufruf; laengere Texte gehen in Stuecken",
            ),
            "SEARCH_ALLOW_UNMASKED": (
                "search",
                "false",
                "Websuche auch ohne aktiven PII-Filter erlauben (Suchanfragen gehen dann unmaskiert raus)",
            ),
        }

        from app.chat.compaction import SETTINGS as compaction_settings

        for key, default, description in compaction_settings:
            all_settings[key] = ("llm", default, description)

        from app.agent.behavior import SETTINGS as behavior_settings

        for key, default, description in behavior_settings:
            all_settings[key] = ("llm", default, description)

        from app.toolworker.config import SETTINGS as tool_worker_settings

        for key, default, description in tool_worker_settings:
            all_settings[key] = ("tools", default, description)

        for key, config in all_settings.items():
            if len(config) == 4:
                category, default, description, is_sensitive = config
            else:
                category, default, description = config
                is_sensitive = False

            existing = db.query(Setting).filter_by(key=key).first()
            if not existing:
                env_value = os.getenv(key)
                value = env_value if env_value is not None else default

                setting = Setting(
                    key=key,
                    value=SettingsService._stored(key, value),
                    default_value=SettingsService._stored(key, default),
                    category=category,
                    is_sensitive=is_sensitive,
                    description=description,
                )
                db.add(setting)

        deprecated_keys = {"MODEL", "BASEURL", "TEMPERATURE", "TOP_P", "CHAT_DB_PATH"}
        db.query(Setting).filter(Setting.key.in_(deprecated_keys)).delete(
            synchronize_session=False
        )

        old_summaries = db.query(Setting).filter_by(key="SUMMARIES_DIR").first()
        if old_summaries and old_summaries.value == "./summaries":
            old_summaries.value = "./data/summaries"

    def _init_app_password(self, db: SQLAlchemySession):
        """APP_PASSWORD_HASH sicherstellen (Default aus .env oder 'admin')."""
        from app.core.settings_service import SettingsService

        settings_service = SettingsService(db)
        if not settings_service.get("APP_PASSWORD_HASH"):
            default_password = os.getenv("ADMIN_PASSWORD", "admin")
            settings_service.set(
                "APP_PASSWORD_HASH", generate_password_hash(default_password)
            )

    # === ChatSession Methods ===

    def create_session(
        self, title: str | None = None, user_id: int | None = None
    ) -> ChatSession:
        """Neue Session erstellen"""
        db = self.get_session()
        try:
            session = ChatSession(id=str(uuid.uuid4()), title=title, user_id=user_id)
            db.add(session)
            db.commit()
            db.refresh(session)
            return session
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_chat_session(
        self, session_id: str, user_id: int | None = None
    ) -> ChatSession | None:
        """Chat-Session nach ID finden; mit ``user_id`` nur wenn sie ihm gehoert"""
        db = self.get_session()
        try:
            q = db.query(ChatSession).filter_by(id=session_id)
            if user_id is not None:
                q = q.filter_by(user_id=user_id)
            return q.first()
        finally:
            db.close()

    def list_sessions(self, user_id: int | None = None) -> list[ChatSession]:
        """Sessions auflisten (mit ``user_id`` nur die eigenen)"""
        db = self.get_session()
        try:
            q = db.query(ChatSession)
            if user_id is not None:
                q = q.filter_by(user_id=user_id)
            return q.order_by(ChatSession.updated_at.desc()).all()
        finally:
            db.close()

    def delete_session(self, session_id: str, user_id: int | None = None) -> bool:
        """Session löschen (mit ``user_id`` nur eine eigene)"""
        db = self.get_session()
        try:
            q = db.query(ChatSession).filter_by(id=session_id)
            if user_id is not None:
                q = q.filter_by(user_id=user_id)
            session = q.first()
            if session:
                db.delete(session)
                db.commit()
                return True
            return False
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_history(self, session_id: str):
        """Zusammenfassung der Session und alle Nachrichten danach (aelteste zuerst)."""
        db = self.get_session()
        try:
            sess = db.query(ChatSession).filter_by(id=session_id).first()
            summary = sess.summary if sess else None
            through = sess.summary_through_id if sess else None
            q = db.query(Message).filter_by(session_id=session_id)
            if summary and through:
                q = q.filter(Message.id > through)
            rows = q.order_by(Message.created_at.asc(), Message.id.asc()).all()
            for r in rows:
                db.expunge(r)
            return summary, rows
        finally:
            db.close()

    def set_session_summary(
        self, session_id: str, summary: str | None, through_id: int | None
    ) -> None:
        """Speichert die Zusammenfassung der Session (``None`` loescht sie)."""
        db = self.get_session()
        try:
            sess = db.query(ChatSession).filter_by(id=session_id).first()
            if sess:
                sess.summary = summary
                sess.summary_through_id = through_id
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def update_session_title(self, session_id: str, title: str) -> bool:
        """Session-Titel aktualisieren"""
        db = self.get_session()
        try:
            session = db.query(ChatSession).filter_by(id=session_id).first()
            if session:
                session.title = title
                session.updated_at = datetime.utcnow()
                db.commit()
                return True
            return False
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # === Message Methods ===

    def save_message(self, session_id: str, role: str, content: str) -> Message:
        """Nachricht speichern"""
        db = self.get_session()
        try:
            message = Message(session_id=session_id, role=role, content=content)
            db.add(message)

            session = db.query(ChatSession).filter_by(id=session_id).first()
            if session:
                session.updated_at = datetime.utcnow()

            db.commit()
            db.refresh(message)
            return message
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_messages(self, session_id: str, limit: int = 100) -> list[Message]:
        """Nachrichten einer Session"""
        db = self.get_session()
        try:
            return (
                db.query(Message)
                .filter_by(session_id=session_id)
                .order_by(Message.created_at.asc())
                .limit(limit)
                .all()
            )
        finally:
            db.close()

    def get_last_session(self) -> ChatSession | None:
        """Letzte Session"""
        db = self.get_session()
        try:
            return db.query(ChatSession).order_by(ChatSession.updated_at.desc()).first()
        finally:
            db.close()

    # === UserDocument Methods ===

    def add_document(
        self,
        document_id: str,
        filename: str,
        knowledge_id: int | None = None,
        markdown_path: str | None = None,
        parser_version: int | None = None,
        uploaded_by: int | None = None,
    ) -> UserDocument:
        """Dokument-Zuordnung erstellen"""
        db = self.get_session()
        try:
            doc = UserDocument(
                document_id=document_id,
                filename=filename,
                knowledge_id=knowledge_id,
                markdown_path=markdown_path,
                parser_version=parser_version,
                uploaded_by=uploaded_by,
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            return doc
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_all_documents(self, knowledge_id: int | None = None) -> list[UserDocument]:
        """Dokumente auflisten (optional nur eines Knowledge-Bereichs)"""
        db = self.get_session()
        try:
            q = db.query(UserDocument)
            if knowledge_id is not None:
                q = q.filter(UserDocument.knowledge_id == knowledge_id)
            return q.order_by(UserDocument.uploaded_at.desc()).all()
        finally:
            db.close()

    # === ChatAttachment Methods ===

    def create_attachment(
        self, session_id: str, filename: str, user_id: int | None = None
    ) -> ChatAttachment:
        db = self.get_session()
        try:
            att = ChatAttachment(
                id=str(uuid.uuid4()),
                session_id=session_id,
                filename=filename,
                status="processing",
                user_id=user_id,
            )
            db.add(att)
            db.commit()
            db.refresh(att)
            return att
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_attachment(self, attachment_id: str) -> ChatAttachment | None:
        db = self.get_session()
        try:
            return db.query(ChatAttachment).filter_by(id=attachment_id).first()
        finally:
            db.close()

    def list_session_attachments(
        self, session_id: str, status: str | None = None
    ) -> list[ChatAttachment]:
        db = self.get_session()
        try:
            q = db.query(ChatAttachment).filter_by(session_id=session_id)
            if status:
                q = q.filter_by(status=status)
            return q.order_by(ChatAttachment.created_at).all()
        finally:
            db.close()

    def set_attachment_status(
        self, attachment_id: str, status: str, error: str | None = None
    ) -> None:
        db = self.get_session()
        try:
            att = db.query(ChatAttachment).filter_by(id=attachment_id).first()
            if att:
                att.status = status
                att.error = error
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def set_attachment_markdown(self, attachment_id: str, path: str) -> None:
        db = self.get_session()
        try:
            att = db.query(ChatAttachment).filter_by(id=attachment_id).first()
            if att:
                att.markdown_path = path
                db.commit()
        finally:
            db.close()

    def delete_attachment_row(self, attachment_id: str) -> None:
        db = self.get_session()
        try:
            db.query(ChatAttachment).filter_by(id=attachment_id).delete()
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def delete_session_attachments(self, session_id: str) -> None:
        db = self.get_session()
        try:
            db.query(ChatAttachment).filter_by(session_id=session_id).delete()
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_document_by_id(self, doc_id: int) -> UserDocument | None:
        """Dokument nach ID finden"""
        db = self.get_session()
        try:
            return db.query(UserDocument).filter_by(id=doc_id).first()
        finally:
            db.close()

    def delete_user_document(self, doc_id: int) -> bool:
        """Dokument-Zuordnung löschen"""
        db = self.get_session()
        try:
            doc = db.query(UserDocument).filter_by(id=doc_id).first()
            if doc:
                db.delete(doc)
                db.commit()
                return True
            return False
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # === UserSummaryFile Methods ===

    def create_summary_file(
        self, filename: str, user_id: int | None = None
    ) -> UserSummaryFile:
        """Neue Summary-Datei-Eintrag erstellen"""
        db = self.get_session()
        try:
            summary = UserSummaryFile(filename=filename, user_id=user_id)
            db.add(summary)
            db.commit()
            db.refresh(summary)
            return summary
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_all_summaries(self) -> list:
        """Alle Summary-Dateien auflisten"""
        db = self.get_session()
        try:
            from app.core.models import UserSummaryFile

            return (
                db.query(UserSummaryFile)
                .order_by(UserSummaryFile.created_at.desc())
                .all()
            )
        finally:
            db.close()

    def get_summary_by_filename(self, filename: str) -> UserSummaryFile | None:
        """Summary-Datei nach Filename finden"""
        db = self.get_session()
        try:
            return db.query(UserSummaryFile).filter_by(filename=filename).first()
        finally:
            db.close()

    def delete_summary_file(self, file_id: int) -> bool:
        """Summary-Datei-Eintrag löschen"""
        db = self.get_session()
        try:
            summary = db.query(UserSummaryFile).filter_by(id=file_id).first()
            if summary:
                db.delete(summary)
                db.commit()
                return True
            return False
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def delete_summary_by_filename(
        self, filename: str, user_id: int | None = None
    ) -> bool:
        """Summary-Datei-Eintrag nach Filename löschen; nur der des angegebenen Nutzers,
        nie der gleichnamige Eintrag eines anderen."""
        db = self.get_session()
        try:
            from app.core.models import UserSummaryFile

            summary = (
                db.query(UserSummaryFile)
                .filter_by(filename=filename, user_id=user_id)
                .first()
            )
            if summary:
                db.delete(summary)
                db.commit()
                return True
            return False
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()


# Singleton für App-Context
db_service = None


def init_db_service(db_path: str | None = None) -> DatabaseService:
    """Database Service initialisieren (Singleton).

    Ohne Argument gilt ``DATABASE_URL`` aus der Umgebung, sonst ``./data/app.db``.
    """
    global db_service
    db_path = db_path or os.getenv("DATABASE_URL") or "./data/app.db"
    if db_service is None:
        db_service = DatabaseService(db_path)
    return db_service


def get_db_service() -> DatabaseService:
    """Bestehenden Database Service zurückgeben"""
    global db_service
    if db_service is None:
        raise RuntimeError(
            "Database Service nicht initialisiert. Zuerst init_db_service() aufrufen."
        )
    return db_service
