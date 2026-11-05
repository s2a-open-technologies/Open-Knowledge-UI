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

import asyncio
import atexit
import logging
import os
import threading
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.core.database_service import init_db_service
from app.core.qdrant_client import get_qdrant_client
from app.core.settings_service import LiveSettingsService, SettingsService

# fastembed >=0.6 emits a UserWarning that the paraphrase-multilingual-MiniLM-L12-v2
# model now uses mean pooling instead of CLS embedding. The new behaviour is correct
# and intended; the warning is purely informational. Suppress only this specific
# warning so the log stays clean. Vectors embedded with the old CLS pooling (fastembed
# <=0.5.1) are NOT compatible with the new mean pooling — re-ingest all documents after
# upgrading fastembed.
warnings.filterwarnings(
    "ignore",
    message=r".*now uses mean pooling instead of CLS embedding.*",
    category=UserWarning,
)

logger = logging.getLogger(__name__)

_qdrant_client = None
_db_service = None

# Hintergrund-Initialisierung der Embedding-Modelle (s. init_resources).
# _resources_ready wird gesetzt, sobald der Hintergrund-Thread fertig ist
# (egal ob Erfolg oder Fehler). _resources_error haelt eine evt. Exception.
_resources_ready = threading.Event()
_resources_error = None


def resources_ready():
    """True, sobald die Hintergrund-Initialisierung erfolgreich abgeschlossen ist."""
    return _resources_ready.is_set() and _resources_error is None


def wait_for_resources(timeout=None):
    """Blockiert bis die Hintergrund-Initialisierung fertig ist oder der Timeout
    ablaeuft. Gibt True zurueck, wenn die Ressourcen bereit sind, False bei Timeout.
    Loest RuntimeError aus, falls die Initialisierung im Hintergrund fehlschlug."""
    if not _resources_ready.wait(timeout):
        return False
    if _resources_error is not None:
        raise RuntimeError(
            "Ressourcen-Initialisierung fehlgeschlagen"
        ) from _resources_error
    return True


def _cleanup():
    """Cleanup beim Beenden"""
    global _qdrant_client
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
        except Exception:
            pass


# Kein eigener SIGTERM/SIGINT-Handler: Uvicorn faehrt geordnet herunter (Lifespan), atexit raeumt auf.
atexit.register(_cleanup)


def init_resources(preload=False):
    """Initialisiert Ressourcen mit Settings aus DB.

    Qdrant-Client und Collection werden sofort (synchron) angelegt, da viele
    Dokument-Endpunkte darauf zugreifen. Das Laden der Embedding-Modelle und der
    Aufbau der darauf aufbauenden Kette (Vectorstore -> Retriever -> Agent)
    laufen je nach ``preload``-Parameter:

    - ``preload=True``: synchron (blockierend) — der Server bindet erst, wenn
      alle Modelle geladen sind. Ideal fuer Produktivumgebungen (keine
      "bitte warten"-Fehler beim ersten Request).

    - ``preload=False`` (Default): asynchron im Hintergrund-Thread, damit ein
      Modell-Download den App-Start nicht blockiert (Docker-Start-Timeout).
      Siehe ``wait_for_resources()`` / ``resources_ready()``.
    """
    global _qdrant_client

    # Settings Service holen
    from app.core import runtime

    settings = runtime.state.settings
    if not settings:
        logger.error("Settings Service nicht verfügbar!")
        _resources_ready.set()
        return

    collection_name = "open_knowledge_ui"
    _qdrant_client = get_qdrant_client(collection_name=collection_name)

    # Collections mit bekannter Dimension sofort anlegen (schnell; Dense + HNSW, BM25 mit IDF;
    # bestehende bekommen den IDF-Modifier nachgetragen). Wo die Dimension noch unbekannt ist
    # (frische Installation), legt model_settings.prepare sie nach dem Laden des Modells an.
    from app.knowledge.service import list_knowledge
    from app.rag.collections import ensure_collection

    try:
        for kb in list_knowledge():
            if kb["embedding_dimension"] and kb.get("kind") not in ("external", "sql"):
                ensure_collection(
                    _qdrant_client, kb["collection_name"], kb["embedding_dimension"]
                )
    except Exception as e:
        logger.warning(f"Knowledge-Collections konnten nicht angelegt werden: {e}")

    # Qdrant-Client sofort fuer Dokument-Endpunkte bereithalten
    runtime.state.client = _qdrant_client

    # Alle Settings im Main-Thread vorlesen (SQLAlchemy-Session ist nicht
    # thread-safe) und als Captures in den Hintergrund-Thread reichen.
    retriever_k = settings.get_int("RETRIEVER_K", 5)
    retriever_fetch_k = settings.get_int("RETRIEVER_FETCH_K", 30)
    retriever_lambda = settings.get_float("RETRIEVER_LAMBDA", 0.5)
    retriever_threshold = settings.get_float("RETRIEVER_SCORE_THRESHOLD", 0.2)
    data_dir = settings.get("DATA_DIR", "./data")
    summaries_dir = settings.get("SUMMARIES_DIR", "./data/summaries")

    # Embedding- und Reranking-Modell im Hintergrund laden (Download beim ersten Start),
    # damit die erste Suche bzw. das erste Einlesen nicht darauf wartet.
    threading.Thread(
        target=_prepare_models,
        args=(_qdrant_client,),
        name="prepare-models",
        daemon=True,
    ).start()

    # Modell-Laden: synchron (preload) oder asynchron (Hintergrund-Thread)
    if preload:
        _load_resources_sync(
            {
                "retriever_k": retriever_k,
                "retriever_fetch_k": retriever_fetch_k,
                "retriever_lambda": retriever_lambda,
                "retriever_threshold": retriever_threshold,
                "data_dir": data_dir,
                "summaries_dir": summaries_dir,
            },
        )
    else:
        _load_resources_async(
            {
                "retriever_k": retriever_k,
                "retriever_fetch_k": retriever_fetch_k,
                "retriever_lambda": retriever_lambda,
                "retriever_threshold": retriever_threshold,
                "data_dir": data_dir,
                "summaries_dir": summaries_dir,
            },
        )


def _prepare_models(client):
    try:
        from app.rag import model_settings

        model_settings.prepare(client)
    except Exception as e:
        logger.warning(f"Modelle konnten nicht vorbereitet werden: {e}")


def _load_resources_sync(settings_dict):
    """Laedt KI-Ressourcen synchron (blockierend)."""
    global _resources_error
    try:
        from app.agent.resources import load_resources

        load_resources(
            qdrant_client=_qdrant_client,
            collection_name="open_knowledge_ui",
            **settings_dict,
        )
    except Exception as e:
        _resources_error = e
        logger.exception("Ressourcen-Initialisierung fehlgeschlagen")
    finally:
        _resources_ready.set()


def _load_resources_async(settings_dict):
    """Startet das Laden der KI-Ressourcen in einem Hintergrund-Thread."""
    threading.Thread(
        target=_load_resources_sync,
        args=(settings_dict,),
        name="init-resources",
        daemon=True,
    ).start()


def ensure_directories(settings):
    """Stellt sicher, dass alle benötigten Verzeichnisse existieren.

    Migriert außerdem einmalig Dateien aus den alten
    data/files/<user_id>/... und data/processed_files/<user_id>/...
    Unterordnern in die neue flache Struktur (Single-User)."""
    data_dir = settings.get("DATA_DIR", "./data")
    summaries_dir = settings.get("SUMMARIES_DIR", "./data/summaries")

    directories = [
        data_dir,
        summaries_dir,
        os.path.join(data_dir, "files"),
        os.path.join(data_dir, "processed_files"),
        os.path.join(data_dir, "images"),
    ]
    for d in directories:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"Verzeichnis sichergestellt: {d}")

    _flatten_user_subdirs(os.path.join(data_dir, "files"))
    _flatten_user_subdirs(os.path.join(data_dir, "processed_files"))


def _flatten_user_subdirs(base_dir: str):
    """Verschiebt Dateien aus numerischen Unterordnern eine Ebene nach oben.

    Überschreibt bei Namenskollisionen die ältere Datei (Single-User, daher
    spielt die Herkunft keine Rolle mehr). Löscht danach leere Unterordner."""
    base = Path(base_dir)
    if not base.exists():
        return

    moved_any = False
    for subdir in base.iterdir():
        if not subdir.is_dir():
            continue
        try:
            int(subdir.name)
        except ValueError:
            continue

        for file_path in subdir.rglob("*"):
            if not file_path.is_file():
                continue
            target = base / file_path.name
            try:
                file_path.rename(target)
                moved_any = True
            except FileExistsError:
                file_path.unlink()
                moved_any = True
            except Exception as e:
                logger.warning(f"Konnte {file_path} nicht migrieren: {e}")

        if moved_any:
            try:
                _remove_empty_dirs(subdir)
            except Exception as e:
                logger.warning(
                    f"Konnte leere Ordner unter {subdir} nicht entfernen: {e}"
                )


def _remove_empty_dirs(path: Path):
    """Löscht leere Verzeichnisse rekursiv von innen nach außen."""
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_dir() and not any(child.iterdir()):
            child.rmdir()
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def create_app(
    preload: bool | None = None, resources: bool = True, routers: tuple[str, ...] | None = None
) -> FastAPI:
    """FastAPI-App bauen.

    ``preload``: Modelle vor dem Start laden (Default aus ``APP_PRELOAD``), sonst im Hintergrund.
    ``resources=False`` (Tests): kein Qdrant, keine Modelle, kein Redis.
    """
    from starlette.middleware.sessions import SessionMiddleware

    from app.api import errors, include_routers
    from app.core import redis_bus, runtime
    from app.core.csrf import CSRFMiddleware, cookie_secure
    from app.core.request_context import RequestContextMiddleware
    from app.core.security_headers import SecurityHeadersMiddleware

    if preload is None:
        preload = os.getenv("APP_PRELOAD", "").strip().lower() in ("1", "true", "yes", "on")

    # Database Service initialisieren
    global _db_service
    _db_service = init_db_service()
    _db_service.init_default_data()

    # Settings Service erstellen
    db_session = _db_service.get_session()
    settings = SettingsService(db_session)

    # Verzeichnisse mit Settings erstellen
    ensure_directories(settings)

    # SECRET_KEY wird persistiert (Settings DB → env → neu generiert), damit
    # Sessions App-Neustarts überleben und sich nicht bei jedem Start vermehren.
    secret_key = settings.get("SECRET_KEY")
    if not secret_key:
        secret_key = os.urandom(32).hex()
        settings.set("SECRET_KEY", secret_key)
    db_session.close()  # nur fuer den Start; danach liest jeder Zugriff frisch

    runtime.state.db = _db_service
    runtime.state.settings = LiveSettingsService()

    if resources:
        init_resources(preload=preload)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        retention_task = None
        if resources:
            await redis_bus.start()
            from app.chat import retention

            retention_task = asyncio.create_task(
                retention.run_periodically(lambda: runtime.state.client),
                name="chat-retention",
            )
        try:
            yield
        finally:
            if retention_task is not None:
                retention_task.cancel()
            await redis_bus.stop()
            _cleanup()

    app = FastAPI(
        title="Open Knowledge UI",
        lifespan=lifespan,
        # Nur das Schema: die Swagger-Seite laedt Skripte vom CDN, das die CSP (nur gleiche Herkunft) sperrt
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.state.csrf_enabled = True
    app.state.cookie_secure = cookie_secure()
    app.state.secret_key = secret_key  # Tests: Session-Cookie signieren

    errors.install(app)
    include_routers(app, routers)
    # Gebaute SPA (Vite schreibt nach app/static, siehe frontend/vite.config.ts)
    from fastapi.staticfiles import StaticFiles

    from app.api.pages import STATIC_DIR

    app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")

    # Middleware: zuletzt hinzugefuegt = aussen. Reihenfolge von aussen nach innen:
    # Request-Kontext -> Sicherheits-Header -> Session-Cookie -> CSRF -> Router.
    app.add_middleware(CSRFMiddleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key=secret_key,
        session_cookie="session",
        max_age=None,  # Browser-Sitzung (wie bisher)
        same_site="lax",
        https_only=app.state.cookie_secure,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)
    return app
