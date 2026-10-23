import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.core import database_service


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Isolierte SQLite-DB mit Standard-Knowledge; Pfade relativ zu tmp_path."""
    monkeypatch.chdir(tmp_path)
    # init_db_service ist ein Singleton: pro Test zuruecksetzen (nie die echte DB).
    monkeypatch.setattr(database_service, "db_service", None)
    (tmp_path / "data").mkdir()
    # TEST_DATABASE_URL=postgresql+psycopg://... fuehrt die Suite gegen Postgres aus
    # (Schema wird pro Test neu angelegt); sonst SQLite in tmp_path.
    target = os.getenv("TEST_DATABASE_URL") or str(tmp_path / "data" / "app.db")
    if "://" in target and not target.startswith("sqlite"):
        from sqlalchemy import create_engine

        from app.core.models import Base

        engine = create_engine(target)
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        engine.dispose()
    service = database_service.init_db_service(target)
    service.init_default_data()
    return service


@pytest.fixture(autouse=True)
def fixed_encryption_key(monkeypatch):
    """Fester Schluessel: keine Schluesseldatei im Arbeitsverzeichnis, gleiches Ergebnis in jedem Test."""
    from cryptography.fernet import Fernet

    from app.core import crypto

    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    crypto.reset()
    yield
    crypto.reset()


@pytest.fixture(autouse=True)
def silence_phase_logger():
    """Der PhaseLogger erzeugt Records ohne Zeilennummer; pytests Log-Format
    (%(lineno)d) scheitert daran. Fuer Tests abschalten."""
    from app.utils.phase_logger import phase_logger

    phase_logger.logger.disabled = True
    yield
    phase_logger.logger.disabled = False


@pytest.fixture
def worker(monkeypatch, db):
    """Pandas-Auswertungen laufen gegen einen Worker im selben Prozess."""
    from app.tools.custom_tools import CustomTools
    from app.toolworker import client as worker_client
    from tests import worker_support

    w = worker_support.install(monkeypatch)
    original_init = CustomTools.__init__

    def init(self, *a, **k):
        original_init(self, *a, **k)
        w.root = self.DATA_DIR

    monkeypatch.setattr(CustomTools, "__init__", init)
    yield w
    worker_client.set_override(None)


@pytest.fixture
def sql_worker(monkeypatch, db, tmp_path):
    """SQL-Abfragen laufen gegen einen Worker im selben Prozess; SQLite-Dateien liegen in tmp_path/sqlite."""
    from app.toolworker import client as worker_client
    from tests import worker_support

    folder = tmp_path / "sqlite"
    folder.mkdir()
    worker_support.install_sql(monkeypatch, folder)
    yield folder
    worker_client.set_override(None, "sql")
