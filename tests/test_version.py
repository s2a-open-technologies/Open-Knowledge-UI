import tomllib
from pathlib import Path

from app.api import register_blueprints
from app.core import version


def test_version_matches_pyproject():
    expected = tomllib.loads(
        (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    )["project"]["version"]
    assert version.get_version() == expected


def test_falls_back_to_env_then_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(version, "PYPROJECT", tmp_path / "fehlt.toml")
    version.get_version.cache_clear()
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    assert version.get_version() == "9.9.9"
    version.get_version.cache_clear()
    monkeypatch.delenv("APP_VERSION")
    assert version.get_version() == "unbekannt"
    version.get_version.cache_clear()


def test_version_endpoint_is_public_and_health_reports_it():
    from flask import Flask

    app = Flask(__name__)
    register_blueprints(app)
    c = app.test_client()
    assert c.get("/api/version").json == {"version": version.get_version()}
    assert c.get("/health").json["version"] == version.get_version()
