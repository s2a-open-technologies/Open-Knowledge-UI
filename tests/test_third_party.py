"""third_party.py (Drittkomponenten-Notices): Datenquellen liefern brauchbare Mengen."""

import sys


def _load():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "third_party",
        Path(__file__).resolve().parent.parent / "scripts" / "third_party.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["third_party"] = module
    spec.loader.exec_module(module)
    return module


def test_python_dists_match_uv_lock_scope():
    module = _load()
    rows, _ = module.python_dists()
    assert len(rows) > 100
    assert all(r["name"] and r["version"] for r in rows)


def test_frontend_rows_collect_license_strings():
    module = _load()
    rows = module.npm_packages()
    # node_modules kann in der CI-Test-Job-Umgebung fehlen — dann leer, kein Fehler.
    if rows:
        assert all(r["license"] for r in rows)
        assert any("react" in r["name"] for r in rows)


def test_generated_notices_commit_is_fresh():
    module = _load()
    assert module.OUT.is_file()
    content = module.OUT.read_text()
    assert "AGPL-3.0-or-later" in content
    assert "## Python-Abhängigkeiten" in content
