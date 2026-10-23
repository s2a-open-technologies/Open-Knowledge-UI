import pytest

from app.agent.toolworker import config


def test_defaults_are_seeded(db):
    load = config.load()
    assert load.url == "http://tool-worker:8090"
    assert (load.timeout_s, load.memory_mb, load.max_output_chars) == (30, 1024, 20000)


def test_save_validates_and_round_trips(db):
    config.save(
        {
            "url": "http://worker.lan:8090/",
            "timeout_s": 60,
            "memory_mb": 2048,
            "max_output_chars": 5000,
        }
    )
    v = config.view()
    assert v["url"] == "http://worker.lan:8090" and v["timeout_s"] == 60


@pytest.mark.parametrize(
    "bad",
    [
        {"url": "ftp://x"},
        {"url": "http://169.254.169.254"},
        {"url": ""},
        {"timeout_s": 0},
        {"timeout_s": 10_000},
        {"memory_mb": 10},
        {"max_output_chars": 5},
    ],
)
def test_invalid_values_are_rejected(db, bad):
    with pytest.raises(config.ConfigError):
        config.save(bad)


def test_token_comes_from_env_or_file(monkeypatch, tmp_path):
    monkeypatch.delenv("TOOL_WORKER_TOKEN", raising=False)
    assert config.token() == ""
    monkeypatch.setenv("TOOL_WORKER_TOKEN", "abc")
    assert config.token() == "abc"
    f = tmp_path / "tok"
    f.write_text("geheim\n")
    monkeypatch.setenv("TOOL_WORKER_TOKEN_FILE", str(f))
    assert config.token() == "geheim"


def test_sql_defaults_are_seeded(db):
    s = config.load_sql()
    assert (s.url, s.row_limit, s.timeout_s, s.max_cell_chars) == (
        "http://tool-worker-sql:8090",
        50,
        15,
        500,
    )
    assert "sql_query" in config.sql_prompt()


def test_sql_settings_save_and_validate(db):
    config.save(
        {
            "sql_url": "http://sql.lan:8090/",
            "sql_row_limit": 200,
            "sql_timeout_s": 30,
            "sql_max_cell_chars": 800,
        }
    )
    v = config.view()
    assert v["sql_url"] == "http://sql.lan:8090" and v["sql_row_limit"] == 200
    for bad in (
        {"sql_url": "ftp://x"},
        {"sql_row_limit": 0},
        {"sql_row_limit": 5000},
        {"sql_timeout_s": 999},
        {"sql_max_cell_chars": 1},
    ):
        with pytest.raises(config.ConfigError):
            config.save(bad)


def test_migrations_seed_the_same_keys():
    import importlib.util
    from pathlib import Path

    for rev, expected in (
        ("0024_tool_worker", config.PANDAS_SETTINGS),
        ("0025_sql_knowledge", config.SQL_SETTINGS),
        ("0028_sql_values", config.VALUES_SETTINGS),
    ):
        spec = importlib.util.spec_from_file_location(
            rev, Path(f"alembic/versions/{rev}.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert [k for k, *_ in mod.SETTINGS] == [k for k, *_ in expected]
        assert {k: v for k, v, *_ in mod.SETTINGS} == {k: v for k, v, *_ in expected}
