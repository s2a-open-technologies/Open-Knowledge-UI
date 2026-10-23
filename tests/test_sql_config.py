import json

import pytest

from app.knowledge.sql.config import (
    SqlConfigError,
    merge_password,
    parse_config,
    validate,
)


def cfg(**over):
    base = {
        "dialect": "postgresql",
        "host": "db.lan",
        "port": 5432,
        "database": "shop",
        "user": "ro",
        "password": "geheim",
    }
    base.update(over)
    return parse_config(base)


def test_parse_accepts_json_strings_and_fills_default_ports():
    c = parse_config(
        json.dumps({"dialect": "mysql", "host": "h", "database": "d", "user": "u"})
    )
    assert c.port == 3306 and c.instance == ""
    ms = parse_config(
        {
            "dialect": "mssql",
            "host": "h",
            "instance": "SQLEXPRESS",
            "database": "d",
            "user": "u",
        }
    )
    assert ms.port == 1433 and ms.instance == "SQLEXPRESS"


def test_public_never_contains_the_password():
    pub = cfg().public()
    assert "password" not in pub and pub["has_password"] is True
    assert "geheim" not in json.dumps(pub)
    assert cfg(password="").public()["has_password"] is False


def test_to_dict_round_trips_including_filters():
    c = cfg(tables=["items", "orders"], blocked_columns={"items": ["salary"]})
    assert parse_config(c.to_dict()) == c
    assert c.allowed_tables() == {"items", "orders"} and c.blocked() == {
        "items": {"salary"}
    }
    assert cfg().allowed_tables() is None


def test_connection_dict_is_what_the_worker_needs():
    assert set(cfg().connection()) == {
        "dialect",
        "host",
        "port",
        "database",
        "user",
        "password",
        "schema",
        "instance",
    }


@pytest.mark.parametrize(
    "over",
    [
        {"dialect": "oracle"},
        {"host": ""},
        {"host": "evil host;rm"},
        {"host": "169.254.169.254"},
        {"port": 70000},
        {"database": ""},
        {"user": ""},
        {"schema": "a;b"},
        {"dialect": "mssql", "instance": "a b;"},
        {"tables": ["ok", "bad name;"]},
    ],
)
def test_invalid_configs_are_refused_with_a_reason(over):
    with pytest.raises(SqlConfigError):
        validate(cfg(**over))


def test_sqlite_needs_only_a_relative_file_name():
    validate(parse_config({"dialect": "sqlite", "database": "shop.db"}))
    for bad in ("", "/etc/passwd", "../x.db", "a/../../x.db"):
        with pytest.raises(SqlConfigError):
            validate(parse_config({"dialect": "sqlite", "database": bad}))


def test_a_stored_password_is_only_reused_for_the_same_target():
    old = cfg()
    assert merge_password(cfg(password=""), old).password == "geheim"
    assert merge_password(cfg(password="neu"), old).password == "neu"
    for other in (
        {"host": "other.lan"},
        {"port": 5433},
        {"database": "x"},
        {"user": "admin"},
    ):
        assert merge_password(cfg(password="", **other), old).password == ""
