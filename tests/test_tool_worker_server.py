import os
import sqlite3

import pytest

from tool_worker.server import create_app

TOKEN = "t" * 32


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    (d / "knowledge_2").mkdir(parents=True)
    (d / "umsatz.csv").write_text("monat,betrag\njan,100\nfeb,250\n")
    (d / "knowledge_2" / "x.csv").write_text("a\n1\n")
    (d / "notiz.txt").write_text("hallo")
    (tmp_path / "ausserhalb.csv").write_text("x\n1\n")
    return d


@pytest.fixture
def client(data):
    return create_app(TOKEN, str(data)).test_client()


def call(client, body, token=TOKEN):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post("/v1/pandas", json=body, headers=headers)


def body(file="umsatz.csv", code="result = df['betrag'].sum()", **limits):
    return {
        "file": file,
        "code": code,
        "limits": {
            "timeout_s": 20,
            "memory_mb": 1024,
            "max_output_chars": 2000,
            **limits,
        },
    }


def test_empty_token_refuses_to_start(data):
    with pytest.raises(ValueError):
        create_app("", str(data))


def test_health_needs_no_token(client):
    assert client.get("/health").get_json()["status"] == "ok"


@pytest.mark.parametrize("token", [None, "falsch", "t" * 31])
def test_wrong_or_missing_token_is_401(client, token):
    assert call(client, body(), token).status_code == 401


def test_runs_code_on_an_allowed_file(client):
    r = call(client, body())
    assert (
        r.status_code == 200
        and r.get_json()["ok"] is True
        and "350" in r.get_json()["output"]
    )


def test_subfolder_files_are_reachable_by_relative_path(client):
    assert (
        call(client, body("knowledge_2/x.csv", "result = df['a'].sum()")).status_code
        == 200
    )


@pytest.mark.parametrize(
    "file",
    [
        "../ausserhalb.csv",
        "/etc/passwd",
        "knowledge_2/../../ausserhalb.csv",
        "nicht_da.csv",
        "notiz.txt",
        "",
        "a\x00b.csv",
    ],
)
def test_paths_outside_or_unsupported_are_refused(client, file):
    assert call(client, body(file)).status_code in (400, 404)


def test_symlink_out_of_the_mount_is_refused(client, data, tmp_path):
    os.symlink(tmp_path / "ausserhalb.csv", data / "link.csv")
    assert call(client, body("link.csv")).status_code == 404


def test_blocked_code_returns_ok_false_with_message(client):
    j = call(client, body(code="import os")).get_json()
    assert j["ok"] is False and "blockiert" in j["error"]


def test_oversized_code_is_413(client):
    assert call(client, body(code="x=1\n" * 10000)).status_code == 413


@pytest.mark.parametrize(
    "bad",
    [{}, {"file": 1, "code": "x"}, {"file": "a.csv"}, {"file": "a.csv", "code": 3}],
)
def test_invalid_bodies_are_400(client, bad):
    assert call(client, bad).status_code == 400


def test_limits_are_capped_by_the_worker(data):
    seen = {}

    def fake(path, code, **kw):
        seen.update(kw)
        return {"ok": True, "output": "x"}

    c = create_app(
        TOKEN,
        str(data),
        runner=fake,
        max_timeout_s=30,
        max_memory_mb=2048,
        max_output_chars=5000,
    ).test_client()
    call(c, body(timeout_s=9999, memory_mb=99999, max_output_chars=10**9))
    assert seen == {"timeout_s": 30, "memory_mb": 2048, "max_output_chars": 5000}


def test_errors_do_not_leak_internals(data):
    def boom(*a, **k):
        raise RuntimeError("/geheimer/pfad")

    c = create_app(TOKEN, str(data), runner=boom).test_client()
    r = call(c, body())
    assert r.status_code == 500 and "geheimer" not in r.get_data(as_text=True)


# --- Rolle sql -----------------------------------------------------------------


@pytest.fixture
def sqldir(tmp_path):
    d = tmp_path / "sqlite"
    d.mkdir()
    con = sqlite3.connect(d / "shop.db")
    con.executescript(
        "create table items (id integer primary key, name text, salary integer); insert into items (name, salary) values ('Stuhl', 5), ('Tisch', 7);"
    )
    con.commit()
    con.close()
    return d


@pytest.fixture
def sqlc(data, sqldir):
    return create_app(
        TOKEN, str(data), role="sql", sqlite_dir=str(sqldir)
    ).test_client()


def sql_call(client, action, body, token=TOKEN):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(f"/v1/sql/{action}", json=body, headers=headers)


def sql_body(**over):
    body = {
        "connection": {"dialect": "sqlite", "database": "shop.db"},
        "sql": "select name from items order by id",
        "allowed_tables": None,
        "blocked_columns": {},
        "limits": {"row_limit": 10, "timeout_s": 5, "max_cell_chars": 100},
    }
    body.update(over)
    return body


def test_roles_only_serve_their_own_endpoints(client, sqlc):
    assert (
        sql_call(client, "query", sql_body()).status_code == 404
    )  # pandas-Rolle kennt kein SQL
    assert call(sqlc, body()).status_code == 404  # sql-Rolle kennt kein pandas
    assert sqlc.get("/health").get_json()["role"] == "sql"


def test_sql_endpoints_need_the_token(sqlc):
    for token in (None, "falsch"):
        assert sql_call(sqlc, "query", sql_body(), token).status_code == 401


def test_sql_query_returns_rows(sqlc):
    j = sql_call(sqlc, "query", sql_body()).get_json()
    assert j == {
        "ok": True,
        "columns": ["name"],
        "rows": [["Stuhl"], ["Tisch"]],
        "truncated": False,
    }


def test_sql_rejections_and_db_errors_are_ok_false_with_text(sqlc):
    j = sql_call(sqlc, "query", sql_body(sql="delete from items")).get_json()
    assert j["ok"] is False and "SELECT" in j["error"]
    j = sql_call(sqlc, "query", sql_body(sql="select nope from items")).get_json()
    assert j["ok"] is False and "nope" in j["error"]


def test_blocked_columns_reach_the_engine(sqlc):
    j = sql_call(
        sqlc,
        "query",
        sql_body(sql="select salary from items", blocked_columns={"items": ["salary"]}),
    ).get_json()
    assert j["ok"] is False and "gesperrt" in j["error"]


def test_sql_schema_and_test(sqlc):
    s = sql_call(
        sqlc,
        "schema",
        {
            "connection": sql_body()["connection"],
            "allowed_tables": None,
            "blocked_columns": {"items": ["salary"]},
        },
    ).get_json()
    assert [c["name"] for c in s["tables"][0]["columns"]] == ["id", "name"]
    t = sql_call(sqlc, "test", {"connection": sql_body()["connection"]}).get_json()
    assert (
        t["ok"] is True
        and t["write_privilege"] is False
        and t["tables"][0]["name"] == "items"
    )


def test_sql_limits_are_capped_by_the_worker(data, sqldir):
    c = create_app(
        TOKEN, str(data), role="sql", sqlite_dir=str(sqldir), max_rows=1
    ).test_client()
    j = sql_call(
        c,
        "query",
        sql_body(limits={"row_limit": 9999, "timeout_s": 5, "max_cell_chars": 100}),
    ).get_json()
    assert len(j["rows"]) == 1 and j["truncated"] is True


@pytest.mark.parametrize("bad", [{}, {"connection": "x"}, {"connection": {}, "sql": 3}])
def test_sql_invalid_bodies_are_400(sqlc, bad):
    assert sql_call(sqlc, "query", bad).status_code == 400


def test_sql_errors_never_contain_the_password(sqlc):
    body = sql_body(
        connection={
            "dialect": "postgresql",
            "host": "127.0.0.1",
            "port": 1,
            "database": "d",
            "user": "u",
            "password": "geheim123",
        }
    )
    text = sql_call(sqlc, "query", body).get_data(as_text=True)
    assert "geheim123" not in text and "127.0.0.1" not in text


# --- Wertverzeichnis: distinct und profile --------------------------------------------------------
@pytest.fixture
def sqlc_goods(data, sqldir):
    con = sqlite3.connect(sqldir / "goods.db")
    con.executescript(
        "create table items (id integer primary key, name text, salary integer);"
    )
    con.executemany(
        "insert into items (name, salary) values (?, ?)",
        [("Stuhl", 1), ("Stuhl", 2), ("Tisch", 3)],
    )
    con.commit()
    con.close()
    return create_app(
        TOKEN, str(data), role="sql", sqlite_dir=str(sqldir), max_distinct=2
    ).test_client()


def goods_body(**over):
    body = {
        "connection": {"dialect": "sqlite", "database": "goods.db"},
        "table": "items",
        "column": "name",
        "allowed_tables": None,
        "blocked_columns": {},
        "max_distinct": 100,
        "limits": {"timeout_s": 5, "max_cell_chars": 100},
    }
    body.update(over)
    return body


def test_distinct_endpoint_returns_values_and_caps_max_distinct(sqlc_goods):
    j = sql_call(sqlc_goods, "distinct", goods_body()).get_json()
    assert j["ok"] and j["values"][0] == ["Stuhl", 2] and j["too_many"] is False
    capped = sql_call(
        sqlc_goods, "distinct", goods_body(column="id", max_distinct=99999)
    ).get_json()
    assert (
        capped["too_many"] is True
    )  # Obergrenze des Workers (2) gilt, nicht die der Anfrage


def test_distinct_endpoint_validates_and_enforces_filters(sqlc_goods):
    assert (
        sql_call(
            sqlc_goods, "distinct", {"connection": goods_body()["connection"]}
        ).status_code
        == 400
    )
    j = sql_call(
        sqlc_goods,
        "distinct",
        goods_body(column="salary", blocked_columns={"items": ["salary"]}),
    ).get_json()
    assert j["ok"] is False and "gesperrt" in j["error"]


def test_profile_endpoint_lists_text_columns(sqlc_goods):
    j = sql_call(
        sqlc_goods,
        "profile",
        {
            "connection": goods_body()["connection"],
            "allowed_tables": None,
            "blocked_columns": {},
        },
    ).get_json()
    assert j["ok"] and [(c["table"], c["column"]) for c in j["columns"]] == [
        ("items", "name")
    ]
