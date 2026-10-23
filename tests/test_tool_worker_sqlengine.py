import os
import sqlite3

import pytest

from tool_worker import sqlengine
from tool_worker.sqlcheck import SqlRejected


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        create table items (id integer primary key, name text not null, category text, salary integer);
        create table orders (id integer primary key, item_id integer references items(id), note text);
        insert into items (name, category, salary) values ('Stuhl','Möbel',1),('Tisch','Möbel',2),('Lampe','Licht',3);
        insert into orders (item_id, note) values (1,'a'),(1,'b'),(2,'c');
        """
    )
    for i in range(300):
        con.execute("insert into orders (item_id, note) values (3, ?)", ("x" * 2000,))
    con.commit()
    con.close()
    return tmp_path


def conn(db_file="shop.db"):
    return {"dialect": "sqlite", "database": db_file}


def kw(db, **extra):
    return dict(
        sqlite_dir=str(db),
        allowed_tables=None,
        blocked_columns=None,
        row_limit=50,
        timeout_s=5,
        max_cell_chars=100,
        **extra,
    )


def test_select_returns_columns_and_rows(db):
    out = sqlengine.run_query(
        conn(), "select name, category from items order by id", **kw(db)
    )
    assert out["columns"] == ["name", "category"]
    assert out["rows"][0] == ["Stuhl", "Möbel"] and out["truncated"] is False


def test_row_limit_truncates_and_says_so(db):
    out = sqlengine.run_query(
        conn(), "select id from orders", **{**kw(db), "row_limit": 10}
    )
    assert len(out["rows"]) == 10 and out["truncated"] is True


def test_union_without_limit_is_cut_by_fetchmany(db):
    out = sqlengine.run_query(
        conn(),
        "select id from orders union all select id from orders",
        **{**kw(db), "row_limit": 7},
    )
    assert len(out["rows"]) == 7 and out["truncated"] is True


def test_long_cells_are_shortened(db):
    out = sqlengine.run_query(
        conn(), "select note from orders where item_id = 3 limit 1", **kw(db)
    )
    assert len(out["rows"][0][0]) <= 101 and out["rows"][0][0].endswith("…")


def test_colons_and_percent_signs_in_sql_are_literal(db):
    out = sqlengine.run_query(
        conn(), "select '12:30' as t, name from items where name like '%tu%'", **kw(db)
    )
    assert out["rows"] == [["12:30", "Stuhl"]]


def test_writes_are_rejected_before_the_database(db):
    with pytest.raises(SqlRejected):
        sqlengine.run_query(conn(), "delete from items", **kw(db))


def test_sqlite_is_opened_read_only_even_if_validation_were_bypassed(db):
    engine = sqlengine.make_engine(conn(), sqlite_dir=str(db), timeout_s=5)
    with engine.connect() as c, pytest.raises(Exception):
        c.exec_driver_sql("delete from items")


def test_blocked_columns_and_tables_are_enforced(db):
    with pytest.raises(SqlRejected):
        sqlengine.run_query(
            conn(),
            "select salary from items",
            **{**kw(db), "blocked_columns": {"items": {"salary"}}},
        )
    with pytest.raises(SqlRejected):
        sqlengine.run_query(
            conn(), "select * from orders", **{**kw(db), "allowed_tables": {"items"}}
        )


def test_database_errors_are_short_and_without_paths(db):
    with pytest.raises(sqlengine.SqlError) as e:
        sqlengine.run_query(conn(), "select nope from items", **kw(db))
    assert "nope" in str(e.value) and str(db) not in str(e.value)


def test_sqlite_path_cannot_leave_the_mounted_folder(db):
    for name in ("../shop.db", "/etc/passwd", "missing.db"):
        with pytest.raises(sqlengine.SqlError):
            sqlengine.run_query(conn(name), "select 1", **kw(db))


def test_endless_query_hits_the_timeout(db):
    sql = "with recursive r(n) as (select 1 union all select n+1 from r) select count(*) from r"
    with pytest.raises(sqlengine.SqlError):
        sqlengine.run_query(conn(), sql, **{**kw(db), "timeout_s": 1})


def test_schema_lists_tables_columns_keys_and_respects_filters(db):
    schema = {
        t["name"]: t
        for t in sqlengine.read_schema(
            conn(),
            sqlite_dir=str(db),
            allowed_tables=None,
            blocked_columns=None,
            timeout_s=5,
        )
    }
    assert set(schema) == {"items", "orders"}
    assert [c["name"] for c in schema["items"]["columns"]] == [
        "id",
        "name",
        "category",
        "salary",
    ]
    assert schema["items"]["columns"][0]["pk"] is True
    assert schema["orders"]["foreign_keys"][0]["ref_table"] == "items"
    only = sqlengine.read_schema(
        conn(),
        sqlite_dir=str(db),
        allowed_tables={"items"},
        blocked_columns={"items": {"salary"}},
        timeout_s=5,
    )
    assert [t["name"] for t in only] == ["items"] and "salary" not in [
        c["name"] for c in only[0]["columns"]
    ]


def check_connection_test_reports_version_tables_and_no_write_privilege(db):
    out = sqlengine.check_connection(conn(), sqlite_dir=str(db))
    assert out["version"].startswith("SQLite") and out["write_privilege"] is False
    assert {"name": "items", "columns": ["id", "name", "category", "salary"]} in out[
        "tables"
    ]


def test_urls_for_the_other_databases_are_built_read_only_without_connecting():
    pg = sqlengine.make_engine(
        {
            "dialect": "postgresql",
            "host": "h",
            "port": 5432,
            "database": "d",
            "user": "u",
            "password": "p",
        },
        sqlite_dir="/x",
        timeout_s=7,
    )
    assert pg.url.drivername == "postgresql+psycopg" and pg.url.password == "p"
    my = sqlengine.make_engine(
        {
            "dialect": "mysql",
            "host": "h",
            "database": "d",
            "user": "u",
            "password": "p",
        },
        sqlite_dir="/x",
        timeout_s=7,
    )
    assert my.url.drivername == "mysql+pymysql" and my.url.port == 3306
    ms = sqlengine.make_engine(
        {
            "dialect": "mssql",
            "host": "h",
            "port": 1433,
            "database": "d",
            "user": "u",
            "password": "p",
        },
        sqlite_dir="/x",
        timeout_s=7,
    )
    assert (
        ms.url.drivername == "mssql+pymssql"
        and ms.url.host == "h"
        and ms.url.port == 1433
    )
    inst = sqlengine.make_engine(
        {
            "dialect": "mssql",
            "host": "h",
            "port": 1433,
            "instance": "SQLEXPRESS",
            "database": "d",
            "user": "u",
            "password": "p",
        },
        sqlite_dir="/x",
        timeout_s=7,
    )
    assert (
        inst.url.host == "h\\SQLEXPRESS" and inst.url.port is None
    )  # benannte Instanz: Port ueber den SQL Browser
    odd = sqlengine.make_engine(
        {
            "dialect": "mssql",
            "host": "h",
            "port": 1500,
            "instance": "I",
            "database": "d",
            "user": "u",
            "password": "p",
        },
        sqlite_dir="/x",
        timeout_s=7,
    )
    assert odd.url.port == 1500  # ein ausdruecklich abweichender Port gilt weiter
    with pytest.raises(sqlengine.SqlError):
        sqlengine.make_engine({"dialect": "oracle"}, sqlite_dir="/x", timeout_s=7)


def test_unreachable_host_gives_a_generic_message_without_the_address():
    with pytest.raises(sqlengine.SqlError) as e:
        sqlengine.run_query(
            {
                "dialect": "postgresql",
                "host": "127.0.0.1",
                "port": 1,
                "database": "d",
                "user": "u",
                "password": "geheim",
            },
            "select 1",
            sqlite_dir="/x",
            allowed_tables=None,
            blocked_columns=None,
            row_limit=5,
            timeout_s=2,
            max_cell_chars=50,
        )
    assert "127.0.0.1" not in str(e.value) and "geheim" not in str(e.value)


# --- Echte Server (nur wenn per Umgebung angegeben: "host:port:db:user:password") -------------------
REAL = {
    "postgresql": os.getenv("SQL_TEST_PG"),
    "mysql": os.getenv("SQL_TEST_MYSQL"),
    "mssql": os.getenv("SQL_TEST_MSSQL"),
}
SLOW = {
    "postgresql": "with recursive r(n) as (select 1 union all select n+1 from r) select count(*) from r",
    "mysql": "select count(*) from seq_1_to_100000000 a, seq_1_to_100",  # MariaDB-Sequence-Engine
    "mssql": "with r(n) as (select 1 union all select n+1 from r where n < 2000000000) select count(*) from r option (maxrecursion 0)",
}


def real_conn(dialect):
    host, port, db, user, pw = REAL[dialect].split(":")
    conn = {
        "dialect": dialect,
        "host": host,
        "port": int(port),
        "database": db,
        "user": user,
        "password": pw,
    }
    if os.getenv("SQL_TEST_MSSQL_INSTANCE") and dialect == "mssql":
        conn["instance"] = os.environ["SQL_TEST_MSSQL_INSTANCE"]
    return conn


@pytest.mark.parametrize("dialect", ["postgresql", "mysql", "mssql"])
def test_real_server_roundtrip_readonly_and_limits(dialect):
    if not REAL[dialect]:
        pytest.skip(f"SQL_TEST_{dialect.upper()} nicht gesetzt")
    conn = real_conn(dialect)
    kw = {
        "sqlite_dir": "/x",
        "allowed_tables": None,
        "blocked_columns": None,
        "row_limit": 5,
        "timeout_s": 5,
        "max_cell_chars": 50,
    }
    assert sqlengine.run_query(conn, "select 'a:b' as c", **kw)["rows"][0][0] == "a:b"
    like = sqlengine.run_query(
        conn, "select case when 'xyx' like '%y%' then 1 else 0 end as m", **kw
    )
    assert like["rows"][0][0] == 1  # "%" bleibt wörtlich
    seven = "select n from (select 1 as n union all select 2 union all select 3 union all select 4 union all select 5 union all select 6 union all select 7) t"
    big = sqlengine.run_query(conn, seven, **kw)
    assert len(big["rows"]) == 5 and big["truncated"] is True
    info = sqlengine.check_connection(conn, sqlite_dir="/x")
    assert info["version"] and isinstance(info["tables"], list)
    with pytest.raises(sqlengine.SqlError):
        sqlengine.run_query(
            conn, SLOW[dialect], **{**kw, "timeout_s": 1}
        )  # Timeout greift


@pytest.mark.parametrize("dialect", ["postgresql", "mysql"])
def test_real_session_is_read_only_even_if_the_check_were_bypassed(dialect):
    if not REAL[dialect]:
        pytest.skip(f"SQL_TEST_{dialect.upper()} nicht gesetzt")
    engine = sqlengine.make_engine(real_conn(dialect), sqlite_dir="/x", timeout_s=5)
    with engine.connect() as c, pytest.raises(Exception):
        c.exec_driver_sql("create table schreibtest_x (a int)")


def test_real_wrong_password_gives_a_generic_message_without_the_address():
    if not REAL["postgresql"]:
        pytest.skip("SQL_TEST_POSTGRESQL nicht gesetzt")
    conn = {**real_conn("postgresql"), "password": "falsch"}
    with pytest.raises(sqlengine.SqlError) as e:
        sqlengine.run_query(
            conn,
            "select 1",
            sqlite_dir="/x",
            allowed_tables=None,
            blocked_columns=None,
            row_limit=5,
            timeout_s=2,
            max_cell_chars=50,
        )
    assert "localhost" not in str(e.value) and "falsch" not in str(e.value)


# --- Wertverzeichnis: Werte und Profil ------------------------------------------------------------
@pytest.fixture
def goods(tmp_path):
    path = tmp_path / "goods.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        create table items (id integer primary key, name text, category text, email text, price real, salary integer);
        """
    )
    rows = [
        ("Stuhl", "Möbel", "a@x.de", 10.0, 1),
        ("Stuhl", "Möbel", "b@x.de", 11.0, 2),
        ("Bürostuhl", "Möbel", "c@x.de", 99.0, 3),
        ("Lampe", "Licht", "d@x.de", 5.0, 4),
        (None, "Licht", "e@x.de", 6.0, 5),
    ]
    con.executemany(
        "insert into items (name, category, email, price, salary) values (?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()
    return tmp_path


def dv(goods, **over):
    args = {
        "sqlite_dir": str(goods),
        "table": "items",
        "column": "name",
        "allowed_tables": None,
        "blocked_columns": None,
        "max_distinct": 100,
        "timeout_s": 5,
        "max_cell_chars": 100,
    }
    args.update(over)
    return sqlengine.distinct_values(
        {"dialect": "sqlite", "database": "goods.db"}, **args
    )


def test_distinct_values_come_with_counts_most_frequent_first_without_nulls(goods):
    out = dv(goods)
    assert out["values"][0] == ["Stuhl", 2]
    assert {v for v, _ in out["values"]} == {"Stuhl", "Bürostuhl", "Lampe"} and out[
        "too_many"
    ] is False


def test_too_many_distinct_values_are_reported_not_returned(goods):
    out = dv(goods, column="email", max_distinct=3)
    assert out["too_many"] is True and out["values"] == []


def test_distinct_respects_allowed_tables_and_blocked_columns(goods):
    with pytest.raises(SqlRejected):
        dv(goods, allowed_tables={"andere"})
    with pytest.raises(SqlRejected):
        dv(goods, column="salary", blocked_columns={"items": {"salary"}})


def test_odd_identifiers_are_quoted_not_injected(goods):
    try:
        out = dv(goods, column='name"; drop table items; --')
    except (sqlengine.SqlError, SqlRejected):
        out = None  # abgelehnt ist gut; SQLite macht aus unbekannten "Namen" ein Literal, auch das ist harmlos
    if out is not None:
        assert all("drop" in v for v, _ in out["values"])
    assert dv(goods)["values"]  # die Tabelle ist noch da


def test_profile_lists_text_columns_with_distinct_counts_and_skips_blocked_and_numeric(
    goods,
):
    cols = sqlengine.profile_columns(
        {"dialect": "sqlite", "database": "goods.db"},
        sqlite_dir=str(goods),
        allowed_tables=None,
        blocked_columns={"items": {"email"}},
        timeout_s=5,
    )
    by = {(c["table"], c["column"]): c for c in cols}
    assert set(by) == {
        ("items", "name"),
        ("items", "category"),
    }  # email gesperrt, price/salary/id nicht Text
    assert (
        by[("items", "name")]["distinct"] == 3
        and by[("items", "name")]["non_null"] == 4
    )
    assert (
        by[("items", "category")]["distinct"] == 2
        and by[("items", "category")]["rows"] == 5
    )
