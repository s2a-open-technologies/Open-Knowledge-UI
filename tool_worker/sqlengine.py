"""Verbindung, Abfrage und Schema fuer SQL-Quellen. Laeuft nur im Tool-Worker (Rolle sql).

Jede Anfrage baut eine eigene Engine (kein Pool): Zugangsdaten bleiben nicht im Speicher des Servers.
"""

import datetime
import decimal
import re
import time
import uuid
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.engine import URL
from sqlalchemy.pool import NullPool
from sqlglot import exp

from tool_worker.sqlcheck import DIALECTS, SqlRejected, validate_select

DEFAULT_PORTS = {"postgresql": 5432, "mysql": 3306, "mssql": 1433}
SYSTEM_SCHEMAS = {
    "information_schema",
    "pg_catalog",
    "pg_toast",
    "sys",
    "mysql",
    "performance_schema",
}
WRITE_RE = re.compile(
    r"\b(ALL PRIVILEGES|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE)\b",
    re.IGNORECASE,
)
MAX_TABLES = 500


class SqlError(Exception):
    """Fehler mit lesbarem Text fuer Nutzer und Modell (keine Zugangsdaten, keine Adressen)."""


def _short(e: BaseException) -> str:
    inner = getattr(e, "orig", None) or e
    first = (str(inner).strip().splitlines() or [""])[0]
    return f"{type(inner).__name__}: {first}"[:300]


def _sqlite_path(conn: dict, sqlite_dir: str) -> Path:
    base = Path(sqlite_dir).resolve()
    name = str(conn.get("database") or "")
    if not name or name.startswith("/") or "\x00" in name:
        raise SqlError("SQLite-Datei nicht verfügbar")
    target = (base / name).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise SqlError("SQLite-Datei nicht verfügbar") from None
    if not target.is_file():
        raise SqlError("SQLite-Datei nicht verfügbar")
    return target


def make_engine(conn: dict, *, sqlite_dir: str, timeout_s: float) -> sa.Engine:
    dialect = conn.get("dialect")
    ms = int(timeout_s * 1000)
    if dialect == "sqlite":
        path = _sqlite_path(conn, sqlite_dir)
        engine = sa.create_engine(
            URL.create(
                "sqlite", database=f"file:{path}?mode=ro", query={"uri": "true"}
            ),
            poolclass=NullPool,
        )
        deadline = time.monotonic() + timeout_s

        @event.listens_for(engine, "connect")
        def _setup(dbapi, _record):
            dbapi.execute("PRAGMA query_only=ON")
            dbapi.set_progress_handler(
                lambda: 1 if time.monotonic() > deadline else 0, 10000
            )

        return engine
    common = {
        "username": conn.get("user") or None,
        "password": conn.get("password") or None,
        "host": conn.get("host") or None,
        "port": int(conn.get("port") or DEFAULT_PORTS.get(dialect, 0)) or None,
        "database": conn.get("database") or None,
    }
    if dialect == "postgresql":
        return sa.create_engine(
            URL.create("postgresql+psycopg", **common),
            poolclass=NullPool,
            connect_args={
                "connect_timeout": 10,
                "options": f"-c default_transaction_read_only=on -c statement_timeout={ms}",
            },
        )
    if dialect == "mysql":
        engine = sa.create_engine(
            URL.create("mysql+pymysql", query={"charset": "utf8mb4"}, **common),
            poolclass=NullPool,
            # read_timeout ist nur der Notnagel auf Socket-Ebene; die Abfrage selbst bricht der Server ab
            connect_args={
                "connect_timeout": 10,
                "read_timeout": max(1, int(timeout_s)) + 5,
            },
        )

        @event.listens_for(engine, "connect")
        def _mysql_setup(dbapi, _record):
            cur = dbapi.cursor()
            cur.execute("SET SESSION TRANSACTION READ ONLY")
            try:  # MariaDB (Sekunden)
                cur.execute(f"SET SESSION max_statement_time={float(timeout_s)}")
            except Exception:  # noqa: BLE001 - MySQL kennt die Variable nicht (Millisekunden, nur SELECT)
                cur.execute(f"SET SESSION max_execution_time={ms}")
            cur.close()

        return engine
    if dialect == "mssql":
        # pymssql (FreeTDS, wie in HUGO): kein Microsoft-ODBC-Treiber im Image noetig. Eine benannte
        # Instanz heisst "host\\instanz"; ihren Port liefert der SQL Browser (UDP 1434).
        instance = (conn.get("instance") or "").strip()
        server = f"{common['host']}\\{instance}" if instance else common["host"]
        port = common["port"]
        if instance and port in (None, 1433):
            port = None
        return sa.create_engine(
            URL.create(
                "mssql+pymssql",
                **{**common, "host": server, "port": port},
                query={"charset": "utf8"},
            ),
            poolclass=NullPool,
            connect_args={"login_timeout": 10, "timeout": max(1, int(timeout_s))},
        )
    raise SqlError("Unbekannter Datenbanktyp")


def _connect(engine: sa.Engine, detailed: bool = False):
    try:
        return engine.connect()
    except Exception as e:  # noqa: BLE001
        raise SqlError(
            _short(e) if detailed else "Verbindung zur Datenbank fehlgeschlagen"
        ) from None


def _cell(value, limit: int):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} Bytes>"
    if isinstance(value, uuid.UUID):
        return str(value)
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def run_query(
    conn: dict,
    sql: str,
    *,
    sqlite_dir: str,
    allowed_tables,
    blocked_columns,
    row_limit: int,
    timeout_s: float,
    max_cell_chars: int,
) -> dict:
    # eine Zeile mehr als das Limit: so erkennt fetchmany, dass gekuerzt wurde
    checked = validate_select(
        sql, conn.get("dialect", ""), allowed_tables, blocked_columns, row_limit + 1
    )
    engine = make_engine(conn, sqlite_dir=sqlite_dir, timeout_s=timeout_s)
    try:
        with _connect(engine) as c:
            try:
                # text() statt exec_driver_sql: "%" und ":" in Literalen bleiben woertlich
                result = c.execute(sa.text(checked.replace(":", r"\:")))
                columns = list(result.keys())
                rows = result.fetchmany(row_limit + 1)
            except Exception as e:  # noqa: BLE001
                raise SqlError(_short(e)) from None
    finally:
        engine.dispose()
    truncated = len(rows) > row_limit
    return {
        "columns": columns,
        "rows": [[_cell(v, max_cell_chars) for v in row] for row in rows[:row_limit]],
        "truncated": truncated,
    }


def read_schema(
    conn: dict,
    *,
    sqlite_dir: str,
    allowed_tables,
    blocked_columns,
    timeout_s: float,
    detailed: bool = False,
) -> list[dict]:
    schema = (conn.get("schema") or "").strip() or None
    if schema and schema.lower() in SYSTEM_SCHEMAS:
        raise SqlError("Dieses Schema ist nicht erlaubt")
    allowed = {t.lower() for t in allowed_tables} if allowed_tables else None
    blocked = {
        t.lower(): {c.lower() for c in cols}
        for t, cols in (blocked_columns or {}).items()
    }
    engine = make_engine(conn, sqlite_dir=sqlite_dir, timeout_s=timeout_s)
    try:
        with _connect(engine, detailed):
            pass
        try:
            insp = sa.inspect(engine)
            entries = [(n, "table") for n in insp.get_table_names(schema=schema)]
            entries += [(n, "view") for n in insp.get_view_names(schema=schema)]
            out = []
            for name, kind in sorted(entries)[:MAX_TABLES]:
                if allowed is not None and name.lower() not in allowed:
                    continue
                hidden = blocked.get(name.lower(), set())
                pk = set(
                    insp.get_pk_constraint(name, schema=schema).get(
                        "constrained_columns"
                    )
                    or []
                )
                cols = [
                    {
                        "name": c["name"],
                        "type": str(c.get("type")),
                        "nullable": bool(c.get("nullable", True)),
                        "pk": c["name"] in pk,
                        "comment": c.get("comment") or "",
                    }
                    for c in insp.get_columns(name, schema=schema)
                    if c["name"].lower() not in hidden
                ]
                fks = [
                    {
                        "columns": fk["constrained_columns"],
                        "ref_table": fk["referred_table"],
                        "ref_columns": fk["referred_columns"],
                    }
                    for fk in insp.get_foreign_keys(name, schema=schema)
                    if not hidden.intersection(
                        c.lower() for c in fk["constrained_columns"]
                    )
                ]
                try:
                    comment = (insp.get_table_comment(name, schema=schema) or {}).get(
                        "text"
                    ) or ""
                except NotImplementedError:
                    comment = ""
                out.append(
                    {
                        "name": name,
                        "kind": kind,
                        "comment": comment,
                        "columns": cols,
                        "foreign_keys": fks,
                    }
                )
            return out
        except SqlError:
            raise
        except Exception as e:  # noqa: BLE001
            raise SqlError(_short(e)) from None
    finally:
        engine.dispose()


def _scalar(c, sql: str):
    return c.execute(sa.text(sql)).scalar()


def _write_privilege(c, conn: dict) -> bool | None:
    dialect = conn.get("dialect")
    try:
        if dialect == "sqlite":
            return False
        if dialect == "postgresql":
            schema = (conn.get("schema") or "public").replace("'", "''")
            return bool(
                _scalar(
                    c,
                    "select coalesce(bool_or(has_table_privilege(current_user, quote_ident(schemaname)||'.'||quote_ident(tablename), "
                    f"'INSERT,UPDATE,DELETE,TRUNCATE')), false) from pg_tables where schemaname = '{schema}'",
                )
            )
        if dialect == "mysql":
            grants = [
                str(r[0]) for r in c.execute(sa.text("SHOW GRANTS FOR CURRENT_USER()"))
            ]
            return any(WRITE_RE.search(g.split(" ON ")[0]) for g in grants)
        if dialect == "mssql":
            return bool(
                _scalar(
                    c,
                    "select case when is_srvrolemember('sysadmin')=1 or is_member('db_owner')=1 or is_member('db_datawriter')=1 then 1 else 0 end",
                )
            )
    except Exception:  # noqa: BLE001
        return None
    return None


VERSION_SQL = {
    "sqlite": "select 'SQLite ' || sqlite_version()",
    "postgresql": "select version()",
    "mysql": "select version()",
    "mssql": "select @@VERSION",
}


def check_connection(conn: dict, *, sqlite_dir: str, timeout_s: float = 10) -> dict:
    engine = make_engine(conn, sqlite_dir=sqlite_dir, timeout_s=timeout_s)
    try:
        with _connect(engine, detailed=True) as c:
            try:
                version = str(_scalar(c, VERSION_SQL[conn["dialect"]]))
            except Exception as e:  # noqa: BLE001
                raise SqlError(_short(e)) from None
            privilege = _write_privilege(c, conn)
    finally:
        engine.dispose()
    tables = read_schema(
        conn,
        sqlite_dir=sqlite_dir,
        allowed_tables=None,
        blocked_columns=None,
        timeout_s=timeout_s,
        detailed=True,
    )
    return {
        "version": version.splitlines()[0][:200],
        "tables": [
            {"name": t["name"], "columns": [c["name"] for c in t["columns"]]}
            for t in tables
        ],
        "write_privilege": privilege,
    }


# --------------------------------------------------------------------------------------
# Wertverzeichnis: verschiedene Werte einer Textspalte und Profil der Textspalten
# --------------------------------------------------------------------------------------

TEXT_TYPE_MARKERS = ("char", "text", "string", "clob")
MAX_PROFILE_TABLES = 100
MAX_PROFILE_COLUMNS = 300


def _ident(conn: dict, name: str) -> str:
    return exp.to_identifier(name, quoted=True).sql(dialect=DIALECTS[conn["dialect"]])


def _qualified(conn: dict, table: str) -> str:
    schema = (conn.get("schema") or "").strip()
    return (
        f"{_ident(conn, schema)}.{_ident(conn, table)}"
        if schema
        else _ident(conn, table)
    )


def distinct_values(
    conn: dict,
    *,
    sqlite_dir: str,
    table: str,
    column: str,
    allowed_tables,
    blocked_columns,
    max_distinct: int,
    timeout_s: float,
    max_cell_chars: int,
) -> dict:
    """Die verschiedenen Werte einer Spalte mit Haeufigkeit (haeufigste zuerst, ohne NULL).

    Hat die Spalte mehr als ``max_distinct`` verschiedene Werte, kommt ``too_many=True`` ohne Werte zurueck
    (dann ist es kein Kategoriefeld)."""
    c, t = _ident(conn, column), _qualified(conn, table)
    sql = f"SELECT {c} AS v, COUNT(*) AS n FROM {t} WHERE {c} IS NOT NULL GROUP BY {c} ORDER BY n DESC"
    out = run_query(
        conn,
        sql,
        sqlite_dir=sqlite_dir,
        allowed_tables=allowed_tables,
        blocked_columns=blocked_columns,
        row_limit=max_distinct + 1,
        timeout_s=timeout_s,
        max_cell_chars=max_cell_chars,
    )
    if len(out["rows"]) > max_distinct:
        return {"values": [], "too_many": True}
    return {
        "values": [
            [str(v), int(n)] for v, n in out["rows"] if v is not None and str(v).strip()
        ],
        "too_many": False,
    }


def profile_columns(
    conn: dict, *, sqlite_dir: str, allowed_tables, blocked_columns, timeout_s: float
) -> list[dict]:
    """Textspalten der freigegebenen Tabellen mit Zeilen-, Nicht-NULL- und Distinct-Zahl (fuer den Vorschlag)."""
    tables = read_schema(
        conn,
        sqlite_dir=sqlite_dir,
        allowed_tables=allowed_tables,
        blocked_columns=blocked_columns,
        timeout_s=timeout_s,
    )[:MAX_PROFILE_TABLES]
    result: list[dict] = []
    budget = MAX_PROFILE_COLUMNS
    for tbl in tables:
        cols = [
            c
            for c in tbl["columns"]
            if any(m in c["type"].lower() for m in TEXT_TYPE_MARKERS)
        ][:budget]
        if not cols:
            continue
        budget -= len(cols)
        parts = ["COUNT(*) AS r"]
        for i, col in enumerate(cols):
            q = _ident(conn, col["name"])
            parts += [f"COUNT(DISTINCT {q}) AS d{i}", f"COUNT({q}) AS n{i}"]
        sql = f"SELECT {', '.join(parts)} FROM {_qualified(conn, tbl['name'])}"
        try:
            row = run_query(
                conn,
                sql,
                sqlite_dir=sqlite_dir,
                allowed_tables=allowed_tables,
                blocked_columns=blocked_columns,
                row_limit=1,
                timeout_s=timeout_s,
                max_cell_chars=50,
            )["rows"][0]
        except (SqlRejected, SqlError):
            continue  # eine Tabelle, die nicht zaehlbar ist, fehlt im Vorschlag
        for i, col in enumerate(cols):
            result.append(
                {
                    "table": tbl["name"],
                    "column": col["name"],
                    "type": col["type"],
                    "rows": int(row[0]),
                    "distinct": int(row[1 + 2 * i]),
                    "non_null": int(row[2 + 2 * i]),
                }
            )
        if budget <= 0:
            break
    return result
