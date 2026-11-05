"""HTTP-Schnittstelle des Tool-Workers (siehe docs/tool-worker-api.md)."""

import hmac
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StrictStr

from tool_worker import __version__, sandbox
from tool_worker.frames import SUFFIXES

log = logging.getLogger("tool_worker")
MAX_CODE_CHARS = 20_000
MAX_FILE_CHARS = 512
MAX_BODY_BYTES = 1024 * 1024


class PandasRequest(BaseModel):
    file: StrictStr
    code: StrictStr
    limits: Any = None


class SqlRequest(BaseModel):
    connection: dict[str, Any]
    sql: Any = None
    table: Any = None
    column: Any = None
    allowed_tables: Any = None
    blocked_columns: Any = None
    limits: Any = None
    max_distinct: Any = None


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


class _Guard:
    """Rolle, Token und Groesse pruefen, bevor eine Anfrage einen Handler erreicht (ASGI)."""

    def __init__(self, app, role: str, expected: bytes):
        self.app, self.role, self.expected = app, role, expected

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/v1/"):
            return await self.app(scope, receive, send)
        path = scope["path"]
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope["headers"]
        }
        # pandas-Rolle: nur /v1/pandas; sql-Rolle: nur /v1/sql/*
        if (self.role == "pandas") == path.startswith("/v1/sql/"):
            return await _error("Nicht gefunden", 404)(scope, receive, send)
        sent = headers.get("authorization", "")
        given = sent[7:].encode() if sent.startswith("Bearer ") else b""
        if not hmac.compare_digest(given, self.expected):
            return await _error("Nicht autorisiert", 401)(scope, receive, send)
        try:
            declared = int(headers.get("content-length") or 0)
        except ValueError:
            declared = 0
        if declared > MAX_BODY_BYTES:
            return await _error("Anfrage zu groß", 413)(scope, receive, send)

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_BODY_BYTES:
                    raise ValueError("Anfrage zu groß")
            return message

        await self.app(scope, limited_receive, send)


def create_app(
    token: str,
    data_dir: "str | Callable[[], str]",
    *,
    runner=sandbox.run_job,
    max_timeout_s: float = 120,
    max_memory_mb: int = 4096,
    max_output_chars: int = 200_000,
    role: str = "pandas",
    sqlite_dir: str = "/sqlite",
    max_rows: int = 1000,
    max_sql_timeout_s: float = 60,
    max_distinct: int = 100_000,
) -> FastAPI:
    if role not in ("pandas", "sql"):
        raise ValueError("TOOL_WORKER_ROLE muss pandas oder sql sein")
    if not token:
        raise ValueError("TOOL_WORKER_TOKEN ist nicht gesetzt")
    app = FastAPI(
        title="Open Knowledge UI Tool-Worker",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(_Guard, role=role, expected=token.encode())

    @app.exception_handler(RequestValidationError)
    async def invalid(_request, _exc):
        # Das Protokoll kennt nur diese eine Meldung (der App-Client wertet sie aus)
        return _error("Ungültige Anfrage", 400)

    def root() -> Path:
        return Path(data_dir() if callable(data_dir) else data_dir).resolve()

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "role": role}

    def resolve(file: str) -> Path | None:
        if (
            not file
            or "\x00" in file
            or len(file) > MAX_FILE_CHARS
            or file.startswith("/")
        ):
            return None
        base = root()
        target = (base / file).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            return None
        if target.suffix.lower() not in SUFFIXES or not target.is_file():
            return None
        return target

    def capper(limits):
        limits = limits if isinstance(limits, dict) else {}

        def cap(name, default, ceiling):
            try:
                value = float(limits.get(name, default))
            except (TypeError, ValueError):
                value = default
            return min(max(value, 1), ceiling)

        return cap

    @app.post("/v1/pandas")
    def pandas_run(data: PandasRequest):
        if len(data.code) > MAX_CODE_CHARS:
            return _error("Code zu lang", 413)
        cap = capper(data.limits)
        target = resolve(data.file)
        if target is None:
            return _error("Datei nicht verfügbar", 404)
        try:
            out = runner(
                str(target),
                data.code,
                timeout_s=cap("timeout_s", 30, max_timeout_s),
                memory_mb=int(cap("memory_mb", 1024, max_memory_mb)),
                max_output_chars=int(cap("max_output_chars", 20_000, max_output_chars)),
            )
        except Exception:
            log.exception("pandas-Lauf fehlgeschlagen")
            return _error("Interner Fehler", 500)
        return out

    def sql_caps(data: SqlRequest):
        cap = capper(data.limits)
        return {
            "row_limit": int(cap("row_limit", 50, max_rows)),
            "timeout_s": cap("timeout_s", 15, max_sql_timeout_s),
            "max_cell_chars": int(cap("max_cell_chars", 500, 5000)),
        }

    def sql_filters(data: SqlRequest):
        tables = data.allowed_tables
        blocked = data.blocked_columns if isinstance(data.blocked_columns, dict) else {}
        return (
            {str(t) for t in tables} if isinstance(tables, list) else None,
            {
                str(t): {str(c) for c in (cols or [])}
                for t, cols in blocked.items()
                if isinstance(cols, list)
            },
        )

    def sql_run(action: str, data: SqlRequest):
        from tool_worker import sqlengine
        from tool_worker.sqlcheck import SqlRejected

        if action == "query" and not isinstance(data.sql, str):
            return _error("Ungültige Anfrage", 400)
        if action == "distinct" and not all(
            isinstance(v, str) and v for v in (data.table, data.column)
        ):
            return _error("Ungültige Anfrage", 400)
        caps = sql_caps(data)
        allowed, blocked = sql_filters(data)
        conn = data.connection
        try:
            if action == "query":
                result = sqlengine.run_query(
                    conn,
                    data.sql,
                    sqlite_dir=sqlite_dir,
                    allowed_tables=allowed,
                    blocked_columns=blocked,
                    **caps,
                )
            elif action == "schema":
                result = {
                    "tables": sqlengine.read_schema(
                        conn,
                        sqlite_dir=sqlite_dir,
                        allowed_tables=allowed,
                        blocked_columns=blocked,
                        timeout_s=caps["timeout_s"],
                    )
                }
            elif action == "distinct":
                try:
                    wanted = int(data.max_distinct or 5000)
                except (TypeError, ValueError):
                    wanted = 5000
                result = sqlengine.distinct_values(
                    conn,
                    sqlite_dir=sqlite_dir,
                    table=data.table,
                    column=data.column,
                    allowed_tables=allowed,
                    blocked_columns=blocked,
                    max_distinct=min(max(wanted, 1), max_distinct),
                    timeout_s=caps["timeout_s"],
                    max_cell_chars=caps["max_cell_chars"],
                )
            elif action == "profile":
                result = {
                    "columns": sqlengine.profile_columns(
                        conn,
                        sqlite_dir=sqlite_dir,
                        allowed_tables=allowed,
                        blocked_columns=blocked,
                        timeout_s=caps["timeout_s"],
                    )
                }
            else:
                result = sqlengine.check_connection(
                    conn, sqlite_dir=sqlite_dir, timeout_s=min(caps["timeout_s"], 15)
                )
        except (SqlRejected, sqlengine.SqlError) as e:
            return {"ok": False, "error": str(e)}
        except Exception:
            log.exception("SQL-Lauf fehlgeschlagen")
            return _error("Interner Fehler", 500)
        return {
            "ok": True,
            **result,
        }  # Zellwerte sind schon JSON-tauglich (sqlengine._cell)

    @app.post("/v1/sql/query")
    def sql_query_route(data: SqlRequest):
        return sql_run("query", data)

    @app.post("/v1/sql/schema")
    def sql_schema_route(data: SqlRequest):
        return sql_run("schema", data)

    @app.post("/v1/sql/distinct")
    def sql_distinct_route(data: SqlRequest):
        return sql_run("distinct", data)

    @app.post("/v1/sql/profile")
    def sql_profile_route(data: SqlRequest):
        return sql_run("profile", data)

    @app.post("/v1/sql/test")
    def sql_test_route(data: SqlRequest):
        return sql_run("test", data)

    return app
