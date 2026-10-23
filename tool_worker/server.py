"""HTTP-Schnittstelle des Tool-Workers (siehe docs/tool-worker-api.md)."""

import hmac
import logging
from collections.abc import Callable
from pathlib import Path

from flask import Flask, jsonify, request

from tool_worker import __version__, sandbox
from tool_worker.frames import SUFFIXES

log = logging.getLogger("tool_worker")
MAX_CODE_CHARS = 20_000
MAX_FILE_CHARS = 512


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
) -> Flask:
    if role not in ("pandas", "sql"):
        raise ValueError("TOOL_WORKER_ROLE muss pandas oder sql sein")
    if not token:
        raise ValueError("TOOL_WORKER_TOKEN ist nicht gesetzt")
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
    expected = token.encode()

    def root() -> Path:
        return Path(data_dir() if callable(data_dir) else data_dir).resolve()

    @app.get("/health")
    def health():
        return jsonify(status="ok", version=__version__, role=role)

    @app.before_request
    def auth():
        path = request.path
        if path.startswith("/v1/"):
            sql_path = path.startswith("/v1/sql/")
            if (
                role == "pandas"
            ) == sql_path:  # pandas-Rolle: nur /v1/pandas; sql-Rolle: nur /v1/sql/*
                return jsonify(error="Nicht gefunden"), 404
            sent = request.headers.get("Authorization", "")
            given = sent[7:].encode() if sent.startswith("Bearer ") else b""
            if not hmac.compare_digest(given, expected):
                return jsonify(error="Nicht autorisiert"), 401

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

    @app.post("/v1/pandas")
    def pandas_run():
        data = request.get_json(silent=True)
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("file"), str)
            or not isinstance(data.get("code"), str)
        ):
            return jsonify(error="Ungültige Anfrage"), 400
        if len(data["code"]) > MAX_CODE_CHARS:
            return jsonify(error="Code zu lang"), 413
        limits = data.get("limits") if isinstance(data.get("limits"), dict) else {}

        def cap(name, default, ceiling):
            try:
                value = float(limits.get(name, default))
            except (TypeError, ValueError):
                value = default
            return min(max(value, 1), ceiling)

        target = resolve(data["file"])
        if target is None:
            return jsonify(error="Datei nicht verfügbar"), 404
        try:
            out = runner(
                str(target),
                data["code"],
                timeout_s=cap("timeout_s", 30, max_timeout_s),
                memory_mb=int(cap("memory_mb", 1024, max_memory_mb)),
                max_output_chars=int(cap("max_output_chars", 20_000, max_output_chars)),
            )
        except Exception:
            log.exception("pandas-Lauf fehlgeschlagen")
            return jsonify(error="Interner Fehler"), 500
        return jsonify(out)

    def sql_request():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("connection"), dict):
            return None
        return data

    def sql_caps(data):
        limits = data.get("limits") if isinstance(data.get("limits"), dict) else {}

        def cap(name, default, ceiling):
            try:
                value = float(limits.get(name, default))
            except (TypeError, ValueError):
                value = default
            return min(max(value, 1), ceiling)

        return {
            "row_limit": int(cap("row_limit", 50, max_rows)),
            "timeout_s": cap("timeout_s", 15, max_sql_timeout_s),
            "max_cell_chars": int(cap("max_cell_chars", 500, 5000)),
        }

    def sql_filters(data):
        tables = data.get("allowed_tables")
        blocked = (
            data.get("blocked_columns")
            if isinstance(data.get("blocked_columns"), dict)
            else {}
        )
        return (
            {str(t) for t in tables} if isinstance(tables, list) else None,
            {
                str(t): {str(c) for c in (cols or [])}
                for t, cols in blocked.items()
                if isinstance(cols, list)
            },
        )

    def sql_run(action):
        from tool_worker import sqlengine
        from tool_worker.sqlcheck import SqlRejected

        data = sql_request()
        if data is None or (action == "query" and not isinstance(data.get("sql"), str)):
            return jsonify(error="Ungültige Anfrage"), 400
        if action == "distinct" and not all(
            isinstance(data.get(k), str) and data[k] for k in ("table", "column")
        ):
            return jsonify(error="Ungültige Anfrage"), 400
        caps = sql_caps(data)
        allowed, blocked = sql_filters(data)
        conn = data["connection"]
        try:
            if action == "query":
                result = sqlengine.run_query(
                    conn,
                    data["sql"],
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
                    wanted = int(data.get("max_distinct") or 5000)
                except (TypeError, ValueError):
                    wanted = 5000
                result = sqlengine.distinct_values(
                    conn,
                    sqlite_dir=sqlite_dir,
                    table=data["table"],
                    column=data["column"],
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
            return jsonify(ok=False, error=str(e))
        except Exception:
            log.exception("SQL-Lauf fehlgeschlagen")
            return jsonify(error="Interner Fehler"), 500
        return jsonify(ok=True, **result)

    @app.post("/v1/sql/query")
    def sql_query_route():
        return sql_run("query")

    @app.post("/v1/sql/schema")
    def sql_schema_route():
        return sql_run("schema")

    @app.post("/v1/sql/distinct")
    def sql_distinct_route():
        return sql_run("distinct")

    @app.post("/v1/sql/profile")
    def sql_profile_route():
        return sql_run("profile")

    @app.post("/v1/sql/test")
    def sql_test_route():
        return sql_run("test")

    return app
