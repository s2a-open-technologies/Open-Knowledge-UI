"""Produktiver Einstieg: ``gunicorn tool_worker.wsgi:app``. Ohne Token startet der Worker nicht."""

import os


def _secret(name: str) -> str:
    path = os.getenv(f"{name}_FILE")
    if path:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    return os.getenv(name, "").strip()


from tool_worker.server import create_app

app = create_app(
    _secret("TOOL_WORKER_TOKEN"),
    os.getenv("TOOL_WORKER_DATA_DIR", "/data"),
    max_timeout_s=float(os.getenv("TOOL_WORKER_MAX_TIMEOUT_S", "120")),
    max_memory_mb=int(os.getenv("TOOL_WORKER_MAX_MEMORY_MB", "4096")),
    role=os.getenv("TOOL_WORKER_ROLE", "pandas"),
    sqlite_dir=os.getenv("TOOL_WORKER_SQLITE_DIR", "/sqlite"),
    max_rows=int(os.getenv("TOOL_WORKER_MAX_ROWS", "1000")),
    max_sql_timeout_s=float(os.getenv("TOOL_WORKER_MAX_SQL_TIMEOUT_S", "60")),
    max_distinct=int(os.getenv("TOOL_WORKER_MAX_DISTINCT", "100000")),
)
