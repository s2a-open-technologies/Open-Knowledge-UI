"""Testhilfe: Worker im selben Prozess (Flask-Testclient) statt Container."""

from app.toolworker import client as worker_client
from tool_worker.server import create_app

TOKEN = "test-token-" + "x" * 20


class InProcessWorker:
    root = "."


def install(monkeypatch) -> InProcessWorker:
    worker = InProcessWorker()
    flask_client = create_app(TOKEN, lambda: worker.root).test_client()

    def transport(method, path, payload, timeout):
        headers = {"Authorization": f"Bearer {TOKEN}"}
        r = flask_client.open(path, method=method, json=payload, headers=headers)
        return r.status_code, r.get_json(silent=True) or {}

    worker_client.set_override(
        worker_client.ToolWorkerClient("http://test", TOKEN, transport)
    )
    return worker


def install_sql(monkeypatch, sqlite_dir) -> None:
    flask_client = create_app(
        TOKEN, "/unused", role="sql", sqlite_dir=str(sqlite_dir)
    ).test_client()

    def transport(method, path, payload, timeout):
        r = flask_client.open(
            path,
            method=method,
            json=payload,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        return r.status_code, r.get_json(silent=True) or {}

    worker_client.set_override(
        worker_client.ToolWorkerClient("http://test-sql", TOKEN, transport), "sql"
    )
