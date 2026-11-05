"""Testhilfe: Worker im selben Prozess (TestClient) statt Container."""

from fastapi.testclient import TestClient

from app.agent.toolworker import client as worker_client
from tool_worker.server import create_app

TOKEN = "test-token-" + "x" * 20


def _json(response) -> dict:
    try:
        return response.json() or {}
    except ValueError:
        return {}


class InProcessWorker:
    root = "."


def install(monkeypatch) -> InProcessWorker:
    worker = InProcessWorker()
    http = TestClient(create_app(TOKEN, lambda: worker.root))

    def transport(method, path, payload, timeout):
        headers = {"Authorization": f"Bearer {TOKEN}"}
        r = http.request(method, path, json=payload, headers=headers)
        return r.status_code, _json(r)

    worker_client.set_override(
        worker_client.ToolWorkerClient("http://test", TOKEN, transport)
    )
    return worker


def install_sql(monkeypatch, sqlite_dir) -> None:
    http = TestClient(
        create_app(TOKEN, "/unused", role="sql", sqlite_dir=str(sqlite_dir))
    )

    def transport(method, path, payload, timeout):
        r = http.request(
            method, path, json=payload, headers={"Authorization": f"Bearer {TOKEN}"}
        )
        return r.status_code, _json(r)

    worker_client.set_override(
        worker_client.ToolWorkerClient("http://test-sql", TOKEN, transport), "sql"
    )
