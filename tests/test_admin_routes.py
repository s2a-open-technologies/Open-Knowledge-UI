import pytest
from flask import Flask

from app.api.admin_routes import admin_bp
from app.llm import connections as store
from app.llm import registry


@pytest.fixture
def client(db):
    # Als geseedeter Admin angemeldet (id 1); es geht um die Key-Weitergabe.
    app = Flask(__name__)
    app.secret_key = "t"
    app.register_blueprint(admin_bp)
    with app.test_client() as c:
        with c.session_transaction() as s:
            s["user_id"] = 1
        yield c


@pytest.fixture
def seen_keys(monkeypatch):
    seen = []

    def fake(connection):
        seen.append((connection["base_url"], connection["api_key"]))
        return {"success": True, "message": "ok", "models": []}

    monkeypatch.setattr(registry, "test_connection", fake)
    return seen


@pytest.fixture
def conn(db):
    return store.create_connection("A", "http://good:4000/v1", "sk-geheim")


def test_stored_key_is_only_used_for_the_stored_url(client, conn, seen_keys):
    client.post(
        "/api/connections/test",
        json={"id": conn["id"], "base_url": "http://good:4000/v1/"},
    )
    client.post(
        "/api/connections/test",
        json={"id": conn["id"], "base_url": "http://evil.example/v1"},
    )
    assert seen_keys == [
        ("http://good:4000/v1", "sk-geheim"),
        ("http://evil.example/v1", ""),  # Key wird NICHT mitgeschickt
    ]


def test_explicit_key_is_always_used(client, conn, seen_keys):
    client.post(
        "/api/connections/test",
        json={"id": conn["id"], "base_url": "http://other/v1", "api_key": "sk-neu"},
    )
    assert seen_keys == [("http://other/v1", "sk-neu")]


def test_changing_url_requires_new_key(client, conn):
    res = client.patch(
        f"/api/connections/{conn['id']}", json={"base_url": "http://evil/v1"}
    )
    assert res.status_code == 400
    assert (
        store.get_connection(conn["id"], with_key=True)["base_url"]
        == "http://good:4000/v1"
    )

    ok = client.patch(
        f"/api/connections/{conn['id']}",
        json={"base_url": "http://new/v1", "api_key": "sk-2"},
    )
    assert ok.status_code == 200
    assert store.get_connection(conn["id"], with_key=True)["api_key"] == "sk-2"


def test_same_url_or_other_fields_keep_key_without_reentry(client, conn):
    res = client.patch(
        f"/api/connections/{conn['id']}",
        json={"name": "B", "base_url": "http://good:4000/v1/", "enabled": False},
    )
    assert res.status_code == 200
    full = store.get_connection(conn["id"], with_key=True)
    assert (full["name"], full["api_key"], full["enabled"]) == ("B", "sk-geheim", False)


def test_tool_worker_overview_reports_reachability(client, monkeypatch):
    from app.agent.toolworker import client as wc

    monkeypatch.setattr(
        wc.ToolWorkerClient,
        "ping",
        lambda self: {"reachable": True, "version": "0.1.0", "error": ""},
    )
    j = client.get("/api/admin/tool-worker").get_json()
    assert (
        j["reachable"] is True
        and j["url"] == "http://tool-worker:8090"
        and "token" not in str(j).lower().replace("token_configured", "")
    )


def test_tool_worker_settings_can_be_saved_and_are_validated(client, monkeypatch):
    from app.agent.toolworker import client as wc

    monkeypatch.setattr(
        wc.ToolWorkerClient,
        "ping",
        lambda self: {"reachable": False, "version": "", "error": "nicht erreichbar"},
    )
    ok = client.put("/api/admin/tool-worker", json={"timeout_s": 45})
    assert ok.status_code == 200 and ok.get_json()["timeout_s"] == 45
    bad = client.put("/api/admin/tool-worker", json={"url": "ftp://x"})
    assert bad.status_code == 400 and "http" in bad.get_json()["error"]


def test_tool_worker_overview_includes_the_sql_worker(client, monkeypatch):
    from app.agent.toolworker import client as wc

    monkeypatch.setattr(
        wc.ToolWorkerClient,
        "ping",
        lambda self: {"reachable": True, "version": "0.1.0", "error": ""},
    )
    j = client.get("/api/admin/tool-worker").get_json()
    assert (
        j["sql_url"] == "http://tool-worker-sql:8090"
        and j["sql_reachable"] is True
        and j["sql_row_limit"] == 50
    )
    ok = client.put("/api/admin/tool-worker", json={"sql_row_limit": 100})
    assert ok.status_code == 200 and ok.get_json()["sql_row_limit"] == 100
    assert (
        client.put("/api/admin/tool-worker", json={"sql_row_limit": 0}).status_code
        == 400
    )


# --- Connection-Typen: TEI und Infinity --------------------------------------------------
def test_connections_have_a_kind_and_reject_unknown_ones(client):
    ok = client.post(
        "/api/connections",
        json={"name": "TEI", "base_url": "http://tei.lan:8080", "kind": "tei"},
    )
    assert ok.status_code == 201 and ok.get_json()["kind"] == "tei"
    assert (
        client.post(
            "/api/connections",
            json={"name": "X", "base_url": "http://x.lan", "kind": "oracle"},
        ).status_code
        == 400
    )
    plain = client.post(
        "/api/connections", json={"name": "L", "base_url": "http://l.lan/v1"}
    ).get_json()
    assert plain["kind"] == "llm"  # Standard bleibt OpenAI-kompatibel
    changed = client.patch(f"/api/connections/{plain['id']}", json={"kind": "infinity"})
    assert changed.get_json()["kind"] == "infinity"
    assert [c["kind"] for c in client.get("/api/connections").get_json()][-2:] == [
        "tei",
        "infinity",
    ]


def test_connection_test_checks_the_real_apis_of_tei_and_infinity(client):
    from tests.fake_inference import TEI, Infinity, serve

    url, stop, _ = serve(TEI, mode="embedding")
    try:
        r = client.post("/api/connections/test", json={"base_url": url, "kind": "tei"})
        assert r.status_code == 200 and r.get_json()["embedding"]["dimension"] == 4
    finally:
        stop()
    url, stop, _ = serve(Infinity)
    try:
        j = client.post(
            "/api/connections/test", json={"base_url": url, "kind": "infinity"}
        ).get_json()
        assert j["success"] and j["embedding"]["ok"] and j["reranker"]["ok"]
    finally:
        stop()


def test_connection_test_uses_the_stored_key_only_for_the_stored_url(client):
    from tests.fake_inference import TEI, serve

    url, stop, _ = serve(TEI, mode="embedding", api_key="geheim")
    try:
        saved = client.post(
            "/api/connections",
            json={"name": "T", "base_url": url, "api_key": "geheim", "kind": "tei"},
        ).get_json()
        assert (
            client.post(
                "/api/connections/test",
                json={"id": saved["id"], "base_url": url, "kind": "tei"},
            ).status_code
            == 200
        )
        other = client.post(
            "/api/connections/test",
            json={
                "id": saved["id"],
                "base_url": url.replace("127.0.0.1", "localhost"),
                "kind": "tei",
            },
        )
        assert other.status_code == 502 and "API-Key" in other.get_json()["error"]
    finally:
        stop()


def test_embedding_models_of_a_connection_are_filtered_by_capability(client):
    from tests.fake_inference import Infinity, serve

    url, stop, _ = serve(Infinity)
    try:
        cid = client.post(
            "/api/connections", json={"name": "I", "base_url": url, "kind": "infinity"}
        ).get_json()["id"]
        q = f"/api/admin/embedding/remote-models?connection_id={cid}"
        assert client.get(q + "&capability=embedding").get_json() == ["emb-model"]
        assert client.get(q + "&capability=reranker").get_json() == ["rr-model"]
        assert client.get(q).get_json() == ["emb-model", "rr-model"]
    finally:
        stop()


def test_tei_and_infinity_connections_do_not_appear_as_chat_models(db, monkeypatch):
    from app.llm import connections as store
    from app.llm import registry

    monkeypatch.setattr(registry, "fetch_remote_models", lambda c, force=False: ["m1"])
    llm = store.create_connection("LLM", "http://127.0.0.1:9/v1", "")
    tei = store.create_connection("TEI", "http://127.0.0.1:9", "", kind="tei")
    models = registry.list_models()
    ids = {m["connection_id"] for m in models["models"]}
    assert llm["id"] in ids and tei["id"] not in ids
    assert models["errors"] == []  # die TEI-Connection wurde nicht einmal abgefragt
