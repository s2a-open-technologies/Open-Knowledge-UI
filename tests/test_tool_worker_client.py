import pytest

from app.agent.toolworker.client import ToolWorkerClient, ToolWorkerUnavailable

LIMITS = {"timeout_s": 5, "memory_mb": 256, "max_output_chars": 1000}


def client_with(status, payload):
    calls = []

    def transport(method, path, body, timeout):
        calls.append((method, path, body, timeout))
        if isinstance(status, Exception):
            raise status
        return status, payload

    return ToolWorkerClient("http://w:8090", "tok", transport), calls


def test_returns_output_and_sends_limits():
    c, calls = client_with(200, {"ok": True, "output": "Ergebnis:\n350"})
    assert c.run_pandas("a.csv", "result=1", **LIMITS) == "Ergebnis:\n350"
    method, path, body, timeout = calls[0]
    assert (method, path) == ("POST", "/v1/pandas")
    assert body == {
        "file": "a.csv",
        "code": "result=1",
        "limits": {"timeout_s": 5, "memory_mb": 256, "max_output_chars": 1000},
    }
    assert timeout > 5  # Netzwerk-Puffer ueber dem Rechenlimit


def test_user_code_error_is_text_not_unavailable():
    c, _ = client_with(
        200,
        {
            "ok": False,
            "error": "FEHLER: 'import' ist aus Sicherheitsgründen blockiert.",
        },
    )
    assert c.run_pandas("a.csv", "import os", **LIMITS).startswith("FEHLER: 'import'")


@pytest.mark.parametrize(
    "status,payload,needle",
    [
        (401, {}, "Zugangsdaten"),
        (500, {}, "Fehler"),
        (503, {}, "Fehler"),
        (ConnectionError("http://w:8090 refused"), None, "nicht erreichbar"),
        (TimeoutError("x"), None, "nicht erreichbar"),
    ],
)
def test_unavailable_messages_do_not_leak_the_address(status, payload, needle):
    c, _ = client_with(status, payload)
    with pytest.raises(ToolWorkerUnavailable) as e:
        c.run_pandas("a.csv", "x", **LIMITS)
    assert needle in str(e.value) and "w:8090" not in str(e.value)


def test_404_maps_to_file_error_text():
    c, _ = client_with(404, {"error": "Datei nicht verfügbar"})
    assert "nicht verfügbar" in c.run_pandas("a.csv", "x", **LIMITS)


def test_ping_reports_reachable_and_not():
    ok, _ = client_with(200, {"status": "ok", "version": "0.1.0"})
    assert ok.ping() == {"reachable": True, "version": "0.1.0", "error": ""}
    down, _ = client_with(ConnectionError("x"), None)
    assert down.ping()["reachable"] is False


def test_sql_calls_post_to_the_sql_endpoints_and_return_the_body():
    c, calls = client_with(
        200, {"ok": True, "columns": ["a"], "rows": [[1]], "truncated": False}
    )
    assert c.sql("query", {"sql": "select 1"}, 20)["rows"] == [[1]]
    assert calls[0][:2] == ("POST", "/v1/sql/query") and calls[0][3] == 20


@pytest.mark.parametrize("status", [401, 404, 500, 503])
def test_sql_unavailable_variants_do_not_leak_the_address(status):
    c, _ = client_with(status, {})
    with pytest.raises(ToolWorkerUnavailable) as e:
        c.sql("query", {}, 5)
    assert "w:8090" not in str(e.value)


def test_overrides_are_kept_per_role():
    from app.agent.toolworker import client as wc

    a, _ = client_with(200, {})
    b, _ = client_with(200, {})
    wc.set_override(a, "pandas")
    wc.set_override(b, "sql")
    assert wc.get_client("pandas") is a and wc.get_client("sql") is b
    wc.set_override(None, "pandas")
    wc.set_override(None, "sql")
