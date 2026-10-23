import pytest

from app.agent.document_agent import DocumentAgent
from app.llm import connections as store
from app.llm import registry


@pytest.fixture(autouse=True)
def fresh_registry():
    registry.clear_caches()
    yield
    registry.clear_caches()


@pytest.fixture
def conn(db):
    return store.create_connection("LiteLLM", "http://litellm:4000/v1", "sk-1")


class FakeAgent:
    def __init__(self, llm, tools, system_prompt=None, middleware=None):
        self.llm, self.tools, self.system_prompt, self.middleware = (
            llm,
            tools,
            system_prompt,
            middleware,
        )


@pytest.fixture
def fake_agent(monkeypatch):
    monkeypatch.setattr(registry, "DocumentAgent", FakeAgent)
    FakeAgent.compose_prompt = DocumentAgent.compose_prompt
    monkeypatch.setattr(registry, "build_chat_llm", lambda *a, **k: ("llm", a, k))


def test_compose_prompt_layers_base_then_default_instructions_then_model_extra():
    base = DocumentAgent.SYSTEM_PROMPT
    assert DocumentAgent.compose_prompt("", behavior="") == base
    full = DocumentAgent.compose_prompt("Sei knapp.", behavior="Antworte formell.")
    assert full.startswith(base)
    assert full.index("Antworte formell.") < full.index("Sei knapp.")
    assert DocumentAgent.compose_prompt("Sei knapp.", behavior="").endswith(
        "Sei knapp."
    )


def test_the_base_prompt_holds_only_tool_and_source_rules():
    base = DocumentAgent.SYSTEM_PROMPT
    for must in (
        "document_search_tool",
        "run_pandas",
        "[STRUKTURIERTE DATEN]",
        "eckigen Klammern",
        "Keine Halluzinationen",
        "nie vollständig wieder",
    ):
        assert must in base
    # Rolle, Sprache und Stil gehoeren in die Standard-Anweisungen (UI), nicht in den festen Prompt
    for gone in ("Senior Document AI Analyst", "Immer auf Deutsch", "formell"):
        assert gone not in base


def test_default_instructions_are_stored_editable_and_reach_the_agent(db):
    from app.agent import behavior

    got = behavior.load()
    assert got["prompt"] == got["default_prompt"] == behavior.DEFAULT_BEHAVIOR
    assert "Deutsch" in got["prompt"] and "Ist X da?" in got["prompt"]
    behavior.save({"prompt": "  Duze den Nutzer.  "})
    assert behavior.current() == "Duze den Nutzer."
    assert "Duze den Nutzer." in DocumentAgent.compose_prompt("")
    behavior.save({"prompt": ""})  # bewusst leer: nur Basis und Zusatz
    assert DocumentAgent.compose_prompt("") == DocumentAgent.SYSTEM_PROMPT
    with pytest.raises(behavior.BehaviorConfigError):
        behavior.save({"prompt": "x" * (behavior.MAX_CHARS + 1)})


def test_a_changed_default_instruction_is_not_served_from_the_agent_cache(
    conn, db, monkeypatch
):
    from app.agent import behavior

    ref = store.make_base_ref(conn["id"], "qwen")
    resolved = registry.resolve_model(ref)
    first = registry.get_agent(resolved, [])
    behavior.save({"prompt": "Neuer Ton."})
    assert registry.get_agent(resolved, []) is not first


def test_resolve_base_model_without_config(conn):
    r = registry.resolve_model(store.make_base_ref(conn["id"], "qwen"))
    assert (r.model_id, r.base_url, r.api_key) == (
        "qwen",
        "http://litellm:4000/v1",
        "sk-1",
    )
    assert r.params == {} and r.system_prompt == "" and r.knowledge_ids is None


def test_resolve_clone_uses_its_config_and_base_model(conn):
    base = store.make_base_ref(conn["id"], "qwen")
    clone = store.clone_model(base, "Recht")
    store.save_model_config(
        clone["ref"],
        {"system_prompt": "Juristisch", "params": {"top_k": 10}, "knowledge_ids": [4]},
    )
    r = registry.resolve_model(clone["ref"])
    assert (r.model_id, r.display_name) == ("qwen", "Recht")
    assert (r.system_prompt, r.params, r.knowledge_ids) == (
        "Juristisch",
        {"top_k": 10},
        (4,),
    )


def test_resolve_errors(conn):
    with pytest.raises(registry.ModelUnavailableError):
        registry.resolve_model("clone:999")
    with pytest.raises(registry.ModelUnavailableError):
        registry.resolve_model("base:999:x")
    store.update_connection(conn["id"], enabled=False)
    with pytest.raises(registry.ModelUnavailableError, match="deaktiviert"):
        registry.resolve_model(store.make_base_ref(conn["id"], "qwen"))


def test_default_falls_back_to_legacy_chat_model(db):
    # Bestandsinstallation: CHAT_MODEL an der Standard-Connection
    first = store.list_connections()[0]
    ref = registry.default_model_ref()
    kind, cid, model = store.parse_ref(ref)
    assert (kind, cid) == ("base", first["id"]) and model

    explicit = store.make_base_ref(first["id"], "anderes")
    store.set_default_model(explicit)
    assert registry.default_model_ref() == explicit
    store.set_default_model("base:999:weg")  # verwaiste Referenz
    assert registry.default_model_ref() == ref


def test_every_request_gets_a_fresh_agent_and_llm(conn, monkeypatch):
    # Jede Chat-Anfrage laeuft in einer eigenen Event-Loop, die danach geschlossen wird.
    # Ein gecachtes LLM haelt einen async HTTP-Pool, dessen Keep-Alive-Verbindungen an
    # die tote Loop gebunden sind ("Event loop is closed" bei der Folge-Anfrage).
    monkeypatch.setattr(registry, "DocumentAgent", FakeAgent)
    FakeAgent.compose_prompt = DocumentAgent.compose_prompt
    monkeypatch.setattr(registry, "build_chat_llm", lambda *a, **k: object())
    resolved = registry.resolve_model(store.make_base_ref(conn["id"], "qwen"))
    a1 = registry.get_agent(resolved, ["t"])
    a2 = registry.get_agent(resolved, ["t"])
    assert a1 is not a2
    assert a1.llm is not a2.llm


def test_agent_follows_config_changes_immediately(conn, fake_agent):
    ref = store.make_base_ref(conn["id"], "qwen")
    tools = ["t"]
    store.save_model_config(ref, {"system_prompt": "Neu"})
    assert "Neu" in registry.get_agent(registry.resolve_model(ref), tools).system_prompt


def test_different_models_get_different_agents(conn, fake_agent):
    a = registry.get_agent(
        registry.resolve_model(store.make_base_ref(conn["id"], "a")), []
    )
    b = registry.get_agent(
        registry.resolve_model(store.make_base_ref(conn["id"], "b")), []
    )
    assert a is not b


def test_list_models_merges_remote_clones_and_reports_errors(conn, monkeypatch):
    base = store.make_base_ref(conn["id"], "qwen")
    store.save_model_config(base, {"visible": False, "system_prompt": "x"})
    clone = store.clone_model(base, "Qwen Recht")

    monkeypatch.setattr(
        registry, "fetch_remote_models", lambda c, force=False: ["llama", "qwen"]
    )
    result = registry.list_models()
    by_ref = {m["ref"]: m for m in result["models"]}
    assert by_ref[base]["visible"] is False and by_ref[base]["customized"] is True
    assert by_ref[store.make_base_ref(conn["id"], "llama")]["customized"] is False
    assert by_ref[clone["ref"]]["is_clone"] and by_ref[clone["ref"]]["available"]
    assert result["errors"] == []

    def boom(c, force=False):
        raise registry.RemoteModelsError("kaputt")

    monkeypatch.setattr(registry, "fetch_remote_models", boom)
    result = registry.list_models()
    refs = [m["ref"] for m in result["models"]]
    assert refs == [clone["ref"]]  # Klon bleibt sichtbar, aber nicht verfuegbar
    assert result["models"][0]["available"] is False
    assert result["errors"][0]["error"] == "kaputt"


def test_fetch_remote_models_parses_openai_format_and_maps_errors(monkeypatch):
    class Resp:
        def __init__(self, status, body):
            self.status_code, self._body = status, body

        def json(self):
            return self._body

    connection = {"id": 1, "base_url": "http://x/v1", "api_key": "k"}
    seen = {}

    def ok(url, headers, timeout, **kw):
        seen["url"], seen["auth"] = url, headers.get("Authorization")
        return Resp(200, {"data": [{"id": "b"}, {"id": "a"}, {"id": "a"}, {}]})

    monkeypatch.setattr(registry.requests, "get", ok)
    assert registry.fetch_remote_models(connection, force=True) == ["a", "b"]
    assert seen == {"url": "http://x/v1/models", "auth": "Bearer k"}

    monkeypatch.setattr(registry.requests, "get", lambda *a, **k: Resp(401, {}))
    with pytest.raises(registry.RemoteModelsError, match="API-Key"):
        registry.fetch_remote_models(connection, force=True)

    # Weiterleitungen werden nicht verfolgt (koennten auf gesperrte Adressen zeigen)
    monkeypatch.setattr(registry.requests, "get", lambda *a, **k: Resp(302, {}))
    with pytest.raises(registry.RemoteModelsError, match="Weiterleitung"):
        registry.fetch_remote_models(connection, force=True)

    monkeypatch.setattr(
        registry.requests, "get", lambda *a, **k: Resp(200, ["kein", "dict"])
    )
    with pytest.raises(registry.RemoteModelsError, match="OpenAI-Format"):
        registry.fetch_remote_models(connection, force=True)


@pytest.mark.parametrize(
    "url",
    ["http://169.254.169.254/v1", "http://[fe80::1]/v1", "http://169.254.1.1:8080/v1"],
)
def test_link_local_targets_are_rejected(url, monkeypatch):
    called = []
    monkeypatch.setattr(registry.requests, "get", lambda *a, **k: called.append(1))
    with pytest.raises(registry.RemoteModelsError, match="Link-Local"):
        registry.fetch_remote_models(
            {"id": 1, "base_url": url, "api_key": "k"}, force=True
        )
    assert called == []  # nie eine Anfrage (und damit nie ein Key) gesendet


def test_private_lan_targets_stay_allowed(monkeypatch):
    class Resp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "m"}]}

    monkeypatch.setattr(registry.requests, "get", lambda *a, **k: Resp())
    conn = {"id": 1, "base_url": "http://192.168.1.100:10002/v1", "api_key": ""}
    assert registry.fetch_remote_models(conn, force=True) == ["m"]


def test_migration_0030_seeds_the_default_instructions_and_drops_replace_prompt(
    tmp_path,
):
    import importlib.util
    import os
    import subprocess
    import sys
    from pathlib import Path

    import sqlalchemy as sa

    from app.agent import behavior

    spec = importlib.util.spec_from_file_location(
        "m0030", Path("alembic/versions/0030_agent_behavior.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.DEFAULT_BEHAVIOR == behavior.DEFAULT_BEHAVIOR and mod.KEY == behavior.KEY

    url = f"sqlite:///{tmp_path / 'm.db'}"
    env = {**os.environ, "DATABASE_URL": url}
    run = lambda *a: subprocess.run(
        [sys.executable, "-m", "alembic", *a], env=env, check=True, capture_output=True
    )
    run("upgrade", "head")
    eng = sa.create_engine(url)
    assert "replace_prompt" not in {
        c["name"] for c in sa.inspect(eng).get_columns("model_configs")
    }
    with eng.connect() as c:
        assert (
            c.execute(
                sa.text("select value from settings where key='AGENT_BEHAVIOR_PROMPT'")
            ).scalar()
            == behavior.DEFAULT_BEHAVIOR
        )
    run("downgrade", "0029")
    assert "replace_prompt" in {
        c["name"]
        for c in sa.inspect(sa.create_engine(url)).get_columns("model_configs")
    }
    run("upgrade", "head")


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_chat_llm_passes_the_servers_thinking_through(field):
    # vLLM liefert das Denken in delta.reasoning, nicht im Inhalt; ChatOpenAI wuerde es verwerfen.
    from langchain_core.messages import AIMessageChunk

    from app.llm.factory import build_chat_llm

    llm = build_chat_llm("http://127.0.0.1:9/v1", "k", "m", {"thinking": "medium"})
    raw = {
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": None, field: "Hmm, 17*23"},
            }
        ]
    }
    out = llm._convert_chunk_to_generation_chunk(raw, AIMessageChunk, None)
    assert out.message.additional_kwargs["reasoning_content"] == "Hmm, 17*23"
    plain = {"choices": [{"index": 0, "delta": {"content": "391"}}]}
    out = llm._convert_chunk_to_generation_chunk(plain, AIMessageChunk, None)
    assert (
        out.message.content == "391"
        and "reasoning_content" not in out.message.additional_kwargs
    )
