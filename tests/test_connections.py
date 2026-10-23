import pytest

from app.llm import connections as store


@pytest.fixture
def conn(db):
    return store.create_connection("LiteLLM", "http://litellm:4000/v1/", "sk-geheim")


def test_default_connection_seeded_from_legacy_settings(db):
    items = store.list_connections()
    assert [c["name"] for c in items] == ["Standard"]
    assert items[0]["base_url"] == "http://localhost:11434/v1"


def test_base_url_is_normalized_and_key_is_never_listed(conn):
    assert conn["base_url"] == "http://litellm:4000/v1"
    assert conn["has_key"] is True
    assert "api_key" not in conn
    assert all("api_key" not in c for c in store.list_connections())
    assert store.get_connection(conn["id"], with_key=True)["api_key"] == "sk-geheim"


def test_update_keeps_key_when_blank(conn):
    store.update_connection(conn["id"], name="Neu", api_key="")
    full = store.get_connection(conn["id"], with_key=True)
    assert (full["name"], full["api_key"]) == ("Neu", "sk-geheim")
    store.update_connection(conn["id"], api_key="sk-neu")
    assert store.get_connection(conn["id"], with_key=True)["api_key"] == "sk-neu"


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("base:3:org/model:tag", ("base", 3, "org/model:tag")),
        ("clone:7", ("clone", 7, None)),
    ],
)
def test_parse_ref(ref, expected):
    assert store.parse_ref(ref) == expected


@pytest.mark.parametrize("ref", ["", "base:x:m", "base:1:", "clone:abc", "foo:1"])
def test_parse_ref_rejects_garbage(ref):
    with pytest.raises(ValueError):
        store.parse_ref(ref)


def test_sanitize_params_drops_empty_and_validates():
    assert store.sanitize_params(
        {"temperature": "0.7", "top_k": "", "max_tokens": None, "thinking": "high"}
    ) == {"temperature": 0.7, "thinking": "high"}
    for bad in (
        {"temperature": 5},
        {"top_p": "x"},
        {"thinking": "irgendwie"},
        {"foo": 1},
    ):
        with pytest.raises(ValueError):
            store.sanitize_params(bad)


def test_base_model_config_is_created_on_save_and_reset_by_delete(conn):
    ref = store.make_base_ref(conn["id"], "qwen")
    assert store.get_model_config(ref) is None

    saved = store.save_model_config(
        ref,
        {
            "system_prompt": " Antworte knapp. ",
            "params": {"temperature": 0.3},
            "knowledge_ids": [2, 1, 2],
            "visible": False,
        },
    )
    assert saved["system_prompt"] == "Antworte knapp."
    assert saved["params"] == {"temperature": 0.3}
    assert saved["knowledge_ids"] == [1, 2]
    assert saved["visible"] is False and saved["is_clone"] is False

    # zweites Speichern aktualisiert dieselbe Zeile
    store.save_model_config(ref, {"system_prompt": "Anders"})
    assert len(store.list_model_configs()) == 1
    assert store.get_model_config(ref)["system_prompt"] == "Anders"

    assert store.delete_model_config(ref) is True
    assert store.get_model_config(ref) is None


def test_clone_copies_config_and_is_independent(conn):
    base = store.make_base_ref(conn["id"], "qwen")
    store.save_model_config(
        base,
        {"system_prompt": "Juristisch", "params": {"top_p": 0.5}, "knowledge_ids": [3]},
    )
    clone = store.clone_model(base, "Qwen Recht")

    assert clone["is_clone"] and clone["ref"] == f"clone:{clone['id']}"
    assert (clone["name"], clone["base_model"]) == ("Qwen Recht", "qwen")
    assert (clone["system_prompt"], clone["params"], clone["knowledge_ids"]) == (
        "Juristisch",
        {"top_p": 0.5},
        [3],
    )

    store.save_model_config(
        clone["ref"], {"system_prompt": "Nur Klon", "name": "Umbenannt"}
    )
    assert store.get_model_config(base)["system_prompt"] == "Juristisch"
    assert store.get_model_config(clone["ref"])["name"] == "Umbenannt"


def test_clone_of_clone_keeps_original_base_model(conn):
    first = store.clone_model(store.make_base_ref(conn["id"], "qwen"), "A")
    second = store.clone_model(first["ref"], "B")
    assert second["base_model"] == "qwen" and second["connection_id"] == conn["id"]


def test_delete_connection_removes_configs_and_clears_default(conn):
    ref = store.make_base_ref(conn["id"], "qwen")
    store.save_model_config(ref, {"system_prompt": "x"})
    store.clone_model(ref, "Klon")
    store.set_default_model(ref)

    assert store.delete_connection(conn["id"]) is True
    assert store.list_model_configs() == []
    assert store.get_default_model() == ""
    assert store.delete_connection(conn["id"]) is False


def test_remove_knowledge_from_models(conn):
    ref = store.make_base_ref(conn["id"], "qwen")
    store.save_model_config(ref, {"knowledge_ids": [1, 2]})
    other = store.make_base_ref(conn["id"], "llama")
    store.save_model_config(other, {"knowledge_ids": [2]})

    store.remove_knowledge_from_models(2)

    assert store.get_model_config(ref)["knowledge_ids"] == [1]
    assert store.get_model_config(other)["knowledge_ids"] is None


def test_base_model_alias_is_saved_shown_and_clearable(conn, monkeypatch):
    from app.llm import registry

    monkeypatch.setattr(
        registry, "fetch_remote_models", lambda c, force=False: ["qwen-122b"]
    )
    ref = store.make_base_ref(conn["id"], "qwen-122b")

    store.save_model_config(ref, {"name": "  Hausmodell  "})
    listed = {m["ref"]: m for m in registry.list_models()["models"]}
    assert (
        listed[ref]["name"] == "Hausmodell" and listed[ref]["model_id"] == "qwen-122b"
    )
    assert listed[ref]["customized"] is True
    assert registry.resolve_model(ref).display_name == "Hausmodell"
    assert (
        registry.resolve_model(ref).model_id == "qwen-122b"
    )  # an den Endpoint geht die Original-ID

    store.save_model_config(ref, {"name": ""})  # Alias entfernen
    assert registry.list_models()["models"][0]["name"] == "qwen-122b"


def test_clone_keeps_a_name_even_if_cleared(conn):
    clone = store.clone_model(store.make_base_ref(conn["id"], "m"), "Eigener")
    store.save_model_config(clone["ref"], {"name": " "})
    assert store.get_model_config(clone["ref"])["name"] == "Eigener"
