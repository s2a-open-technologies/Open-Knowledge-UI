# ruff: noqa: F811  (Fixtures aus anderen Testmodulen werden importiert und als Argument genutzt)
from app.auth import access
from app.knowledge import service as knowledge
from app.llm import connections as store
from tests.test_access import (  # noqa: F401
    add_user,
    app,
    chat,
    chat_env,
    client_for,
    conn,
)


def setup_model(conn, ids, **cfg):
    ref = store.make_base_ref(conn["id"], "m")
    store.save_model_config(ref, {"knowledge_ids": ids, **cfg})
    return ref


def names(scope):
    return set(scope.collections)


def test_model_knowledge_is_the_base_and_the_users_own_knowledge_is_added(
    app, db, conn, chat_env
):
    base = knowledge.create_knowledge("Modell")
    own = knowledge.create_knowledge("Eigenes")
    foreign = knowledge.create_knowledge("Fremd")
    alice = add_user(db, "alice", "editor")
    access.add_grant("knowledge", own["id"], "user", alice["id"], "read")
    ref = setup_model(conn, [base["id"]])  # Alice hat keine Freigabe auf "Modell"

    chat(client_for(app, alice), model=ref, knowledge_ids=[own["id"], foreign["id"]])
    assert names(chat_env[-1]) == {
        base["collection_name"],
        own["collection_name"],
    }  # "Fremd" bleibt draussen


def test_without_a_selection_only_the_model_knowledge_is_searched(
    app, db, conn, chat_env
):
    base = knowledge.create_knowledge("Modell")
    own = knowledge.create_knowledge("Eigenes")
    alice = add_user(db, "alice", "editor")
    access.add_grant("knowledge", own["id"], "user", alice["id"], "read")
    ref = setup_model(conn, [base["id"]])

    chat(client_for(app, alice), model=ref)  # None = nichts gewaehlt
    assert names(chat_env[-1]) == {base["collection_name"]}


def test_the_admin_can_make_a_model_exclusive(app, db, conn, chat_env):
    base = knowledge.create_knowledge("Modell")
    own = knowledge.create_knowledge("Eigenes")
    alice = add_user(db, "alice", "editor")
    access.add_grant("knowledge", own["id"], "user", alice["id"], "read")
    ref = setup_model(conn, [base["id"]], allow_extra_knowledge=False)

    chat(client_for(app, alice), model=ref, knowledge_ids=[own["id"]])
    assert names(chat_env[-1]) == {base["collection_name"]}


def test_selecting_the_model_knowledge_again_does_not_duplicate_it(
    app, db, conn, chat_env
):
    base = knowledge.create_knowledge("Modell")
    alice = add_user(db, "alice", "editor")
    ref = setup_model(conn, [base["id"]])
    chat(client_for(app, alice), model=ref, knowledge_ids=[base["id"]])
    assert list(chat_env[-1].collections) == [base["collection_name"]]


def test_models_without_knowledge_behave_as_before(app, db, conn, chat_env):
    own = knowledge.create_knowledge("Eigenes")
    foreign = knowledge.create_knowledge("Fremd")
    alice = add_user(db, "alice", "editor")
    access.add_grant("knowledge", own["id"], "user", alice["id"], "read")
    ref = store.make_base_ref(conn["id"], "m")
    chat(client_for(app, alice), model=ref, knowledge_ids=[own["id"], foreign["id"]])
    assert names(chat_env[-1]) == {own["collection_name"]}


def test_model_list_tells_users_the_base_knowledge_and_whether_the_choice_is_locked(
    app, db, conn, monkeypatch
):
    from app.llm import registry

    monkeypatch.setattr(registry, "fetch_remote_models", lambda c, force=False: ["m"])
    base = knowledge.create_knowledge("Verträge")
    ref = setup_model(conn, [base["id"]])
    alice = add_user(db, "alice", "user")
    m = next(
        x
        for x in client_for(app, alice).get("/api/models?visible=1").json["models"]
        if x["ref"] == ref
    )
    assert m["knowledge_locked"] is False and m["knowledge_base"] == [
        {"id": base["id"], "name": "Verträge"}
    ]
    setup_model(conn, [base["id"]], allow_extra_knowledge=False)
    m = next(
        x
        for x in client_for(app, alice).get("/api/models?visible=1").json["models"]
        if x["ref"] == ref
    )
    assert m["knowledge_locked"] is True


def test_admins_also_get_the_base_knowledge_so_the_chat_cannot_deselect_it(
    app, db, conn, monkeypatch
):
    from app.llm import registry

    monkeypatch.setattr(registry, "fetch_remote_models", lambda c, force=False: ["m"])
    base = knowledge.create_knowledge("Standard")
    ref = setup_model(conn, [base["id"]])
    m = next(
        x
        for x in client_for(app, {"id": 1}).get("/api/models?visible=1").json["models"]
        if x["ref"] == ref
    )
    assert m["knowledge_base"] == [{"id": base["id"], "name": "Standard"}]
    assert m["config"]["knowledge_ids"] == [
        base["id"]
    ]  # Admins behalten die Konfiguration


def test_the_switch_defaults_to_on_and_clones_inherit_it(db, conn):
    ref = setup_model(conn, [1], allow_extra_knowledge=False)
    cfg = next(c for c in store.list_model_configs() if c["ref"] == ref)
    assert cfg["allow_extra_knowledge"] is False
    plain = store.save_model_config(
        store.make_base_ref(conn["id"], "andere"), {"system_prompt": "x"}
    )
    assert plain["allow_extra_knowledge"] is True
