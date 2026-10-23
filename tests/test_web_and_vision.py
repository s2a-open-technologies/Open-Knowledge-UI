import pytest
from flask import Flask

from app.agent.tools import web_access
from app.api.admin_routes import admin_bp
from app.api.api_routes import api_bp
from app.api.routes import bp as main_bp
from app.core.settings_service import SettingsService
from app.llm import connections as store
from app.llm import registry
from app.llm.local_llm import VisionLLM
from tests.test_access import client_for

ADMIN = {"id": 1, "role": "admin", "username": "admin"}


@pytest.fixture
def app(db, monkeypatch):
    monkeypatch.setattr(
        registry, "fetch_remote_models", lambda c, force=False: ["m1", "m2"]
    )
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    for bp in (main_bp, api_bp, admin_bp):
        app.register_blueprint(bp)
    return app


def set_settings(db, **values):
    s = db.get_session()
    SettingsService(s).set_batch(values)
    s.close()


def test_web_search_needs_searxng_and_an_active_pii_filter(db):
    assert web_access.web_search_available() is False
    set_settings(db, SEARCH_SEARXNG_URL="http://sx:8080")
    assert web_access.web_search_available() is False  # PII fehlt
    set_settings(
        db, PII_FILTER_URL="http://pii:9500/api/v1/sanitize", PII_FILTER_ENABLED="false"
    )
    assert web_access.web_search_available() is False  # PII abgeschaltet
    set_settings(db, PII_FILTER_ENABLED="true")
    assert web_access.web_search_available() is True
    assert web_access.build_web_search_tool().name == "web_search_tool"
    set_settings(db, SEARCH_SEARXNG_URL="")
    assert web_access.build_web_search_tool() is None


def test_web_search_per_model_is_off_by_default_and_blocked_without_pii(app, db):
    admin = client_for(app, ADMIN)
    conn = store.create_connection("C", "http://127.0.0.1:9/v1", "k")
    ref = store.make_base_ref(conn["id"], "m1")
    assert registry.resolve_model(ref).web_search is False

    blocked = admin.put("/api/models/config", json={"ref": ref, "web_search": True})
    assert blocked.status_code == 400 and "PII-Filter" in blocked.json["error"]
    assert registry.resolve_model(ref).web_search is False
    assert admin.get("/api/models").json["web_search_available"] is False

    set_settings(
        db, SEARCH_SEARXNG_URL="http://sx:8080", PII_FILTER_URL="http://pii:9500/x"
    )
    assert admin.get("/api/models").json["web_search_available"] is True
    ok = admin.put("/api/models/config", json={"ref": ref, "web_search": True})
    assert ok.status_code == 200 and ok.json["web_search"] is True
    resolved = registry.resolve_model(ref)
    assert resolved.web_search is True

    # Der Agent-Cache unterscheidet Modelle mit und ohne Websuche-Werkzeug
    a = resolved.signature()
    admin.put("/api/models/config", json={"ref": ref, "web_search": False})
    assert registry.resolve_model(ref).signature() != a


def test_clone_inherits_web_search(app, db):
    set_settings(
        db, SEARCH_SEARXNG_URL="http://sx:8080", PII_FILTER_URL="http://pii:9500/x"
    )
    conn = store.create_connection("C", "http://127.0.0.1:9/v1", "k")
    ref = store.make_base_ref(conn["id"], "m1")
    store.save_model_config(ref, {"web_search": True})
    clone = store.clone_model(ref, "Kopie")
    assert clone["web_search"] is True


def test_infrastructure_settings_come_from_the_environment_not_the_database(
    db, monkeypatch
):
    s = db.get_session()
    svc = SettingsService(s)
    svc.set("QDRANT_HOST", "alt-aus-der-db")
    monkeypatch.setenv("QDRANT_HOST", "qdrant")
    assert svc.get("QDRANT_HOST", "localhost") == "qdrant"
    monkeypatch.delenv("QDRANT_HOST")
    assert svc.get("QDRANT_HOST", "localhost") == "localhost"
    s.close()


def test_tools_api_is_admin_only_and_validates(app, db):
    from tests.test_access import add_user

    ed = add_user(db, "ed", "editor")
    assert client_for(app, ed).get("/api/admin/tools").status_code == 403
    assert client_for(app, ed).put("/api/admin/tools", json={}).status_code == 403
    admin = client_for(app, ADMIN)
    assert (
        admin.put("/api/admin/tools", json={"search": {"url": "ftp://x"}}).status_code
        == 400
    )
    assert (
        admin.put(
            "/api/admin/tools", json={"pii": {"url": "http://169.254.169.254/x"}}
        ).status_code
        == 400
    )
    assert (
        admin.put(
            "/api/admin/tools", json={"stt": {"base_url": "http://169.254.169.254/v1"}}
        ).status_code
        == 400
    )
    assert admin.post("/api/admin/tools/test", json={"kind": "nope"}).status_code == 400


def test_tools_test_uses_form_values_and_stored_key_only_for_the_stored_url(
    app, db, monkeypatch
):
    import app.api.user_routes as ur

    seen = []
    monkeypatch.setattr(
        ur,
        "_test_stt",
        lambda s: (
            seen.append((s.get("STT_BASEURL"), s.get("STT_API_KEY")))
            or {"success": True}
        ),
    )
    admin = client_for(app, ADMIN)
    admin.put(
        "/api/admin/tools",
        json={"stt": {"base_url": "http://stt:5001/v1", "api_key": "geheim"}},
    )
    assert (
        admin.post(
            "/api/admin/tools/test",
            json={
                "kind": "stt",
                "values": {"base_url": "http://stt:5001/v1", "api_key": ""},
            },
        ).status_code
        == 200
    )
    assert (
        admin.post(
            "/api/admin/tools/test",
            json={
                "kind": "stt",
                "values": {"base_url": "http://anderer:5001/v1", "api_key": ""},
            },
        ).status_code
        == 200
    )
    assert seen[0] == ("http://stt:5001/v1", "geheim")
    assert seen[1] == (
        "http://anderer:5001/v1",
        "",
    )  # gespeicherter Key geht nie an eine fremde Adresse


def put_tasks(admin, **tasks):
    return admin.put("/api/admin/embedding", json={"tasks": tasks})


def test_vision_model_comes_from_connections(app, db, monkeypatch):
    admin = client_for(app, ADMIN)
    conn = store.create_connection("C", "http://127.0.0.1:9/v1", "k")
    default_ref = store.make_base_ref(conn["id"], "m1")
    vision_ref = store.make_base_ref(conn["id"], "m2")
    store.set_default_model(default_ref)

    llm = VisionLLM()
    assert llm.vision_model == "m1"  # ohne Auswahl: Standardmodell

    assert (
        put_tasks(admin, vision={"model_ref": vision_ref}).json["tasks"]["vision"][
            "model_ref"
        ]
        == vision_ref
    )

    assert llm.vision_model == "m2"

    assert put_tasks(admin, vision={"model_ref": "base:99:x"}).status_code == 400
    assert (
        put_tasks(admin, vision={"model_ref": ""}).json["tasks"]["vision"]["model_ref"]
        == ""
    )


def test_vision_and_context_use_separate_models_and_parameters(app, db, monkeypatch):
    from app.llm import local_llm

    admin = client_for(app, ADMIN)
    conn = store.create_connection("C", "http://127.0.0.1:9/v1", "k")
    store.set_default_model(store.make_base_ref(conn["id"], "m1"))
    r = put_tasks(
        admin,
        vision={
            "model_ref": store.make_base_ref(conn["id"], "m2"),
            "temperature": 0.7,
            "max_tokens": 900,
        },
        context={
            "model_ref": store.make_base_ref(conn["id"], "m1"),
            "temperature": "",
            "max_tokens": "",
        },
    )
    assert r.status_code == 200
    built = []
    real = local_llm.build_chat_llm
    monkeypatch.setattr(
        local_llm,
        "build_chat_llm",
        lambda *a, **k: built.append((a[2], a[3])) or real(*a, **k),
    )
    vision, context = local_llm.TaskLLM("vision"), local_llm.TaskLLM("context")

    assert (
        built[0][0] == "m2"
        and built[0][1]["temperature"] == 0.7
        and built[0][1]["max_tokens"] == 900
    )
    assert built[1][0] == "m1" and "max_tokens" not in built[1][1]
    assert vision.vision_model == "m2" and context.vision_model == "m1"


def test_prompts_live_in_the_database_with_seed_fallback_and_reset(app, db):
    from app.ingest import prompts, tasks
    from app.llm.local_llm import TaskLLM

    admin = client_for(app, ADMIN)
    overview = admin.get("/api/admin/embedding").json["tasks"]
    assert (
        overview["vision"]["prompt"]
        == prompts.VISION_PROMPT
        == overview["vision"]["default_prompt"]
    )
    assert overview["context"]["excerpt_chars"] == 2000

    r = put_tasks(
        admin,
        vision={"prompt": "Beschreibe knapp."},
        context={"prompt": "Ort: {abschnitt}", "excerpt_chars": 500},
    )
    assert r.status_code == 200
    assert (
        "Beschreibe knapp."
        in TaskLLM("vision").generate_image_message("AAAA").content[0]["text"]
    )
    assert tasks.load_task("context")["excerpt_chars"] == 500
    assert (
        tasks.render_context_prompt("Ort: {abschnitt} {x}", "D", "Teil {y}")
        == "Ort: Teil {y} {x}"
    )
    # Zuruecksetzen: der Standard steht in der Datenbank (default_value) und kommt mit der Antwort
    back = put_tasks(
        admin, vision={"prompt": r.json["tasks"]["vision"]["default_prompt"]}
    ).json["tasks"]["vision"]
    assert back["prompt"] == prompts.VISION_PROMPT


def test_prompt_and_parameter_validation(app, db):
    admin = client_for(app, ADMIN)
    assert put_tasks(admin, context={"prompt": "ohne Platzhalter"}).status_code == 400
    assert put_tasks(admin, vision={"prompt": "  "}).status_code == 400
    assert put_tasks(admin, vision={"prompt": "x" * 9000}).status_code == 400
    assert put_tasks(admin, vision={"temperature": 3}).status_code == 400
    assert put_tasks(admin, vision={"max_tokens": 0}).status_code == 400
    assert put_tasks(admin, context={"excerpt_chars": 10}).status_code == 400
    assert put_tasks(admin, unbekannt={}).status_code == 400
    assert (
        client_for(app, add_user_editor(db))
        .put("/api/admin/embedding", json={"tasks": {}})
        .status_code
        == 403
    )


def add_user_editor(db):
    from tests.test_access import add_user

    return add_user(db, "ed", "editor")


def test_empty_prompt_in_the_database_falls_back_to_the_seed(app, db):
    from app.ingest import prompts, tasks

    s = db.get_session()
    SettingsService(s).set("VISION_PROMPT", "")
    s.close()
    assert tasks.load_task("vision")["prompt"] == prompts.VISION_PROMPT


def test_vision_can_be_switched_off(app, db):
    from app.ingest.loaders.preprocess_base import _vision_enabled

    admin = client_for(app, ADMIN)
    assert _vision_enabled() is True
    r = put_tasks(admin, vision={"enabled": False})
    assert r.status_code == 200 and r.json["tasks"]["vision"]["enabled"] is False
    assert _vision_enabled() is False


def test_hidden_models_stay_selectable_for_vision(app, db):
    admin = client_for(app, ADMIN)
    conn = store.create_connection("C", "http://127.0.0.1:9/v1", "k")
    ref = store.make_base_ref(conn["id"], "m2")
    admin.put("/api/models/config", json={"ref": ref, "visible": False})
    assert ref not in [
        m["ref"] for m in admin.get("/api/models?visible=1").json["models"]
    ]
    assert ref in [m["ref"] for m in admin.get("/api/models").json["models"]]
    assert put_tasks(admin, vision={"model_ref": ref}).status_code == 200


def test_models_response_names_the_default_parameters_for_the_config_dialog(app, db):
    r = client_for(app, ADMIN).get("/api/models")
    defaults = r.json["param_defaults"]
    assert (
        defaults["temperature"] == 0.1
        and defaults["top_k"] == 40
        and defaults["thinking"] == "none"
    )
    set_settings(db, CHAT_TEMPERATURE="0.5")
    assert (
        client_for(app, ADMIN).get("/api/models").json["param_defaults"]["temperature"]
        == 0.5
    )
