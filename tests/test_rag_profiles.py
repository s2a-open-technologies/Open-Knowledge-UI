import pytest
from flask import Flask
from qdrant_client import QdrantClient
from werkzeug.security import generate_password_hash

from app.auth import access
from app.auth.user_admin_routes import users_bp
from app.core.models import User
from app.rag import knowledge, model_settings
from app.rag import profiles as prof
from app.rag.profiles import ProfileError
from app.rag.routes import rag_bp
from app.vector.collections import ensure_collection
from app.web.admin_routes import admin_bp
from app.web.api_routes import api_bp
from app.web.routes import bp as main_bp
from tests.test_rag import FakeDense, FakeSparse, add_point

# --- Konfiguration ---------------------------------------------------------


def test_defaults_are_filled_in():
    c = prof.validate_config({})
    assert c["search"]["mode"] == "hybrid" and c["rerank"]["enabled"] is False
    assert (
        c["query_enhancement"]["mode"] == "off"
        and c["final_k"] == 5
        and c["neighbors"] == 0
    )
    assert c["query_enhancement"]["prompt"] == prof.DEFAULT_QUERY_PROMPT


@pytest.mark.parametrize(
    "bad,fragment",
    [
        ({"final_k": 0}, "final_k"),
        ({"final_k": 99}, "final_k"),
        ({"neighbors": 9}, "neighbors"),
        ({"search": {"mode": "magic"}}, "search.mode"),
        ({"search": {"candidates": 0}}, "search.candidates"),
        ({"search": {"dense_weight": -1}}, "dense_weight"),
        ({"rerank": {"enable": True}}, "rerank.enable"),
        ({"mmr": {"lambda_mult": 2}}, "lambda_mult"),
        (
            {
                "query_enhancement": {
                    "mode": "multi_query",
                    "prompt": "ohne Platzhalter",
                }
            },
            "{frage}",
        ),
        ({"query_enhancement": {"n_queries": 50}}, "n_queries"),
        ({"unbekannt": 1}, "unbekannt"),
    ],
)
def test_invalid_configs_are_rejected_with_a_readable_message(bad, fragment):
    with pytest.raises(ProfileError) as e:
        prof.validate_config(bad)
    assert fragment in e.value.message


def test_prompt_placeholder_is_only_required_when_enhancement_is_active():
    assert prof.validate_config(
        {"query_enhancement": {"mode": "off", "prompt": "egal"}}
    )


# --- Seed und CRUD ----------------------------------------------------------


def test_seeds_three_profiles_with_the_recommended_one_as_default(db):
    items = prof.list_profiles()
    assert [p["name"] for p in items] == ["Standard", "Gründlich", "Schnell"]
    standard, thorough, fast = items
    assert (
        standard["is_default"] and not thorough["is_default"] and not fast["is_default"]
    )
    assert all(p["description"] for p in items)
    db.init_default_data()
    assert len(prof.list_profiles()) == 3  # idempotent


def test_standard_profile_is_the_recommended_strategy(db):
    cfg = prof.default_config()
    assert (
        cfg["search"]["mode"] == "hybrid" and cfg["search"]["dense_min_score"] is None
    )
    assert cfg["rerank"]["enabled"] is True and cfg["rerank"]["model"] == "default"
    assert (
        cfg["neighbors"] == 1 and cfg["final_k"] == 5 and cfg["mmr"]["enabled"] is False
    )
    # Kein Zusatz-LLM-Aufruf pro Frage im Standard (Messung: kein Gewinn, aber Latenz).
    assert cfg["query_enhancement"]["mode"] == "off"


def test_thorough_profile_adds_llm_questions_and_fast_matches_legacy_behavior(db):
    by_name = {p["name"]: p["config"] for p in prof.list_profiles()}
    thorough, fast = by_name["Gründlich"], by_name["Schnell"]
    assert (
        thorough["query_enhancement"]["mode"] == "multi_query"
        and thorough["rerank"]["enabled"]
    )
    assert fast["rerank"]["enabled"] is False and fast["neighbors"] == 0
    assert (fast["final_k"], fast["search"]["candidates"]) == (
        5,
        30,
    )  # RETRIEVER_*-Defaults
    assert fast["mmr"] == {"enabled": True, "lambda_mult": 0.5}
    assert fast["search"]["dense_min_score"] == 0.2


def test_new_profiles_start_from_the_standard_profile(db):
    from app.rag import routes

    assert routes.profiles.default_config()["rerank"]["enabled"] is True


def test_crud_roundtrip_and_unique_names(db):
    p = prof.create_profile("Verträge", "präzise", {"final_k": 8, "neighbors": 2})
    assert p["config"]["final_k"] == 8 and not p["is_default"]
    with pytest.raises(ProfileError) as e:
        prof.create_profile("Verträge", "", {})
    assert e.value.status == 409
    with pytest.raises(ProfileError):
        prof.create_profile("  ", "", {})

    updated = prof.update_profile(
        p["id"], {"name": "Verträge 2", "config": {"final_k": 3}}
    )
    assert (
        updated["name"],
        updated["config"]["final_k"],
        updated["config"]["neighbors"],
    ) == ("Verträge 2", 3, 0)
    with pytest.raises(ProfileError) as e:
        prof.update_profile(999, {"name": "x"})
    assert e.value.status == 404


def test_default_profile_cannot_be_deleted(db):
    with pytest.raises(ProfileError) as e:
        prof.delete_profile(prof.get_profile(None)["id"])
    assert e.value.status == 409


def test_profile_assignment_is_switchable_and_falls_back_to_default(db):
    kb = knowledge.create_knowledge("FAQ")
    assert (
        prof.config_for_knowledge(knowledge.get_knowledge(kb["id"]))
        == prof.default_config()
    )

    quality = next(p for p in prof.list_profiles() if p["name"] == "Gründlich")
    assert prof.set_knowledge_profile(kb["id"], quality["id"])
    k = knowledge.get_knowledge(kb["id"])
    assert k["retrieval_profile_id"] == quality["id"]
    assert prof.config_for_knowledge(k)["query_enhancement"]["mode"] == "multi_query"

    assert prof.set_knowledge_profile(kb["id"], None)  # zurueck zum Standard
    assert (
        prof.config_for_knowledge(knowledge.get_knowledge(kb["id"]))
        == prof.default_config()
    )

    with pytest.raises(ProfileError):
        prof.set_knowledge_profile(kb["id"], 999)
    assert prof.set_knowledge_profile(999, None) is False


def test_deleting_a_profile_resets_assigned_knowledge(db):
    kb = knowledge.create_knowledge("X")
    custom = prof.create_profile("Eigenes", "", {"final_k": 9})
    prof.set_knowledge_profile(kb["id"], custom["id"])
    prof.delete_profile(custom["id"])
    assert knowledge.get_knowledge(kb["id"])["retrieval_profile_id"] is None


# --- API -------------------------------------------------------------------


def add_user(db, username, role):
    s = db.get_session()
    u = User(
        username=username,
        display_name=username,
        password_hash=generate_password_hash("pw"),
        role=role,
        is_active=True,
    )
    s.add(u)
    s.commit()
    uid = u.id
    s.close()
    return {"id": uid, "role": role}


@pytest.fixture
def app(db, monkeypatch):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    for bp in (main_bp, api_bp, admin_bp, users_bp, rag_bp):
        app.register_blueprint(bp)
    # Standardprofil enthaelt Reranking: Tests duerfen kein Modell laden.
    monkeypatch.setattr(
        model_settings,
        "rerank_score",
        lambda model, q, texts: [-float(i) for i in range(len(texts))],
    )
    qd = QdrantClient(":memory:")
    ensure_collection(qd, "open_knowledge_ui", 7)
    for i, t in enumerate(
        ["Dach Sanierung", "Dach Reparatur", "Dach Garantie", "Dach Daemmung"]
    ):
        add_point(qd, "open_knowledge_ui", i + 1, t, filename=f"d{i}.md")
    app.extensions.update(client=qd, sparse_embeddings=FakeSparse())
    # Retrieval-Test und Suche nutzen den Embedder des Knowledge: hier der Fake
    monkeypatch.setattr(model_settings, "embedder_for", lambda k: FakeDense())
    return app


def client_for(app, user):
    c = app.test_client()
    with c.session_transaction() as s:
        s["user_id"] = user["id"]
    return c


ADMIN = {"id": 1}


def test_profile_api_permissions(app, db):
    ed = client_for(app, add_user(db, "ed", "editor"))
    us = client_for(app, add_user(db, "us", "user"))
    admin = client_for(app, ADMIN)

    assert admin.get("/api/retrieval/profiles").status_code == 200
    assert (
        ed.get("/api/retrieval/profiles").status_code == 200
    )  # zum Auswaehlen am Knowledge
    assert (
        us.get("/api/retrieval/profiles").status_code == 200
    )  # auch Benutzer waehlen ein Profil fuer eigenes Knowledge
    for c in (ed, us):
        assert c.post("/api/retrieval/profiles", json={"name": "x"}).status_code in (
            403,
        )
        assert (
            c.patch("/api/retrieval/profiles/1", json={"name": "y"}).status_code == 403
        )
        assert c.delete("/api/retrieval/profiles/2").status_code == 403

    created = admin.post(
        "/api/retrieval/profiles",
        json={"name": "Neu", "description": "d", "config": {"final_k": 7}},
    )
    assert created.status_code == 201 and created.json["config"]["final_k"] == 7


def test_profile_api_validates(app):
    admin = client_for(app, ADMIN)
    bad = admin.post(
        "/api/retrieval/profiles", json={"name": "Bad", "config": {"final_k": 0}}
    )
    assert bad.status_code == 400 and "final_k" in bad.json["error"]
    assert (
        admin.post("/api/retrieval/profiles", json={"name": "Standard"}).status_code
        == 409
    )
    assert admin.delete("/api/retrieval/profiles/1").status_code == 409  # Standard
    # Das Reranking-Modell ist global: ein Modellname im Profil wird ignoriert
    ok = admin.post(
        "/api/retrieval/profiles",
        json={"name": "Alt", "config": {"rerank": {"model": "irgendwas"}}},
    )
    assert ok.status_code == 201 and ok.json["config"]["rerank"]["model"] == "default"


def test_assigning_a_profile_needs_write_access_on_the_knowledge(app, db):
    owner, other = add_user(db, "owner", "editor"), add_user(db, "other", "editor")
    kb = knowledge.create_knowledge("Meins")
    access.add_grant("knowledge", kb["id"], "user", owner["id"], "write")
    access.add_grant("knowledge", kb["id"], "user", other["id"], "read")
    url = f"/api/knowledge/{kb['id']}/retrieval-profile"

    assert (
        client_for(app, other).put(url, json={"profile_id": 2}).status_code == 404
    )  # nur lesen
    ok = client_for(app, owner).put(url, json={"profile_id": 2})
    assert (
        ok.status_code == 200
        and knowledge.get_knowledge(kb["id"])["retrieval_profile_id"] == 2
    )
    assert client_for(app, owner).put(url, json={"profile_id": 999}).status_code == 404
    assert client_for(app, owner).put(url, json={"profile_id": "x"}).status_code == 400
    assert client_for(app, owner).put(url, json={"profile_id": None}).status_code == 200


def test_pipeline_test_endpoint_returns_documents_and_trace(app):
    admin = client_for(app, ADMIN)
    res = admin.post("/api/retrieval/test", json={"question": "Dach"})
    assert res.status_code == 200
    result = res.json["results"][0]
    assert result["knowledge"] == "Standard" and result["documents"]
    stages = [s["stage"] for s in result["trace"]["stages"]]
    assert stages[0] == "search" and stages[-1] == "final"
    assert "ms" in result["trace"]["stages"][0] and result["trace"]["total_ms"] >= 0


def test_switching_the_knowledge_profile_changes_the_result_immediately(app):
    admin = client_for(app, ADMIN)
    few = admin.post(
        "/api/retrieval/profiles",
        json={"name": "Knapp", "config": {"final_k": 1, "mmr": {"enabled": False}}},
    ).json
    many = admin.post(
        "/api/retrieval/profiles",
        json={"name": "Breit", "config": {"final_k": 4, "mmr": {"enabled": False}}},
    ).json
    kid = knowledge.get_knowledge(None)["id"]
    access.add_grant(
        "knowledge", kid, "user", 1, "write"
    )  # Admins haben keinen Sonderzugriff

    def count():
        return len(
            admin.post("/api/retrieval/test", json={"question": "Dach"}).json[
                "results"
            ][0]["documents"]
        )

    admin.put(f"/api/knowledge/{kid}/retrieval-profile", json={"profile_id": few["id"]})
    assert count() == 1
    admin.put(
        f"/api/knowledge/{kid}/retrieval-profile", json={"profile_id": many["id"]}
    )
    assert count() == 4


def test_test_endpoint_can_try_other_profiles_without_assigning_them(app):
    admin = client_for(app, ADMIN)
    few = admin.post(
        "/api/retrieval/profiles",
        json={"name": "Knapp", "config": {"final_k": 1, "mmr": {"enabled": False}}},
    ).json
    res = admin.post(
        "/api/retrieval/test", json={"question": "Dach", "profile_id": few["id"]}
    )
    assert len(res.json["results"][0]["documents"]) == 1
    assert (
        knowledge.get_knowledge(None)["retrieval_profile_id"] is None
    )  # nichts zugewiesen


def test_drafts_are_admin_only_and_validated(app, db):
    draft = {"final_k": 2, "mmr": {"enabled": False}}
    ed = client_for(app, add_user(db, "ed", "editor"))
    assert (
        ed.post(
            "/api/retrieval/test", json={"question": "Dach", "config": draft}
        ).status_code
        == 403
    )
    admin = client_for(app, ADMIN)
    ok = admin.post("/api/retrieval/test", json={"question": "Dach", "config": draft})
    assert ok.status_code == 200 and len(ok.json["results"][0]["documents"]) == 2
    assert (
        admin.post(
            "/api/retrieval/test", json={"question": "Dach", "config": {"final_k": 0}}
        ).status_code
        == 400
    )


def test_test_endpoint_respects_knowledge_access(app, db):
    ed = client_for(app, add_user(db, "ed", "editor"))
    private = knowledge.create_knowledge("Privat")
    res = ed.post(
        "/api/retrieval/test",
        json={"question": "Dach", "knowledge_ids": [private["id"]]},
    )
    assert res.status_code == 400  # nicht lesbar -> nichts zu durchsuchen
    ok = ed.post(
        "/api/retrieval/test", json={"question": "Dach"}
    )  # Standard ist oeffentlich
    assert [r["knowledge"] for r in ok.json["results"]] == ["Standard"]
    assert ed.post("/api/retrieval/test", json={"question": " "}).status_code == 400


def test_normal_users_run_pipeline_tests_only_on_readable_knowledge(app, db):
    us = client_for(app, add_user(db, "us", "user"))
    private = knowledge.create_knowledge("Privat")
    res = us.post(
        "/api/retrieval/test",
        json={"question": "Dach", "knowledge_ids": [private["id"]]},
    )
    assert res.status_code in (400, 404) or not res.json.get(
        "results"
    )  # fremdes Knowledge wird nie getestet
