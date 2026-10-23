import sqlite3
import time

import pytest
from flask import Flask

from app.api.admin_routes import admin_bp
from app.api.api_routes import api_bp
from app.api.routes import bp as main_bp
from app.chat.pii_session import PIISession, set_session
from app.knowledge import service as knowledge
from app.knowledge.sql import valueindex
from app.knowledge.sql.tools import SqlTools
from app.rag import retrieval
from app.rag.retrieval import RetrievalScope
from tests.test_access import add_user, client_for
from tests.test_pii_tools import FakeFilter

ADMIN = {"id": 1, "role": "admin", "username": "admin"}


@pytest.fixture
def app(db):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    for bp in (main_bp, api_bp, admin_bp):
        app.register_blueprint(bp)
    return app


@pytest.fixture
def shop(sql_worker):
    con = sqlite3.connect(sql_worker / "shop.db")
    con.executescript(
        "create table items (id integer primary key, name text, category text, email text, note text, salary integer);"
    )
    names = (
        ["Stuhl"] * 5
        + ["Stühle"] * 2
        + ["Bürostuhl"] * 3
        + ["Tisch"] * 4
        + ["Lampe"] * 2
    )
    for i, n in enumerate(names):
        con.execute(
            "insert into items (name, category, email, note, salary) values (?,?,?,?,?)",
            (
                n,
                "Möbel" if "isch" in n or "tuhl" in n or "tühle" in n else "Licht",
                f"u{i}@x.de",
                f"Freitext {i} " * 3,
                i,
            ),
        )
    con.commit()
    con.close()
    return sql_worker


def make_source(app, **over):
    src = {"dialect": "sqlite", "database": "shop.db"}
    src.update(over)
    r = client_for(app, ADMIN).post(
        "/api/knowledge", json={"name": "Shop", "kind": "sql", "source": src}
    )
    assert r.status_code == 201, r.json
    return r.json["id"]


def wait_built(app, kid, timeout=15):
    c = client_for(app, ADMIN)
    end = time.time() + timeout
    while time.time() < end:
        st = c.get(f"/api/knowledge/{kid}/sql-values").json
        if st["status"] in ("ready", "error"):
            return st
        time.sleep(0.1)
    raise AssertionError("Aufbau dauert zu lange")


# --- Vorschlag ---------------------------------------------------------------------------------
def test_suggestions_include_categories_and_exclude_keys_contacts_and_free_text():
    prof = [
        {
            "table": "t",
            "column": "kategorie",
            "type": "TEXT",
            "rows": 1000,
            "distinct": 12,
            "non_null": 1000,
        },
        {
            "table": "t",
            "column": "email",
            "type": "TEXT",
            "rows": 1000,
            "distinct": 12,
            "non_null": 1000,
        },
        {
            "table": "t",
            "column": "kunde_id",
            "type": "TEXT",
            "rows": 1000,
            "distinct": 12,
            "non_null": 1000,
        },
        {
            "table": "t",
            "column": "bemerkung",
            "type": "TEXT",
            "rows": 1000,
            "distinct": 900,
            "non_null": 1000,
        },
        {
            "table": "t",
            "column": "titel",
            "type": "TEXT",
            "rows": 1000,
            "distinct": 800,
            "non_null": 1000,
        },
        {
            "table": "t",
            "column": "leer",
            "type": "TEXT",
            "rows": 10,
            "distinct": 0,
            "non_null": 0,
        },
        {
            "table": "t",
            "column": "klein",
            "type": "TEXT",
            "rows": 10,
            "distinct": 9,
            "non_null": 10,
        },
    ]
    by = {s["column"]: s for s in valueindex.suggest(prof, max_distinct=850)}
    assert by["kategorie"]["include"] is True
    assert by["email"]["include"] is False and "Kontakt" in by["email"]["reason"]
    assert by["kunde_id"]["include"] is False
    assert (
        by["bemerkung"]["include"] is False
        and "verschiedene" in by["bemerkung"]["reason"]
    )
    assert by["titel"]["include"] is False and "eindeutig" in by["titel"]["reason"]
    assert by["leer"]["include"] is False
    assert (
        by["klein"]["include"] is True
    )  # kleine Tabellen: viele eindeutige Werte sind normal


# --- API und Aufbau -----------------------------------------------------------------------------
def test_full_flow_suggest_save_build_and_search(app, db, shop):
    kid = make_source(app)
    admin = client_for(app, ADMIN)
    sug = admin.post(f"/api/knowledge/{kid}/sql-values/suggest").json
    by = {(c["table"], c["column"]): c for c in sug["columns"]}
    assert by[("items", "name")]["include"] and by[("items", "category")]["include"]
    assert by[("items", "email")]["include"] is False

    saved = admin.put(
        f"/api/knowledge/{kid}/sql-values",
        json={"columns": {"items": ["name", "category"]}},
    )
    assert saved.status_code == 200 and saved.json["columns"] == {
        "items": ["name", "category"]
    }
    assert admin.post(f"/api/knowledge/{kid}/sql-values/build").status_code == 202
    st = wait_built(app, kid)
    assert (
        st["status"] == "ready"
        and st["value_count"] == 5 + 2
        and st["column_count"] == 2
    )  # 5 Namen + 2 Kategorien

    res = admin.post(
        f"/api/knowledge/{kid}/sql-values/search", json={"term": "Stühle"}
    ).json["results"]
    values = [r["value"] for r in res]
    assert values[:2] == ["Stühle", "Stuhl"] or set(values[:3]) == {
        "Stühle",
        "Stuhl",
        "Bürostuhl",
    }
    assert (
        {"Stuhl", "Stühle", "Bürostuhl"} <= set(values)
        and "Tisch" not in values
        and "Lampe" not in values
    )
    assert (
        next(r for r in res if r["value"] == "Stuhl")["count"] == 5
    )  # Haeufigkeit stammt aus der Quelle


def test_a_failed_rebuild_keeps_the_old_directory(app, db, shop):
    from app.agent.toolworker import client as wc

    kid = make_source(app)
    admin = client_for(app, ADMIN)
    admin.put(f"/api/knowledge/{kid}/sql-values", json={"columns": {"items": ["name"]}})
    admin.post(f"/api/knowledge/{kid}/sql-values/build")
    assert wait_built(app, kid)["status"] == "ready"

    def down(*a, **k):
        raise ConnectionError("weg")

    wc.set_override(wc.ToolWorkerClient("http://x", "t", down), "sql")
    try:
        admin.post(f"/api/knowledge/{kid}/sql-values/build")
        st = wait_built(app, kid)
    finally:
        wc.set_override(None, "sql")
    assert st["status"] == "error" and st["error"]
    assert admin.post(
        f"/api/knowledge/{kid}/sql-values/search", json={"term": "Stuhl"}
    ).json["results"]  # altes Verzeichnis


def test_columns_that_are_not_categories_are_skipped_with_a_note(
    app, db, shop, monkeypatch
):
    kid = make_source(app)
    admin = client_for(app, ADMIN)
    from app.agent.toolworker import config as wc

    monkeypatch.setattr(
        wc, "values_settings", lambda: (3, 20)
    )  # mehr als 3 verschiedene Werte = kein Kategoriefeld
    admin.put(
        f"/api/knowledge/{kid}/sql-values",
        json={"columns": {"items": ["name", "category"]}},
    )
    admin.post(f"/api/knowledge/{kid}/sql-values/build")
    st = wait_built(app, kid)
    assert (
        st["status"] == "ready"
        and "items.name" in st["error"]
        and st["value_count"] == 2
    )  # nur category


def test_selection_is_validated_against_allowed_tables_and_blocked_columns(
    app, db, shop
):
    kid = make_source(app, tables=["items"], blocked_columns={"items": ["email"]})
    admin = client_for(app, ADMIN)
    assert (
        admin.put(
            f"/api/knowledge/{kid}/sql-values", json={"columns": {"items": ["email"]}}
        ).status_code
        == 400
    )
    assert (
        admin.put(
            f"/api/knowledge/{kid}/sql-values", json={"columns": {"andere": ["x"]}}
        ).status_code
        == 400
    )
    assert (
        admin.put(
            f"/api/knowledge/{kid}/sql-values", json={"columns": {"items": ["na;me"]}}
        ).status_code
        == 400
    )
    assert (
        admin.post(f"/api/knowledge/{kid}/sql-values/build").status_code == 400
    )  # nichts gewaehlt


def test_only_admins_manage_the_directory_and_only_for_sql_sources(app, db, shop):
    kid = make_source(app)
    ed = client_for(app, add_user(db, "ed", "editor"))
    for call in (
        lambda: ed.get(f"/api/knowledge/{kid}/sql-values"),
        lambda: ed.post(f"/api/knowledge/{kid}/sql-values/suggest"),
        lambda: ed.put(f"/api/knowledge/{kid}/sql-values", json={"columns": {}}),
        lambda: ed.post(f"/api/knowledge/{kid}/sql-values/build"),
        lambda: ed.post(f"/api/knowledge/{kid}/sql-values/search", json={"term": "x"}),
    ):
        assert call().status_code == 403
    plain = knowledge.create_knowledge("Doku")
    assert (
        client_for(app, ADMIN)
        .get(f"/api/knowledge/{plain['id']}/sql-values")
        .status_code
        == 404
    )


def test_editing_the_source_keeps_the_column_selection_and_deleting_removes_the_values(
    app, db, shop
):
    kid = make_source(app)
    admin = client_for(app, ADMIN)
    admin.put(f"/api/knowledge/{kid}/sql-values", json={"columns": {"items": ["name"]}})
    admin.post(f"/api/knowledge/{kid}/sql-values/build")
    wait_built(app, kid)
    r = admin.patch(
        f"/api/knowledge/{kid}",
        json={"source": {"dialect": "sqlite", "database": "shop.db"}},
    )
    assert r.status_code == 200
    assert admin.get(f"/api/knowledge/{kid}/sql-values").json["columns"] == {
        "items": ["name"]
    }  # Dialog kennt sie nicht
    assert admin.delete(f"/api/knowledge/{kid}").status_code == 200
    from app.core.models import SqlValue, SqlValueIndex

    s = db.get_session()
    assert s.query(SqlValue).count() == 0 and s.query(SqlValueIndex).count() == 0
    s.close()


# --- Werkzeug ---------------------------------------------------------------------------------------
@pytest.fixture
def built(app, db, shop):
    kid = make_source(app)
    admin = client_for(app, ADMIN)
    admin.put(
        f"/api/knowledge/{kid}/sql-values",
        json={"columns": {"items": ["name", "category"]}},
    )
    admin.post(f"/api/knowledge/{kid}/sql-values/build")
    wait_built(app, kid)
    cfg = knowledge.get_knowledge(kid)["source_config"]
    token = retrieval.set_scope(
        RetrievalScope(("kb",), None, sql_sources=((kid, "Shop", cfg),))
    )
    set_session(None)
    yield kid
    retrieval._scope.reset(token)
    set_session(None)


def find(**args):
    return {t.name: t for t in SqlTools().get_tools()}["sql_find_values"].invoke(args)


def test_the_tool_lists_exact_spellings_with_counts(built):
    text = find(term="Stühle")
    assert (
        "items.name:" in text
        and "Stuhl (5)" in text
        and "Stühle (2)" in text
        and "Bürostuhl (3)" in text
    )
    assert "Tisch" not in text and "IN (...)" in text


def test_the_tool_can_be_narrowed_and_reports_no_hits_and_missing_directories(
    built, app, db, shop
):
    assert "items.category" not in find(term="Möbel", table="items", column="name")
    assert "keinen passenden Wert" in find(term="Gurke")
    empty = make_source(app)  # ohne Wertverzeichnis
    cfg = knowledge.get_knowledge(empty)["source_config"]
    retrieval.set_scope(
        RetrievalScope(("kb",), None, sql_sources=((empty, "Leer", cfg),))
    )
    assert "kein Wertverzeichnis" in find(term="Stuhl") and "LIKE" in find(term="Stuhl")


def test_with_a_filter_terms_are_unmasked_and_results_masked(built):
    class NameFilter(FakeFilter):
        NAMES = ("Stuhl",)

    session = PIISession(NameFilter(), {"[NAME_1]": "Stuhl"})
    set_session(session)
    text = find(term="[NAME_1]")  # das Modell sah nur den Platzhalter
    assert "Stuhl" not in text.replace("[NAME_", "") and "[NAME_1]" in text
    assert session.unmask(text).count("Stuhl") >= 1


def test_find_values_masks_only_the_values_and_keeps_the_instructions_intact(built):
    from tests.test_pii_tools import FakeFilter as Base

    class Greedy(Base):
        NAMES = ("Stuhl", "Verzeichnis", "IN", "Wertverzeichnis")

    f = Greedy()
    session = PIISession(f)
    set_session(session)
    text = find(term="Stuhl")
    assert (
        "Setze die passenden Werte mit IN (...) in sql_query ein." in text
    )  # Anweisung unzerstoert
    assert (
        "items.name:" in text and "Stuhl (5)" not in text
    )  # nur der Wert ist maskiert
    assert session.unmask(text).count("Stuhl (5)") == 1
    assert all(
        "Setze" not in c and "Quelle" not in c for c in f.calls
    )  # der Filter sah nur Werte
    miss = find(
        term="[NAME_9]"
    )  # kein Treffer: Hinweis bleibt, der Begriff steht wie gesendet da
    assert "keinen passenden Wert im Wertverzeichnis" in miss and "[NAME_9]" in miss


def test_the_chat_hint_mentions_the_value_search_only_when_a_directory_exists(
    app, db, shop, built, monkeypatch
):
    from app.agent.toolworker import config as wc
    from app.auth import access
    from tests.test_sanitizer import chat_setup

    kid = built
    agent = chat_setup(app, db, monkeypatch, "")
    ed = add_user(db, "ed2", "editor")
    access.add_grant("knowledge", kid, "user", ed["id"], "read")
    client_for(app, ed).post(
        "/chat", json={"message": "Wie viele Stühle?", "knowledge_ids": [kid]}
    ).get_data()
    assert wc.values_hint() in agent.inputs[0]["messages"][-1]["content"]
    other = make_source(app)  # ohne Verzeichnis
    access.add_grant("knowledge", other, "user", ed["id"], "read")
    agent.inputs.clear()
    client_for(app, ed).post(
        "/chat", json={"message": "Hallo", "knowledge_ids": [other]}
    ).get_data()
    assert wc.values_hint() not in agent.inputs[0]["messages"][-1]["content"]


# --- Sperren nach dem Aufbau: gesperrte Werte duerfen nie mehr ausgeliefert werden -------------------
def build_names(app, kid):
    admin = client_for(app, ADMIN)
    admin.put(
        f"/api/knowledge/{kid}/sql-values",
        json={"columns": {"items": ["name", "category"]}},
    )
    admin.post(f"/api/knowledge/{kid}/sql-values/build")
    return wait_built(app, kid)


def search_values(app, kid, term):
    return [
        r["value"]
        for r in client_for(app, ADMIN)
        .post(f"/api/knowledge/{kid}/sql-values/search", json={"term": term})
        .json["results"]
    ]


def test_blocking_a_column_later_removes_its_values_and_selection(app, db, shop):
    kid = make_source(app)
    build_names(app, kid)
    assert "Stuhl" in search_values(app, kid, "Stuhl")
    admin = client_for(app, ADMIN)
    r = admin.patch(
        f"/api/knowledge/{kid}",
        json={
            "source": {
                "dialect": "sqlite",
                "database": "shop.db",
                "blocked_columns": {"items": ["name"]},
            }
        },
    )
    assert r.status_code == 200, r.json
    assert (
        search_values(app, kid, "Stuhl") == []
    )  # Werte der gesperrten Spalte sind weg
    info = admin.get(f"/api/knowledge/{kid}/sql-values").json
    assert (
        info["columns"] == {"items": ["category"]} and info["value_count"] == 2
    )  # Auswahl und Zaehlung angepasst
    assert search_values(app, kid, "Möbel")  # andere Spalte bleibt


def test_removing_a_table_from_the_allowed_list_removes_its_values(app, db, shop):
    kid = make_source(app)
    build_names(app, kid)
    admin = client_for(app, ADMIN)
    admin.patch(
        f"/api/knowledge/{kid}",
        json={
            "source": {"dialect": "sqlite", "database": "shop.db", "tables": ["andere"]}
        },
    )
    assert search_values(app, kid, "Stuhl") == []
    assert admin.get(f"/api/knowledge/{kid}/sql-values").json["columns"] == {}


def test_search_filters_by_the_current_config_even_if_rows_remain(app, db, shop):
    """Verteidigung in der Tiefe: auch wenn Zeilen im Verzeichnis uebrig blieben, liefert die Suche sie nicht."""
    from app.knowledge.service import update_sql_source
    from app.knowledge.sql.config import SqlConfig, parse_config

    kid = make_source(app)
    build_names(app, kid)
    cfg = parse_config(knowledge.get_knowledge(kid)["source_config"])
    update_sql_source(
        kid,
        SqlConfig(
            **{**cfg.__dict__, "blocked_columns": {"items": ("name",)}}
        ).to_dict(),
    )  # Umgehung: ohne API/Prune
    assert search_values(app, kid, "Stuhl") == []
