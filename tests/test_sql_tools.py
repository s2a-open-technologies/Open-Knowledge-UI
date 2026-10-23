# ruff: noqa: F811  (Fixtures aus anderen Testmodulen werden importiert und als Argument genutzt)
import json
import sqlite3

import pytest

from app.agent.document_agent import KNOWLEDGE_TOOLS, SQL_TOOLS, select_tools
from app.auth import access
from app.chat.pii_session import PIISession, set_session
from app.core.models import Message
from app.knowledge import service as knowledge
from app.knowledge.sql import format as sqlformat
from app.knowledge.sql.tools import SqlTools
from app.rag import retrieval
from app.rag.retrieval import RetrievalScope
from tests.test_access import (  # noqa: F401
    add_user,
    app,
    chat,
    chat_env,
    client_for,
    conn,
)
from tests.test_pii_tools import FakeFilter
from tests.test_sanitizer import (  # noqa: F401
    FakeAgent,
    Sanitizer,
    chat_setup,
    sanitizer,
    set_settings,
)

SRC = {
    "dialect": "sqlite",
    "database": "shop.db",
    "tables": ["items", "orders"],
    "blocked_columns": {"items": ["salary"]},
}


@pytest.fixture
def shop(sql_worker):
    con = sqlite3.connect(sql_worker / "shop.db")
    con.executescript(
        """
        create table items (id integer primary key, name text, salary integer);
        create table orders (id integer primary key, item_id integer references items(id), customer text);
        insert into items (name, salary) values ('Stuhl', 1), ('Tisch', 2);
        insert into orders (item_id, customer) values (1, 'Anna'), (1, 'Bernd'), (2, 'Anna');
        """
    )
    for i in range(80):
        con.execute("insert into orders (item_id, customer) values (2, ?)", (f"K{i}",))
    con.commit()
    con.close()
    return sql_worker


@pytest.fixture(autouse=True)
def clean_context():
    token = retrieval.set_scope(None)
    set_session(None)
    yield
    retrieval._scope.reset(token)
    set_session(None)


def tools():
    return {t.name: t for t in SqlTools().get_tools()}


def use(*sources):
    retrieval.set_scope(
        RetrievalScope(
            ("kb",),
            None,
            sql_sources=tuple(
                (i + 1, n, json.dumps(c)) for i, (n, c) in enumerate(sources)
            ),
        )
    )


# --- Formatierung -----------------------------------------------------------------
def test_result_text_is_a_markdown_table_with_a_truncation_note():
    text = sqlformat.result_text(
        "Shop", ["a", "b"], [[1, "x|y"], [None, "z\nw"]], True, 2
    )
    assert "| a | b |" in text and "x\\|y" in text and "NULL" in text and "z w" in text
    assert "auf 2 Zeilen gekürzt" in text


def test_schema_text_names_the_sql_dialect_when_known():
    assert "SQL-Dialekt: PostgreSQL" in sqlformat.schema_text("Shop", [], "postgresql")
    assert "SQL-Dialekt" not in sqlformat.schema_text("Shop", [])


def test_schema_text_shows_keys_and_foreign_keys():
    text = sqlformat.schema_text(
        "Shop",
        [
            {
                "name": "o",
                "kind": "table",
                "comment": "Bestellungen",
                "columns": [
                    {
                        "name": "id",
                        "type": "INTEGER",
                        "nullable": False,
                        "pk": True,
                        "comment": "",
                    }
                ],
                "foreign_keys": [
                    {
                        "columns": ["item_id"],
                        "ref_table": "items",
                        "ref_columns": ["id"],
                    }
                ],
            }
        ],
    )
    assert (
        "### o" in text
        and "Bestellungen" in text
        and "id (INTEGER, PK, NOT NULL)" in text
        and "FK: item_id → items(id)" in text
    )


# --- Werkzeuge -----------------------------------------------------------------------
def test_schema_tool_lists_only_allowed_tables_and_columns(db, shop):
    use(("Shop", SRC))
    text = tools()["sql_schema"].invoke({})
    assert (
        "### items" in text
        and "salary" not in text
        and "FK: item_id → items(id)" in text
    )


def test_query_tool_returns_rows_and_truncates(db, shop):
    use(("Shop", SRC))
    text = tools()["sql_query"].invoke(
        {
            "query": "select customer, count(*) as n from orders group by customer order by customer limit 3"
        }
    )
    assert "| Anna | 2 |" in text
    big = tools()["sql_query"].invoke({"query": "select id from orders"})
    assert "gekürzt" in big and big.count("\n| ") <= 52


def test_query_tool_reports_rejections_and_errors_as_text(db, shop):
    use(("Shop", SRC))
    q = tools()["sql_query"]
    assert q.invoke({"query": "delete from orders"}).startswith("FEHLER")
    assert "gesperrt" in q.invoke({"query": "select salary from items"})
    assert "nope" in q.invoke({"query": "select nope from items"})


def test_without_a_source_or_with_two_sources_the_tool_asks(db, shop):
    assert "Keine SQL-Datenbank" in tools()["sql_query"].invoke({"query": "select 1"})
    use(("A", SRC), ("B", SRC))
    assert "Mehrere" in tools()["sql_query"].invoke({"query": "select 1"})
    assert "Unbekannte Quelle" in tools()["sql_query"].invoke(
        {"query": "select 1", "source": "C"}
    )
    assert "| 1 |" in tools()["sql_query"].invoke(
        {"query": "select 1 as x", "source": "b"}
    )


def test_a_down_worker_gives_a_clear_message_without_the_address(db, shop):
    from app.agent.toolworker import client as wc

    def down(*a, **k):
        raise ConnectionError("http://tool-worker-sql:8090")

    wc.set_override(wc.ToolWorkerClient("http://x", "t", down), "sql")
    use(("Shop", SRC))
    text = tools()["sql_query"].invoke({"query": "select 1"})
    assert (
        text.startswith("FEHLER: SQL-Dienst nicht verfügbar")
        and "tool-worker-sql" not in text
    )


# --- Werkzeugwahl ------------------------------------------------------------------
def test_sql_tools_only_appear_with_a_sql_knowledge():
    class T:
        def __init__(self, name):
            self.name = name

    allt = [T(n) for n in (*KNOWLEDGE_TOOLS, *SQL_TOOLS, "web_search_tool")]
    names = lambda ts: {t.name for t in ts}
    assert names(select_tools(allt, True, True)) == names(allt)
    assert names(select_tools(allt, True)) == names(allt) - SQL_TOOLS
    assert (
        names(select_tools(allt, False, True)) == {"web_search_tool"} | SQL_TOOLS
    )  # nur SQL: keine Dokument-Werkzeuge
    assert names(select_tools(allt, False)) == {"web_search_tool"}


# --- PII ---------------------------------------------------------------------------
def test_with_a_filter_arguments_are_unmasked_and_results_masked(db, shop):
    use(("Shop", SRC))
    session = PIISession(FakeFilter(), {"[NAME_1]": "Anna"})
    set_session(session)
    text = tools()["sql_query"].invoke(
        {
            "query": "select customer, count(*) as n from orders where customer = '[NAME_1]' group by customer"
        }
    )
    assert (
        "Anna" not in text and "[NAME_1] | 2" in text
    )  # Bedingung wurde demaskiert, Ergebnis maskiert


def test_a_second_person_in_a_result_gets_its_own_placeholder(db, shop):
    use(("Shop", SRC))
    session = PIISession(FakeFilter(), {"[NAME_1]": "Anna"})
    set_session(session)
    text = tools()["sql_query"].invoke(
        {
            "query": "select distinct customer from orders where customer in ('Anna','Bernd') order by customer"
        }
    )
    assert "Bernd" not in text and "Anna" not in text
    assert (
        session.unmask(text).count("Bernd") == 1
        and session.unmask(text).count("Anna") == 1
    )


class GreedyFilter(FakeFilter):
    """Wie die echte Erkennung auf technischem Text: haelt "IN", "Verzeichnis", "2026" und Orte fuer Entitaeten."""

    NAMES = ("Anna", "Bernd", "IN", "Verzeichnis", "2026", "Stuhl")


def test_only_the_data_values_go_through_the_filter_not_the_text_around_them(db, shop):
    use(("Shop", SRC))
    f = GreedyFilter()
    session = PIISession(f)
    set_session(session)
    text = tools()["sql_query"].invoke(
        {
            "query": "select customer, count(*) as n from orders where customer in ('Anna','Bernd') group by customer"
        }
    )
    assert (
        text.startswith("Quelle „Shop“ · 2 Zeilen") and "| customer | n |" in text
    )  # Kopf und Zeilenzahl unberuehrt
    assert "Anna" not in text and "Bernd" not in text  # Werte sind maskiert
    assert "Quelle" not in "".join(f.calls) and "Zeilen" not in "".join(
        f.calls
    )  # der Filter sah nur die Werte
    assert (
        session.unmask(text).count("Anna") == 1
        and session.unmask(text).count("Bernd") == 1
    )


def test_plain_numbers_stay_readable_and_a_year_is_not_taken_for_a_place(db, shop):
    use(("Shop", SRC))
    f = GreedyFilter()
    set_session(PIISession(f))
    text = tools()["sql_query"].invoke(
        {"query": "select 2026 as jahr, count(*) as n from orders"}
    )
    assert (
        "| 2026 | 83 |" in text and f.calls == []
    )  # reine Zahlen gehen gar nicht erst zum Filter


def test_a_failing_filter_returns_no_rows(db, shop):
    use(("Shop", SRC))
    f = FakeFilter()
    f.fail = True
    set_session(PIISession(f))
    text = tools()["sql_query"].invoke({"query": "select customer from orders"})
    assert text.startswith("FEHLER") and "PII-Filter" in text and "Anna" not in text


def test_without_a_filter_results_are_unchanged(db, shop):
    use(("Shop", SRC))
    set_session(PIISession(None))
    assert "Anna" in tools()["sql_query"].invoke(
        {"query": "select customer from orders limit 1"}
    )


# --- Chat-Scope und Ende-zu-Ende ------------------------------------------------------------
def test_chat_scope_carries_sql_sources_only_for_readable_knowledge(
    app, db, conn, chat_env, shop
):
    from app.llm import connections as store

    sql = knowledge.create_sql_knowledge("Shop", "", SRC)
    alice = add_user(db, "alice", "editor")
    ref = store.make_base_ref(conn["id"], "m")
    c = client_for(app, alice)

    chat(c, model=ref, knowledge_ids=[sql["id"]])
    assert (
        chat_env[-1].sql_sources == ()
    )  # ohne Freigabe unsichtbar, auch wenn ausdruecklich angefordert

    access.add_grant("knowledge", sql["id"], "user", alice["id"], "read")
    chat(c, model=ref)
    scope = chat_env[-1]
    assert [(i, n) for i, n, _cfg in scope.sql_sources] == [(sql["id"], "Shop")]
    assert (
        sql["collection_name"] not in scope.collections
    )  # kein Qdrant-Zugriff auf eine SQL-Quelle


class ToolAgent(FakeAgent):
    """Statt des Modells: ruft die SQL-Werkzeuge mit festen Argumenten auf und streamt eine feste Antwort."""

    def __init__(self, calls, chunks):
        super().__init__(chunks)
        self.calls = calls
        self.results = []

    async def astream_events(self, input_data, version="v2"):
        tools = {t.name: t for t in SqlTools().get_tools()}
        for name, args in self.calls:
            self.results.append(tools[name].invoke(args))
        async for event in super().astream_events(input_data, version):
            yield event


def streamed(res):
    out = ""
    for line in res.get_data(as_text=True).splitlines():
        if line.startswith("data: {"):
            out += json.loads(line[6:]).get("token", "")
    return out


@pytest.fixture
def orders(sql_worker):
    con = sqlite3.connect(sql_worker / "shop.db")
    con.executescript("create table orders (id integer primary key, customer text);")
    con.executemany(
        "insert into orders (customer) values (?)",
        [("Max Mustermann",)] * 3 + [("Anna",)],
    )
    con.commit()
    con.close()


def run_chat(app, db, monkeypatch, pii_url, answer, who="[NAME_1]"):
    agent = chat_setup(app, db, monkeypatch, pii_url)
    ed = add_user(db, "ed", "editor")
    sql = knowledge.create_sql_knowledge(
        "Shop", "", {"dialect": "sqlite", "database": "shop.db"}
    )
    access.add_grant("knowledge", sql["id"], "user", ed["id"], "read")
    tool_agent = ToolAgent(
        [
            (
                "sql_query",
                {"query": f"select count(*) as n from orders where customer = '{who}'"},
            ),
            (
                "sql_query",
                {
                    "query": f"select distinct customer from orders where customer = '{who}'"
                },
            ),
        ],
        [answer],
    )
    monkeypatch.setattr("app.api.routes.get_agent", lambda *a, **k: tool_agent)
    res = client_for(app, ed).post(
        "/chat",
        json={
            "message": "Wie viele Bestellungen hat Max Mustermann?",
            "knowledge_ids": [sql["id"]],
        },
    )
    res.get_data()
    return res, tool_agent, agent


def test_end_to_end_with_a_filter_the_model_never_sees_the_name(
    app, db, orders, monkeypatch, sanitizer
):
    res, agent, _ = run_chat(
        app, db, monkeypatch, sanitizer, "[NAME_1] hat 3 Bestellungen"
    )
    assert "Max Mustermann" not in json.dumps(
        agent.inputs
    )  # das Modell sah nur Platzhalter
    assert (
        "| 3 |" in agent.results[0]
    )  # die Bedingung wurde demaskiert, sonst waere das Ergebnis 0
    assert "Max Mustermann" not in agent.results[1] and "[NAME_1]" in agent.results[1]
    assert streamed(res) == "Max Mustermann hat 3 Bestellungen"  # live demaskiert
    s = db.get_session()
    saved = [m.content for m in s.query(Message).filter_by(role="assistant")]
    s.close()
    assert (
        saved and "Max Mustermann hat 3 Bestellungen" in saved[-1]
    )  # gespeichert wird Klartext


def test_end_to_end_without_a_filter_nothing_is_masked(app, db, orders, monkeypatch):
    Sanitizer.seen = []
    res, agent, _ = run_chat(
        app,
        db,
        monkeypatch,
        "",
        "Max Mustermann hat 3 Bestellungen",
        who="Max Mustermann",
    )
    assert Sanitizer.seen == []
    assert "Max Mustermann" in json.dumps(agent.inputs)
    assert (
        "| 3 |" in agent.results[0] and "Max Mustermann" in agent.results[1]
    )  # ohne Filter unveraendert
    assert streamed(res) == "Max Mustermann hat 3 Bestellungen"


def test_end_to_end_a_failing_filter_stops_before_the_model_and_the_tools(
    app, db, orders, monkeypatch, sanitizer
):
    Sanitizer.behavior = "500"
    res, agent, _base = run_chat(app, db, monkeypatch, sanitizer, "egal")
    assert "HTTP 500" in res.get_data(
        as_text=True
    ) and "nicht gesendet" in res.get_data(as_text=True)
    assert agent.results == [] and agent.inputs == []
