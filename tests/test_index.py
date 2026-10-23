# ruff: noqa: F811  (pytest-Fixtures werden importiert und als Argument genutzt)
from pathlib import Path

import pytest
from flask import Flask
from werkzeug.security import generate_password_hash

from app.agent.tools.custom_tools import CustomTools
from app.api import register_blueprints
from app.auth import access
from app.core.models import User
from app.knowledge import index_doc
from app.knowledge import service as knowledge
from app.rag import retrieval
from app.rag.retrieval import RetrievalScope
from tests.test_ingest import (  # noqa: F401
    DOC,
    FakeDense,
    FakeSparse,
    FakeVision,
    Shared,
    ingest_docx,
    make_docx_with_tables,
    make_ingestion,
    no_enrichment,
    put_file,
    qd,
    set_setting,
)


def add_doc(db, kb, name, markdown, fg="0123456789abcdef", **kw):
    from app.ingest import indexer

    path = indexer.save_markdown(
        knowledge.knowledge_paths(kb).markdown_dir, fg, markdown
    )
    db.add_document(
        fg, name, knowledge_id=kb["id"], markdown_path=str(path), parser_version=1
    )


def default_kb():
    return knowledge.get_knowledge(None)


def paths_of(kb):
    return knowledge.knowledge_paths(kb)


def stored(kb):
    """Das am Knowledge gespeicherte Inhaltsverzeichnis (Text oder None)."""
    return index_doc.get_index(kb["id"])[0]


# --- Inhalt ----------------------------------------------------------------


def test_index_lists_documents_with_outline_tables_and_data_files(db):
    kb = default_kb()
    add_doc(
        db,
        kb,
        "Vertrag.pdf",
        "# Vertrag\n\n## 1 Miete\n\ntext\n\n## 2 Kaution\n\n### 2.1 Hoehe\n\n```\n# kein Titel im Codeblock\n```\n",
        fg="ab12cd34-ffff",
    )
    data = Path("./data")
    data.mkdir(exist_ok=True)
    (data / "Vertrag--ab12cd34--tabelle-1.csv").write_text(
        "posten,betrag\nmiete,950\nkaution,2850\n"
    )
    (data / "umsatz.csv").write_text("monat,betrag,kosten\njan,1,2\nfeb,3,4\nmar,5,6\n")
    (data / "notiz.txt").write_text("nicht aufgelistet")

    text = index_doc.build_index(kb, db.get_all_documents(), paths_of(kb))

    assert text.startswith("# Inhaltsverzeichnis: Standard")
    assert "### Vertrag.pdf" in text and "Eingelesen am" in text
    assert "Gliederung: Vertrag · 1 Miete · 2 Kaution · 2.1 Hoehe" in text
    assert "kein Titel im Codeblock" not in text
    assert (
        "Tabelle: `Vertrag--ab12cd34--tabelle-1.csv` (2 Zeilen; Spalten: posten, betrag)"
        in text
    )
    assert (
        "## Strukturierte Dateien" in text
        and "`umsatz.csv` (3 Zeilen; Spalten: monat, betrag, kosten)" in text
    )
    assert "notiz.txt" not in text
    assert (
        "1 Dokumente · 2 Tabellen/Datendateien" in text
    )  # eine extrahierte + eine eigene


def test_extracted_tables_belong_to_their_own_document_only(db):
    kb = default_kb()
    add_doc(db, kb, "A.pdf", "# A\n\ntext", fg="aaaaaaaa-1111")
    add_doc(db, kb, "B.pdf", "# B\n\ntext", fg="bbbbbbbb-2222")
    data = Path("./data")
    data.mkdir(exist_ok=True)
    (data / "A--aaaaaaaa--tabelle-1.csv").write_text("x,y\n1,2\n3,4\n")
    text = index_doc.build_index(kb, db.get_all_documents(), paths_of(kb))
    a, b = text.split("### B.pdf")
    assert "A--aaaaaaaa--tabelle-1.csv" in a and "tabelle" not in b


def test_other_knowledge_files_carry_the_name_prefix_pandas_expects(db):
    kb = knowledge.create_knowledge("Zweites", "Beschreibung des Wissens")
    paths = paths_of(kb)
    paths.data_dir.mkdir(parents=True)
    (paths.data_dir / "daten.csv").write_text("a,b\n1,2\n")
    text = index_doc.build_index(kb, [], paths)
    assert f"`knowledge_{kb['id']}/daten.csv`" in text
    assert "Beschreibung des Wissens" in text and "Noch keine Dokumente." in text


def test_long_outlines_are_capped(db):
    kb = default_kb()
    md = "# Titel\n\n" + "".join(f"## Abschnitt {i}\n\nx\n\n" for i in range(40))
    add_doc(db, kb, "Lang.pdf", md)
    text = index_doc.build_index(kb, db.get_all_documents(), paths_of(kb))
    assert "Abschnitt 10" in text and "Abschnitt 30" not in text and "· …" in text


# --- Neu erzeugen statt anhaengen -----------------------------------------


def test_refresh_rebuilds_from_the_current_state(db):
    kb = default_kb()
    assert index_doc.refresh_index(kb["id"]) is True
    assert "Noch keine Dokumente." in stored(kb)

    add_doc(db, kb, "Neu.pdf", "# Neu\n\n## Teil", fg="11111111-aaaa")
    index_doc.refresh_index(kb["id"])
    first = stored(kb)
    assert "### Neu.pdf" in first and "Noch keine Dokumente." not in first

    index_doc.refresh_index(kb["id"])  # zweiter Lauf: kein Wachstum
    again = stored(kb)
    assert again.count("### Neu.pdf") == 1 and len(again) - len(first) <= 3

    db.delete_user_document(db.get_all_documents()[0].id)
    index_doc.refresh_index(kb["id"])
    assert "Neu.pdf" not in stored(kb)  # kann nicht veralten


def test_the_index_can_be_switched_off(db):
    kb = default_kb()
    index_doc.refresh_index(kb["id"])
    assert stored(kb)
    set_setting(db, "INGEST_INDEX_ENABLED", "false")
    assert index_doc.refresh_index(kb["id"]) is False and stored(kb) is None


def test_refresh_never_raises(db, monkeypatch):
    monkeypatch.setattr(index_doc, "build_index", lambda *a, **k: 1 / 0)
    assert index_doc.refresh_index(default_kb()["id"]) is False
    assert index_doc.refresh_index(999) is False


# --- Ausloeser -------------------------------------------------------------


def test_ingestion_creates_the_index_with_tables(db, qd, no_enrichment, tmp_path):
    ingest_docx(qd, make_docx_with_tables(tmp_path))
    text = stored(default_kb())
    assert "### preise.docx" in text and "Gliederung: Preisliste" in text
    assert (
        text.count("Tabelle: `preise--") == 2
        and "Spalten: Monat, Umsatz, Kosten" in text
    )


def test_second_ingest_extends_the_index_automatically(db, qd, no_enrichment):
    put_file("eins.md", "## Eins\n\nErster Text hier zum Testen.")
    with make_ingestion(qd) as ing:
        ing.ingest_documents()
    put_file("zwei.md", "## Zwei\n\nZweiter Text hier zum Testen.")
    with make_ingestion(qd) as ing:
        ing.ingest_documents()
    text = stored(default_kb())
    assert "### eins.md" in text and "### zwei.md" in text and "2 Dokumente" in text


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
def app(db):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    register_blueprints(app)
    return app


def client_for(app, user):
    c = app.test_client()
    with c.session_transaction() as s:
        s["user_id"] = user["id"]
    return c


def test_api_changes_refresh_the_index(app, db, qd):
    from app.rag.collections import ensure_collection

    kb = default_kb()
    ensure_collection(qd, "open_knowledge_ui", 7)
    app.extensions["client"] = qd
    add_doc(db, kb, "Bericht.pdf", "# Bericht", fg="22222222-bbbb")
    index_doc.refresh_index(kb["id"])
    admin = client_for(app, {"id": 1})
    access.add_grant(
        "knowledge", kb["id"], "user", 1, "write"
    )  # Admins haben keinen Sonderzugriff
    assert "Bericht.pdf" in stored(kb)

    assert (
        admin.delete(f"/api/documents/{db.get_all_documents()[0].id}").status_code
        == 200
    )
    assert "Bericht.pdf" not in stored(kb)

    other = knowledge.create_knowledge("Alt")
    access.add_grant("knowledge", other["id"], "user", 1, "write")
    index_doc.refresh_index(other["id"])
    admin.patch(f"/api/knowledge/{other['id']}", json={"name": "Neuer Name"})
    assert "Inhaltsverzeichnis: Neuer Name" in stored(other)

    (Path("./data")).mkdir(exist_ok=True)
    admin.post(
        "/api/documents/upload",
        data={"files": (__import__("io").BytesIO(b"a,b\n1,2\n"), "neu.csv")},
    )
    assert "`neu.csv`" in stored(kb)


# --- Zugriff durch das Modell ---------------------------------------------


def scope_for(*kbs, **kw):
    indexes = tuple((k["name"], stored(k) or "") for k in kbs)
    return RetrievalScope(
        tuple(k["collection_name"] for k in kbs), None, indexes=indexes, **kw
    )


@pytest.fixture(autouse=True)
def reset_scope():
    token = retrieval.set_scope(None)
    yield
    retrieval._scope.reset(token)


def test_tool_returns_the_indexes_of_the_knowledge_in_scope(db):
    a, b = default_kb(), knowledge.create_knowledge("Zweites")
    for kb in (a, b):
        index_doc.refresh_index(kb["id"])
    retrieval.set_scope(scope_for(a, b))
    tools = {t.name: t for t in CustomTools(llm=None, retriever=None).get_tools()}
    out = tools["read_knowledge_index"].invoke({})
    assert (
        "Inhaltsverzeichnis: Standard" in out and "Inhaltsverzeichnis: Zweites" in out
    )

    retrieval.set_scope(scope_for(b))  # nur ausgewaehltes Wissen
    assert "Standard" not in tools["read_knowledge_index"].invoke({})
    retrieval.set_scope(RetrievalScope(()))
    assert (
        tools["read_knowledge_index"].invoke({}) == "Kein Inhaltsverzeichnis vorhanden."
    )


def test_tool_output_is_capped(db):
    kb = default_kb()
    for i in range(300):
        add_doc(
            db,
            kb,
            f"Dokument {i:03d} mit langem Namen.pdf",
            "# T\n\n## " + "Abschnitt " * 6,
            fg=f"{i:08x}-0000",
        )
    index_doc.refresh_index(kb["id"])
    retrieval.set_scope(scope_for(kb))
    out = index_doc.read_indexes()
    assert len(out) <= index_doc.TOOL_MAX_CHARS + 120 and "gekürzt" in out


def test_inline_hint_only_for_small_indexes(db):
    kb = default_kb()
    index_doc.refresh_index(kb["id"])
    scope = scope_for(kb)
    assert "Inhaltsverzeichnis des durchsuchbaren Wissens" in index_doc.inline_hint(
        scope.indexes
    )
    assert index_doc.inline_hint(scope.indexes, max_chars=10) == ""  # zu gross
    assert index_doc.inline_hint(()) == "" and index_doc.inline_hint((("x", ""),)) == ""


class CapturingAgent:
    messages: list = []

    async def astream_events(self, input_data, *a, **k):
        CapturingAgent.messages.append(input_data["messages"][-1]["content"])
        if False:
            yield None


@pytest.fixture
def chat_env(app, db, monkeypatch):
    from app.api import chat as routes
    from app.llm import connections as store

    CapturingAgent.messages = []
    conn = store.create_connection("C", "http://x/v1", "k")
    monkeypatch.setattr(routes, "get_agent", lambda resolved, tools: CapturingAgent())
    monkeypatch.setattr(routes, "resources_ready", lambda: True)
    app.extensions["agent_tools"] = []
    return store.make_base_ref(conn["id"], "m")


def chat(client, ref, **body):
    client.post(
        "/chat", json={"message": "Welche Dokumente gibt es?", "model": ref, **body}
    ).get_data()
    return CapturingAgent.messages[-1]


def test_a_small_index_is_given_to_the_model_with_the_question(app, db, chat_env):
    kb = default_kb()
    add_doc(db, kb, "Vertrag.pdf", "# Vertrag\n\n## 1 Miete")
    index_doc.refresh_index(kb["id"])
    sent = chat(client_for(app, {"id": 1}), chat_env)
    assert sent.startswith("Welche Dokumente gibt es?") and "### Vertrag.pdf" in sent


def test_a_large_index_is_left_to_the_tool(app, db, chat_env):
    kb = default_kb()
    for i in range(120):
        add_doc(
            db,
            kb,
            f"Dokument {i:03d}.pdf",
            "# T\n\n## " + "Teil " * 5,
            fg=f"{i:08x}-1111",
        )
    index_doc.refresh_index(kb["id"])
    assert chat(client_for(app, {"id": 1}), chat_env) == "Welche Dokumente gibt es?"


def test_the_index_never_reveals_knowledge_the_user_cannot_read(app, db, chat_env):
    private = knowledge.create_knowledge("Geheimprojekt")
    add_doc(db, private, "Gehaltsliste.pdf", "# Gehälter", fg="99999999-cccc")
    index_doc.refresh_index(private["id"])
    index_doc.refresh_index(default_kb()["id"])

    stranger = client_for(app, add_user(db, "fremd", "editor"))
    sent = chat(stranger, chat_env, knowledge_ids=[private["id"], default_kb()["id"]])
    assert (
        "Gehaltsliste" not in sent
        and "Geheimprojekt" not in sent
        and "Standard" in sent
    )

    owner = add_user(db, "chef", "editor")
    access.add_grant("knowledge", private["id"], "user", owner["id"], "read")
    assert "Gehaltsliste.pdf" in chat(
        client_for(app, owner), chat_env, knowledge_ids=[private["id"]]
    )


def test_no_index_is_sent_when_it_is_switched_off(app, db, chat_env):
    set_setting(db, "INGEST_INDEX_ENABLED", "false")
    index_doc.refresh_index(default_kb()["id"])
    assert chat(client_for(app, {"id": 1}), chat_env) == "Welche Dokumente gibt es?"


# --- Das Verzeichnis haengt am Knowledge -----------------------------------


def test_each_knowledge_keeps_its_own_index_and_says_so(db):
    a, b = default_kb(), knowledge.create_knowledge("Zweites")
    add_doc(db, a, "nur-in-a.pdf", "# A", fg="aaaaaaaa-0000")
    add_doc(db, b, "nur-in-b.pdf", "# B", fg="bbbbbbbb-0000")
    for kb in (a, b):
        index_doc.refresh_index(kb["id"])
    assert "nur-in-a.pdf" in stored(a) and "nur-in-b.pdf" not in stored(a)
    assert "nur-in-b.pdf" in stored(b) and "nur-in-a.pdf" not in stored(b)
    assert "Gilt nur für das Knowledge „Zweites“." in stored(b)
    retrieval.set_scope(
        scope_for(b)
    )  # ein Chat mit Knowledge B liest nie das Verzeichnis von A
    assert "nur-in-a.pdf" not in index_doc.read_indexes()


def test_a_missing_index_is_created_once_on_first_use_and_remote_sources_have_none(db):
    kb = knowledge.create_knowledge("Altbestand")
    assert stored(kb) is None  # z. B. angelegt vor Migration 0029
    text = index_doc.ensure_index(kb)
    assert "Inhaltsverzeichnis: Altbestand" in text and stored(kb) == text
    sql = knowledge.create_sql_knowledge(
        "DB", "", {"dialect": "sqlite", "database": "x.db"}
    )
    assert (
        index_doc.ensure_index(sql) is None
        and index_doc.refresh_index(sql["id"]) is False
    )


def test_the_old_index_file_is_removed_when_the_index_is_rebuilt(db):
    kb = default_kb()
    legacy = index_doc.legacy_index_file(paths_of(kb))
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("alt", encoding="utf-8")
    index_doc.refresh_index(kb["id"])
    assert not legacy.exists() and stored(kb)


def test_same_file_name_in_two_knowledge_gives_two_sources_with_their_knowledge_name(
    db,
):
    from langchain_core.documents import Document

    from app.rag.format import format_hits, record_sources

    a, b = default_kb(), knowledge.create_knowledge("Zweites")
    doc = lambda col: Document(
        page_content="Text",
        metadata={"filename": "vertrag.pdf", "collection": col, "chunk_index": 0},
    )
    scope = RetrievalScope(
        (a["collection_name"], b["collection_name"]),
        None,
        knowledge_names=(
            (a["collection_name"], a["name"]),
            (b["collection_name"], "Zweites"),
        ),
    )
    retrieval.set_scope(scope)
    docs = [doc(a["collection_name"]), doc(b["collection_name"])]
    numbers = record_sources(docs)
    assert numbers == [1, 2] and [e["knowledge"] for e in scope.sources] == [
        a["name"],
        "Zweites",
    ]
    out = format_hits(docs, numbers)
    assert f"Knowledge „{a['name']}“" in out and "Knowledge „Zweites“" in out
    retrieval.set_scope(
        RetrievalScope(
            (a["collection_name"],),
            None,
            knowledge_names=((a["collection_name"], a["name"]),),
        )
    )
    assert "Knowledge „" not in format_hits(
        docs[:1], [1]
    )  # bei einem einzigen Knowledge unnoetig
