# ruff: noqa: F811  (Fixtures aus anderen Testmodulen werden importiert und als Argument genutzt)
"""Sprache der Stichwortsuche (BM25): app/rag/sparse.py."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

from app.rag import knowledge, model_settings, pipeline
from app.rag import sparse as sparse_lang
from app.rag.retrieval import MultiCollectionRetriever, RetrievalScope, set_scope
from tests.test_access import app, client_for  # noqa: F401
from tests.test_ingest import no_enrichment, qd  # noqa: F401
from tests.test_web_and_vision import ADMIN


def set_language(db, value):
    from app.core.settings_service import SettingsService

    s = db.get_session()
    SettingsService(s).set("BM25_LANGUAGE", value)
    s.close()


class TaggedSparse:
    def __init__(self, language):
        self.language = language


class FakeRouter:
    def __init__(self):
        self.asked = []

    def for_language(self, language):
        self.asked.append(language)
        return TaggedSparse(language)


def test_default_is_german_and_unknown_values_fall_back(db):
    assert sparse_lang.default_language() == "german"
    set_language(db, "english")
    assert sparse_lang.default_language() == "english"
    set_language(db, "klingonisch")
    assert sparse_lang.default_language() == "german"
    assert (
        "german" in sparse_lang.supported_languages()
        and "english" in sparse_lang.supported_languages()
    )


def test_legacy_knowledge_counts_as_english(db):
    assert sparse_lang.knowledge_language({"bm25_language": None}) == "english"
    assert sparse_lang.knowledge_language({"bm25_language": "german"}) == "german"
    assert sparse_lang.knowledge_language(None) == "english"


def test_resolve_leaves_encoders_without_languages_alone():
    fake = object()
    assert sparse_lang.resolve(fake, "german") is fake
    assert sparse_lang.resolve(FakeRouter(), "german").language == "german"


def test_the_german_encoder_matches_word_forms_the_english_one_does_not():
    router = sparse_lang.SparseRouter()
    try:
        german, english = router.for_language("german"), router.for_language("english")
    except Exception as e:  # Modell nicht verfuegbar (kein Netz)
        pytest.skip(f"BM25-Modell nicht ladbar: {e}")

    def tokens(enc, text):
        return set(enc.embed_documents([text])[0].indices)

    assert tokens(german, "Kündigungsfristen") == tokens(german, "Kündigungsfrist")
    assert tokens(english, "Kündigungsfristen") != tokens(english, "Kündigungsfrist")
    assert not tokens(german, "und") and tokens(
        english, "und"
    )  # deutsches Stoppwort entfaellt nur im deutschen Encoder
    assert german is router.for_language("german")  # je Sprache einmal geladen


def test_new_and_reindexed_knowledge_get_the_current_language(db):
    set_language(db, "german")
    kb = knowledge.create_knowledge("Neu")
    assert kb["bm25_language"] == "german" and not model_settings.is_stale(kb)

    set_language(db, "english")
    assert model_settings.is_stale(kb)  # Index in anderer Sprache -> Neu indexieren
    knowledge.swap_collection(kb["id"], "kb_neu_2", model_settings.embedding_spec(), 7)
    assert knowledge.get_knowledge(kb["id"])["bm25_language"] == "english"
    assert not model_settings.is_stale(knowledge.get_knowledge(kb["id"]))


def test_existing_knowledge_without_a_language_is_english_and_stale_under_the_german_default(
    db,
):
    kb = knowledge.get_knowledge(None)
    s = db.get_session()
    from app.core.models import Knowledge

    s.query(Knowledge).filter_by(id=kb["id"]).update({"bm25_language": None})
    s.commit()
    s.close()
    legacy = knowledge.get_knowledge(kb["id"])
    assert sparse_lang.knowledge_language(legacy) == "english"
    assert model_settings.is_stale(legacy)
    set_language(db, "english")
    assert not model_settings.is_stale(legacy)


def test_each_collection_is_searched_in_the_language_it_was_indexed_with(
    db, monkeypatch
):
    seen = []
    monkeypatch.setattr(
        pipeline,
        "run",
        lambda q, collections, cfg, **kw: (
            seen.append((tuple(collections), kw["sparse"].language)) or []
        ),
    )
    router = FakeRouter()
    retriever = MultiCollectionRetriever(
        client=QdrantClient(":memory:"),
        dense_embeddings=object(),
        sparse_embeddings=router,
    )
    set_scope(
        RetrievalScope(
            collections=("alt", "neu", "neu2"),
            sparse_langs=(
                ("neu", "german"),
                ("neu2", "german"),
            ),  # "alt" fehlt: Altbestand = Englisch
        )
    )
    try:
        retriever.invoke("Frage")
    finally:
        set_scope(None)
    assert sorted(seen) == [(("alt",), "english"), (("neu", "neu2"), "german")]


def test_documents_are_indexed_in_the_knowledge_language_attachments_in_the_default(db):
    from app.vector.ingestion import DocumentIngestion

    router = FakeRouter()
    german_kb = SimpleNamespace(
        knowledge={"bm25_language": "german"}, track_documents=True
    )
    legacy_kb = SimpleNamespace(knowledge={"bm25_language": None}, track_documents=True)
    attachment = SimpleNamespace(
        knowledge={"bm25_language": None}, track_documents=False
    )
    set_language(db, "german")
    assert DocumentIngestion._sparse_for_index(german_kb, router).language == "german"
    assert DocumentIngestion._sparse_for_index(legacy_kb, router).language == "english"
    assert (
        DocumentIngestion._sparse_for_index(attachment, router).language == "german"
    )  # Anhaenge: Standardsprache


def test_admin_can_choose_the_language_and_unknown_ones_are_rejected(app, db):
    admin = client_for(app, ADMIN)
    overview = admin.get("/api/admin/embedding").json
    assert (
        overview["ingest"]["bm25_language"] == "german"
        and "german" in overview["bm25_languages"]
    )
    ingest = {k: v for k, v in overview["ingest"].items()}
    res = admin.put(
        "/api/admin/embedding", json={"ingest": {**ingest, "bm25_language": "english"}}
    )
    assert res.status_code == 200 and res.json["ingest"]["bm25_language"] == "english"
    res = admin.put(
        "/api/admin/embedding", json={"ingest": {"bm25_language": "klingonisch"}}
    )
    assert res.status_code == 400 and "unbekannte Sprache" in res.json["error"]


def test_migration_0034_runs_up_down_up(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "m0034", Path("alembic/versions/0034_bm25_language.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SETTING[1] == sparse_lang.DEFAULT_LANGUAGE
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{tmp_path / 'm.db'}"}
    run = lambda *a: subprocess.run(
        [sys.executable, "-m", "alembic", *a], env=env, check=True, capture_output=True
    )
    run("upgrade", "head")
    run("downgrade", "0033")
    run("upgrade", "head")


def test_end_to_end_a_german_index_finds_other_word_forms(db, qd, no_enrichment):
    """Echter BM25-Encoder durch Ingestion und Suche: 'Kündigungsfristen' findet 'Kündigungsfrist'."""
    from tests.test_ingest import FakeDense, ensure_collection, make_ingestion

    router = sparse_lang.SparseRouter()
    try:
        router.for_language("german")
    except Exception as e:
        pytest.skip(f"BM25-Modell nicht ladbar: {e}")
    client = qd
    kb = knowledge.get_knowledge(None)
    kb = {**kb, "bm25_language": "german"}
    ensure_collection(client, "kb_t", FakeDense.dimension)
    docs = {
        "a.md": "# Mietvertrag\n\nDie Kündigungsfrist beträgt drei Monate zum Monatsende.",
        "b.md": "# Wartung\n\nDie Heizungsanlage wird jährlich gewartet und geprüft.",
    }
    ing = make_ingestion(
        client,
        knowledge=kb,
        collection_name="kb_t",
        sparse_embeddings=router,
        embeddings=FakeDense(),
    )
    for i, (name, text) in enumerate(docs.items()):
        ing.index_markdown(text, name, f"g{i}", router)
    cfg = {**__import__("app.rag.profiles", fromlist=["x"]).default_config()}
    cfg = {
        **cfg,
        "search": {**cfg["search"], "mode": "sparse"},
        "rerank": {**cfg["rerank"], "enabled": False},
        "mmr": {**cfg["mmr"], "enabled": False},
        "neighbors": 0,
    }
    hits = pipeline.run(
        "Wie lang sind die Kündigungsfristen?",
        ["kb_t"],
        cfg,
        client=client,
        dense=FakeDense(),
        sparse=sparse_lang.resolve(router, "german"),
        rerank_fn=lambda *a: [],
    )
    assert hits and hits[0].payload["filename"] == "a.md"
