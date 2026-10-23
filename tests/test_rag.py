import re
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from app.rag import format as rag_format
from app.rag import pipeline, queries, retrieval
from app.rag import profiles as prof
from app.rag.collections import ensure_collection
from app.rag.retrieval import MultiCollectionRetriever, RetrievalScope

VOCAB = ["kuendigung", "kaution", "miete", "dach", "frist", "strom"]


def tokens(text):
    return re.findall(r"[a-zäöüß]+", text.lower().replace("ü", "ue"))


class FakeDense:
    """Bag-of-Keywords: deterministisch, ohne Modell."""

    def embed_query(self, text):
        toks = tokens(text)
        vec = [float(sum(t.startswith(w[:6]) for t in toks)) for w in VOCAB]
        return vec + [0.01]


class FakeSparse:
    def embed_query(self, text):
        idx = sorted(
            {VOCAB.index(w) for t in tokens(text) for w in VOCAB if t.startswith(w[:6])}
        )
        return SimpleNamespace(indices=idx or [99], values=[1.0] * (len(idx) or 1))


def cfg(**over):
    base = prof.validate_config(
        {"mmr": {"enabled": False}, "search": {"dense_min_score": None}}
    )
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return prof.validate_config(base)


def add_point(client, collection, pid, text, **payload):
    dense, sparse = FakeDense(), FakeSparse()
    sv = sparse.embed_query(text)
    client.upsert(
        collection,
        [
            qm.PointStruct(
                id=pid,
                vector={
                    "": dense.embed_query(text),
                    "langchain-sparse": qm.SparseVector(
                        indices=sv.indices, values=sv.values
                    ),
                },
                payload={
                    "page_content": text,
                    "original_content": text,
                    "filename": "doc.md",
                    **payload,
                },
            )
        ],
    )


@pytest.fixture
def client():
    c = QdrantClient(":memory:")
    ensure_collection(c, "kb", 7)
    return c


@pytest.fixture
def corpus(client):
    add_point(
        client,
        "kb",
        1,
        "Die Kuendigung hat eine Frist von sieben Monaten",
        filename="vertrag.md",
    )
    add_point(
        client,
        "kb",
        2,
        "Die Kaution betraegt drei Monatsmieten miete",
        filename="vertrag.md",
    )
    add_point(
        client,
        "kb",
        3,
        "Das Dach muss im Fruehjahr saniert werden",
        filename="gutachten.md",
    )
    add_point(client, "kb", 4, "Strom Abrechnung jaehrlich", filename="strom.md")
    return client


def search(client, question, config=None, collections=("kb",), **kw):
    trace = pipeline.Trace()
    hits = pipeline.run(
        question,
        list(collections),
        config or cfg(),
        client=client,
        dense=FakeDense(),
        sparse=FakeSparse(),
        trace=trace,
        **kw,
    )
    return hits, trace


def files(hits):
    return [h.filename for h in hits]


# --- Suche -----------------------------------------------------------------


def test_hybrid_search_finds_the_matching_document_first(corpus):
    hits, _ = search(corpus, "Kuendigung Frist")
    assert hits[0].filename == "vertrag.md" and "Kuendigung" in hits[0].text


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_all_search_modes_work(corpus, mode):
    hits, trace = search(corpus, "Dach", cfg(search={"mode": mode}))
    assert hits[0].filename == "gutachten.md"
    assert trace.stages[0]["stage"] == "search" and trace.stages[0]["mode"] == mode


def test_dense_min_score_filters_weak_matches(corpus):
    hits, _ = search(
        corpus, "Dach", cfg(search={"mode": "dense", "dense_min_score": 0.9})
    )
    assert files(hits) == ["gutachten.md"]


def test_fusion_weights_decide_between_dense_and_sparse_preferences(client):
    # Dense bevorzugt A (viele passende Woerter), Sparse bevorzugt B (genau das seltene Wort).
    add_point(client, "kb", 1, "kuendigung kuendigung kuendigung dach", filename="A.md")
    add_point(client, "kb", 2, "dach", filename="B.md")
    question = "kuendigung dach"
    dense_first, _ = search(client, question, cfg(search={"mode": "dense"}))
    sparse_first, _ = search(client, question, cfg(search={"mode": "sparse"}))
    assert dense_first and sparse_first
    heavy_dense, _ = search(
        client, question, cfg(search={"dense_weight": 10, "sparse_weight": 0.1})
    )
    heavy_sparse, _ = search(
        client, question, cfg(search={"dense_weight": 0.1, "sparse_weight": 10})
    )
    assert heavy_dense[0].filename == dense_first[0].filename
    assert heavy_sparse[0].filename == sparse_first[0].filename


def test_dbsf_fusion_runs(corpus):
    hits, _ = search(corpus, "Kaution", cfg(search={"fusion": "dbsf"}))
    assert "Kaution" in hits[0].text


def test_missing_collection_is_reported_not_fatal(corpus):
    hits, trace = search(corpus, "Dach", collections=("kb", "gibt-es-nicht"))
    assert hits and hits[0].filename == "gutachten.md"
    assert any("gibt-es-nicht" in e for e in trace.stages[0]["errors"])


def test_empty_question_or_collections_return_nothing(corpus):
    assert search(corpus, "  ")[0] == []
    assert search(corpus, "Dach", collections=())[0] == []


def test_final_k_limits_results(corpus):
    hits, _ = search(corpus, "kuendigung kaution dach strom", cfg(final_k=2))
    assert len(hits) == 2


# --- Query-Enhancement ------------------------------------------------------


QE = {"mode": "multi_query", "n_queries": 2}


def test_parse_questions_cleans_dedupes_and_limits():
    raw = (
        '<think>egal</think>\n1. Erste Frage?\n- Zweite Frage?\n* "Dritte Frage?"\n\nerste frage?\n'
        + "x" * 400
    )
    assert queries.parse_questions(raw, "Original?", 2) == [
        "Erste Frage?",
        "Zweite Frage?",
    ]
    assert queries.parse_questions("Original?\n original? ", "Original?", 5) == []


def test_prompt_placeholders_are_replaced_without_str_format():
    c = prof.validate_config(
        {
            "query_enhancement": {
                "mode": "multi_query",
                "n_queries": 4,
                "prompt": 'JSON {"a": 1}: {anzahl} Fragen zu {frage}',
            }
        }
    )["query_enhancement"]
    assert queries.build_prompt("Was?", c) == 'JSON {"a": 1}: 4 Fragen zu Was?'


def test_multi_query_finds_documents_the_original_question_misses(corpus):
    seen = []

    def helper(prompt, qe):
        seen.append(prompt)
        return "Kaution Hoehe\nDach Sanierung"

    plain, _ = search(
        corpus, "Kuendigung", cfg(final_k=10, search={"dense_min_score": 0.5})
    )
    enhanced, trace = search(
        corpus,
        "Kuendigung",
        cfg(final_k=10, search={"dense_min_score": 0.5}, query_enhancement=QE),
        helper=helper,
    )

    assert "gutachten.md" not in files(plain)
    assert {"vertrag.md", "gutachten.md"} <= set(files(enhanced))
    stage = trace.stages[0]
    assert stage["stage"] == "query_enhancement"
    assert stage["queries"] == ["Kuendigung", "Kaution Hoehe", "Dach Sanierung"]
    assert "Kuendigung" in seen[0] and "2" in seen[0]


def test_include_original_can_be_switched_off(corpus):
    hits, trace = search(
        corpus,
        "Kuendigung",
        cfg(query_enhancement={**QE, "include_original": False}),
        helper=lambda p, q: "Dach Sanierung",
    )
    assert trace.stages[0]["queries"] == ["Dach Sanierung"]
    assert hits[0].filename == "gutachten.md"


def test_failing_or_missing_helper_falls_back_to_the_original_question(corpus):
    def boom(prompt, qe):
        raise TimeoutError("zu langsam")

    hits, trace = search(corpus, "Dach", cfg(query_enhancement=QE), helper=boom)
    assert hits[0].filename == "gutachten.md"
    assert "TimeoutError" in trace.stages[0]["error"] and trace.stages[0][
        "queries"
    ] == ["Dach"]

    hits, trace = search(corpus, "Dach", cfg(query_enhancement=QE), helper=None)
    assert hits and "Kein Helfer-LLM" in trace.stages[0]["skipped"]


def test_empty_helper_answer_keeps_the_original_question(corpus):
    _, trace = search(
        corpus,
        "Dach",
        cfg(
            query_enhancement=QE,
        ),
        helper=lambda p, q: "  \n ",
    )
    assert trace.stages[0]["queries"] == ["Dach"]


# --- Reranking --------------------------------------------------------------


def test_rerank_orders_by_cross_encoder_score_against_the_original_question(corpus):
    captured = {}

    def fake_rerank(model, question, texts):
        captured.update(model=model, question=question)
        return [10.0 if "Kaution" in t else 1.0 for t in texts]

    hits, trace = search(
        corpus,
        "kuendigung kaution",
        cfg(final_k=3, rerank={"enabled": True, "model": "m-x"}),
        rerank_fn=fake_rerank,
    )
    assert hits[0].text.startswith("Die Kaution") and hits[0].score == 10.0
    assert captured == {
        "model": "default",
        "question": "kuendigung kaution",
    }  # Profil-Modell ist global
    assert "rerank" in [s["stage"] for s in trace.stages]


def test_rerank_uses_the_original_question_even_with_query_enhancement(corpus):
    questions = []
    search(
        corpus,
        "Kuendigung",
        cfg(query_enhancement=QE, rerank={"enabled": True}),
        helper=lambda p, q: "Dach Sanierung",
        rerank_fn=lambda m, q, t: questions.append(q) or [1.0] * len(t),
    )
    assert questions == ["Kuendigung"]


def test_rerank_min_score_drops_weak_candidates(corpus):
    hits, _ = search(
        corpus,
        "kuendigung kaution dach",
        cfg(final_k=5, rerank={"enabled": True, "min_score": 5.0}),
        rerank_fn=lambda m, q, t: [9.0 if "Dach" in x else 1.0 for x in t],
    )
    assert [h.filename for h in hits] == ["gutachten.md"]


def test_rerank_failure_falls_back_to_the_unranked_order(corpus):
    def boom(model, question, texts):
        raise RuntimeError("Modell nicht ladbar")

    hits, trace = search(corpus, "Dach", cfg(rerank={"enabled": True}), rerank_fn=boom)
    assert hits[0].filename == "gutachten.md"
    stage = next(s for s in trace.stages if s["stage"] == "rerank")
    assert "RuntimeError" in stage["error"] and "ohne Reranking" in stage["fallback"]


# --- MMR -------------------------------------------------------------------


def test_mmr_prefers_diverse_results_over_near_duplicates(client):
    for i in range(3):
        add_point(
            client, "kb", i + 1, "kuendigung frist kuendigung", filename=f"dup{i}.md"
        )
    # weniger relevant (Cosine ca. 0.45), aber inhaltlich anders als die Duplikate
    add_point(client, "kb", 10, "kuendigung kaution miete strom", filename="anders.md")

    # Dense-Modus: die Fake-Sparse-Vektoren kennen keine Termgewichte (Gleichstand).
    dense = {"mode": "dense"}
    plain, _ = search(client, "kuendigung", cfg(final_k=2, search=dense))
    diverse, trace = search(
        client,
        "kuendigung",
        cfg(final_k=2, search=dense, mmr={"enabled": True, "lambda_mult": 0.1}),
    )

    assert "anders.md" not in files(plain)
    assert "anders.md" in files(diverse)
    assert any(s["stage"] == "mmr" for s in trace.stages)


# --- Nachbar-Chunks ---------------------------------------------------------


@pytest.fixture
def chunked(client):
    texts = [
        "Einleitung",
        "Abschnitt eins",
        "Abschnitt zwei dach",
        "Abschnitt drei",
        "Abschnitt vier",
        "Schluss",
    ]
    for i, t in enumerate(texts):
        add_point(
            client,
            "kb",
            100 + i,
            t,
            filename="lang.md",
            file_group_id="fg-1",
            chunk_index=i,
            chunk_total=len(texts),
        )
    return client


def test_neighbors_are_added_in_document_order(chunked):
    hits, _trace = search(chunked, "dach", cfg(final_k=1, neighbors=1))
    assert len(hits) == 1
    assert hits[0].text == "Abschnitt eins\n\nAbschnitt zwei dach\n\nAbschnitt drei"
    assert hits[0].payload["_chunks"] == [1, 3]
    assert hits[0].to_document().metadata["chunks"] == [1, 3]


def test_neighbors_respect_document_boundaries(chunked):
    add_point(
        chunked,
        "kb",
        200,
        "Ende kuendigung",
        filename="lang.md",
        file_group_id="fg-1",
        chunk_index=6,
        chunk_total=7,
    )
    hits, _ = search(
        chunked, "Einleitung", cfg(final_k=1, neighbors=2, search={"mode": "sparse"})
    )
    assert not hits or all(c >= 0 for c in (hits[0].payload.get("_chunks") or [0]))
    first, _ = search(
        chunked, "Schluss", cfg(final_k=1, neighbors=3, search={"mode": "dense"})
    )
    assert first == [] or first[0].payload["_chunks"][1] <= 6


def test_adjacent_hits_merge_into_one_passage(chunked):
    hits, _ = search(
        chunked, "abschnitt", cfg(final_k=2, neighbors=1, search={"mode": "dense"})
    )
    assert len(hits) >= 1
    merged = hits[0]
    start, end = merged.payload["_chunks"]
    assert merged.text.count("Abschnitt") == len(re_find(merged.text)) and end > start


def re_find(text):
    return re.findall(r"Abschnitt", text)


def test_hits_without_chunk_index_pass_through_unchanged(corpus):
    hits, trace = search(corpus, "Dach", cfg(final_k=1, neighbors=2))
    assert hits[0].text == "Das Dach muss im Fruehjahr saniert werden"
    assert (
        next(s for s in trace.stages if s["stage"] == "neighbors")[
            "without_chunk_index"
        ]
        == 1
    )


def test_missing_neighbor_chunks_do_not_break_the_passage(client):
    for i in (0, 2):  # Chunk 1 fehlt
        add_point(
            client,
            "kb",
            i + 1,
            f"Teil {i} dach",
            filename="loch.md",
            file_group_id="fg",
            chunk_index=i,
            chunk_total=3,
        )
    hits, _ = search(client, "dach", cfg(final_k=2, neighbors=1))
    assert sorted(h.payload["_chunks"] for h in hits) == [[0, 0], [2, 2]]


# --- Formatierung -----------------------------------------------------------


def test_format_hits_names_the_source():
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content=" Text A ", metadata={"filename": "a.pdf", "chunk_index": 2}
        ),
        Document(
            page_content="Text B", metadata={"filename": "b.md", "chunks": [1, 3]}
        ),
        Document(page_content="Text C", metadata={}),
    ]
    out = rag_format.format_hits(docs)
    assert "[1] Quelle: a.pdf (Abschnitt 3)\nText A" in out
    assert "[2] Quelle: b.md (Abschnitte 2–4)" in out
    assert "[3] Quelle: unbekannt\nText C" in out
    assert rag_format.format_hits([]) == rag_format.NO_RESULTS


# --- Retriever: Scope und Profile ------------------------------------------


@pytest.fixture
def retriever(db, client):
    return MultiCollectionRetriever(
        client=client, dense_embeddings=FakeDense(), sparse_embeddings=FakeSparse()
    )


def with_scope(scope):
    return retrieval.set_scope(scope)


def test_default_scope_is_the_standard_collection():
    assert retrieval.get_scope().collections == ("open_knowledge_ui",)


def test_each_collection_searches_with_its_own_profile(db, retriever, client):
    ensure_collection(client, "kb2", 7)
    add_point(client, "kb", 1, "Dach Sanierung", filename="eins.md")
    add_point(client, "kb", 2, "Dach Daemmung", filename="eins-b.md")
    add_point(client, "kb2", 1, "Dach Reparatur", filename="zwei.md")
    add_point(client, "kb2", 2, "Dach Garantie", filename="drei.md")

    with_scope(
        RetrievalScope(
            ("kb", "kb2"),
            None,
            profiles=(("kb", cfg(final_k=1)), ("kb2", cfg(final_k=3))),
        )
    )
    docs = retriever.invoke("Dach")

    by_collection = [d.metadata["collection"] for d in docs]
    assert by_collection.count("kb") == 1  # Profil von kb: final_k=1
    assert by_collection.count("kb2") == 2  # kb2 liefert beide, hat final_k=3
    assert by_collection[:2] == ["kb", "kb2"]  # Round-Robin


def test_collections_with_equal_profiles_share_one_pipeline_run(
    db, retriever, client, monkeypatch
):
    ensure_collection(client, "kb2", 7)
    calls = []
    real = pipeline.run
    monkeypatch.setattr(
        pipeline,
        "run",
        lambda q, colls, c, **kw: calls.append(list(colls)) or real(q, colls, c, **kw),
    )
    with_scope(
        RetrievalScope(("kb", "kb2"), None, profiles=(("kb", cfg()), ("kb2", cfg())))
    )
    retriever.invoke("Dach")
    assert calls == [["kb", "kb2"]]


def test_attachments_come_first_and_only_from_this_session(db, retriever, client):
    ensure_collection(client, "user_uploads", 7)
    add_point(
        client,
        "user_uploads",
        1,
        "Anhang Dach A",
        filename="anhang.pdf",
        session_id="s-1",
    )
    add_point(
        client,
        "user_uploads",
        2,
        "Anhang Dach B",
        filename="fremd.pdf",
        session_id="s-2",
    )
    add_point(client, "kb", 1, "Wissen Dach", filename="wissen.md")

    with_scope(RetrievalScope(("kb",), "s-1", profiles=(("kb", cfg()),)))
    names = [d.metadata["filename"] for d in retriever.invoke("Dach")]
    assert (
        names[0] == "anhang.pdf" and "fremd.pdf" not in names and "wissen.md" in names
    )


def test_nothing_selected_and_no_session_returns_nothing(db, retriever):
    with_scope(RetrievalScope(()))
    assert retriever.invoke("Dach") == []


def test_helper_in_scope_reaches_the_pipeline(db, retriever, client):
    add_point(client, "kb", 1, "Dach Sanierung", filename="eins.md")
    add_point(client, "kb", 2, "Kaution Hoehe", filename="zwei.md")
    c = cfg(final_k=5, search={"dense_min_score": 0.5}, query_enhancement=QE)
    with_scope(
        RetrievalScope(
            ("kb",), None, profiles=(("kb", c),), helper=lambda p, q: "Kaution Hoehe"
        )
    )
    assert {d.metadata["filename"] for d in retriever.invoke("Dach")} == {
        "eins.md",
        "zwei.md",
    }


def test_tool_output_contains_sources_end_to_end(db, retriever, client):
    from app.agent.tools.custom_tools import CustomTools

    add_point(
        client,
        "kb",
        1,
        "Die Kuendigungsfrist betraegt sieben Monate",
        filename="vertrag.md",
    )
    with_scope(RetrievalScope(("kb",), None, profiles=(("kb", cfg()),)))
    tools = CustomTools(llm=None, retriever=retriever).get_tools()
    search_tool = next(t for t in tools if t.name == "document_search_tool")
    out = search_tool.invoke({"query": "Kuendigungsfrist"})
    assert out.startswith("[1] Quelle: vertrag.md") and "sieben Monate" in out


def test_search_records_sources_with_score_for_the_chat(db, retriever, client):
    from app.rag.format import record_sources

    add_point(
        client, "kb", 1, "Die Kaution betraegt drei Monatsmieten", filename="vertrag.md"
    )
    scope = RetrievalScope(("kb",), None, profiles=(("kb", cfg()),))
    with_scope(scope)
    docs = retriever.invoke("Kaution")
    record_sources(docs)
    record_sources(docs)  # doppelte Treffer nur einmal
    assert len(scope.sources) == 1
    entry = scope.sources[0]
    assert entry["filename"] == "vertrag.md" and isinstance(entry["score"], float)
    assert "Kaution" in entry["snippet"]


def test_source_numbers_are_stable_across_searches_and_appear_in_the_tool_output(
    db, retriever, client
):
    from app.rag.format import format_hits, record_sources

    add_point(
        client, "kb", 1, "Die Kaution betraegt drei Monatsmieten", filename="vertrag.md"
    )
    add_point(
        client,
        "kb",
        2,
        "Die Kuendigungsfrist betraegt drei Monate",
        filename="frist.md",
        chunk_index=1,
    )
    scope = RetrievalScope(("kb",), None, profiles=(("kb", cfg()),))
    with_scope(scope)
    first = retriever.invoke("Kaution")
    nums1 = record_sources(first)
    text = format_hits(first, nums1)
    assert text.startswith(f"[{nums1[0]}] Quelle:")
    second = retriever.invoke("Kuendigungsfrist")
    nums2 = record_sources(second)
    by_file = {e["filename"]: e["n"] for e in scope.sources}
    assert sorted(by_file.values()) == list(range(1, len(by_file) + 1))
    assert nums2[0] == by_file[second[0].metadata["filename"]]
    assert record_sources(first) == nums1  # gleiche Stelle, gleiche Nummer


def test_pages_reach_the_model_output_and_the_source_list(db, retriever, client):
    from app.rag.format import format_hits, record_sources

    add_point(
        client,
        "kb",
        1,
        "Die Kaution betraegt drei Monatsmieten",
        filename="vertrag.pdf",
        page_start=3,
        page_end=4,
    )
    scope = RetrievalScope(("kb",), None, profiles=(("kb", cfg()),))
    with_scope(scope)
    docs = retriever.invoke("Kaution")
    text = format_hits(docs, record_sources(docs))
    assert "Seiten 3–4" in text
    assert scope.sources[0]["page"] == "Seiten 3–4"
