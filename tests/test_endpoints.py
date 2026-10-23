import pytest

from app.llm import connections as store
from app.rag import endpoints
from app.rag import models_runtime as mr
from tests.fake_inference import DIM, TEI, Infinity, serve


@pytest.fixture
def tei_embedding():
    url, stop, cls = serve(TEI, mode="embedding")
    yield url, cls
    stop()


@pytest.fixture
def tei_reranker():
    url, stop, cls = serve(TEI, mode="reranker", model_id="BAAI/bge-reranker-v2-m3")
    yield url, cls
    stop()


@pytest.fixture
def infinity():
    url, stop, cls = serve(Infinity)
    yield url, cls
    stop()


def conn(kind, url, key=""):
    return {"id": 1, "kind": kind, "base_url": url, "api_key": key}


# --- Modelle und Fähigkeiten -------------------------------------------------------------
def test_tei_describes_its_single_model_and_type(tei_embedding, tei_reranker):
    assert endpoints.describe(conn("tei", tei_embedding[0])) == [
        {"id": "BAAI/bge-m3", "capabilities": ["embedding"]}
    ]
    assert endpoints.describe(conn("tei", tei_reranker[0])) == [
        {"id": "BAAI/bge-reranker-v2-m3", "capabilities": ["reranker"]}
    ]


def test_infinity_lists_models_with_capabilities(infinity):
    assert endpoints.describe(conn("infinity", infinity[0])) == [
        {"id": "emb-model", "capabilities": ["embedding"]},
        {"id": "rr-model", "capabilities": ["reranker"]},
    ]


def test_a_base_url_ending_in_v1_still_finds_servers_without_it(
    tei_embedding, infinity
):
    assert (
        endpoints.describe(conn("tei", tei_embedding[0] + "/v1"))[0]["id"]
        == "BAAI/bge-m3"
    )
    assert len(endpoints.describe(conn("infinity", infinity[0] + "/v1"))) == 2


# --- Embedding und Rerank im jeweiligen Format -------------------------------------------
def test_tei_embeds_with_inputs_and_keeps_order(tei_embedding):
    url, cls = tei_embedding
    vectors = endpoints.embed(conn("tei", url), "BAAI/bge-m3", ["a b", "Miete Wohnung"])
    assert len(vectors) == 2 and len(vectors[0]) == DIM and vectors[1][2] == 1.0
    assert cls.seen[-1][0] == "/embed" and cls.seen[-1][1]["inputs"] == [
        "a b",
        "Miete Wohnung",
    ]


def test_tei_rerank_uses_texts_and_maps_scores_back_to_the_input_order(tei_reranker):
    url, cls = tei_reranker
    scores = endpoints.rerank(
        conn("tei", url),
        "m",
        "Wie hoch ist die Miete",
        ["Das Wetter ist schön", "Die Miete ist hoch"],
    )
    assert (
        scores[1] > scores[0]
    )  # TEI liefert nach Score sortiert, wir ordnen nach index
    assert "texts" in cls.seen[-1][1] and "documents" not in cls.seen[-1][1]


def test_infinity_embeds_and_reranks_with_the_model_name(infinity):
    url, cls = infinity
    assert len(endpoints.embed(conn("infinity", url), "emb-model", ["x"])[0]) == DIM
    scores = endpoints.rerank(
        conn("infinity", url), "rr-model", "Miete", ["Wetter", "Miete hoch"]
    )
    assert scores[1] > scores[0]
    assert cls.seen[-1][1]["model"] == "rr-model" and "documents" in cls.seen[-1][1]


def test_large_inputs_are_sent_in_batches(tei_embedding):
    url, cls = tei_embedding
    endpoints.embed(conn("tei", url), "m", [f"t{i}" for i in range(70)])
    sizes = [len(b["inputs"]) for p, b in cls.seen if p == "/embed"]
    assert sum(sizes) == 70 and max(sizes) <= mr.EMBED_BATCH


def test_api_key_is_sent_as_bearer_and_wrong_key_is_reported():
    url, stop, _cls = serve(TEI, mode="embedding", api_key="geheim")
    try:
        assert endpoints.embed(conn("tei", url, "geheim"), "m", ["x"])
        with pytest.raises(endpoints.EndpointError) as e:
            endpoints.embed(conn("tei", url, "falsch"), "m", ["x"])
        assert "API-Key" in str(e.value)
    finally:
        stop()


def test_wrong_model_type_is_a_clear_error(tei_reranker):
    with pytest.raises(endpoints.EndpointError):
        endpoints.embed(conn("tei", tei_reranker[0]), "m", ["x"])


# --- Test an der Connection ----------------------------------------------------------------
def test_connection_test_reports_embedding_for_a_tei_embedding_server(tei_embedding):
    out = endpoints.test_connection(conn("tei", tei_embedding[0]))
    assert out["success"] and out["embedding"] == {
        "model": "BAAI/bge-m3",
        "dimension": DIM,
        "ok": True,
    }
    assert out["reranker"] is None and "Embedding" in out["message"]


def test_connection_test_checks_the_ranking_for_a_reranker(tei_reranker):
    out = endpoints.test_connection(conn("tei", tei_reranker[0]))
    assert out["success"] and out["reranker"]["ok"] is True and out["embedding"] is None


def test_connection_test_covers_both_apis_of_infinity(infinity):
    out = endpoints.test_connection(conn("infinity", infinity[0]))
    assert out["success"]
    assert (
        out["embedding"]["model"] == "emb-model"
        and out["embedding"]["dimension"] == DIM
    )
    assert out["reranker"] == {"model": "rr-model", "ok": True}


def test_connection_test_fails_clearly_when_unreachable_or_key_is_wrong():
    out = endpoints.test_connection(conn("tei", "http://127.0.0.1:9"))
    assert out["success"] is False and "127.0.0.1" not in out["error"]
    url, stop, _ = serve(TEI, mode="embedding", api_key="geheim")
    try:
        bad = endpoints.test_connection(conn("tei", url, "falsch"))
        assert bad["success"] is False and "API-Key" in bad["error"]
    finally:
        stop()


# --- Anbindung an das Modell-Laufzeitmodul --------------------------------------------------
def test_runtime_uses_the_tei_and_infinity_formats_through_connections(
    db, tei_embedding, tei_reranker, infinity
):
    t_emb = store.create_connection("TEI E", tei_embedding[0], "", kind="tei")
    t_rr = store.create_connection("TEI R", tei_reranker[0], "", kind="tei")
    inf = store.create_connection("Inf", infinity[0], "", kind="infinity")
    assert (
        len(
            mr.get_embedder(
                mr.ModelSpec("endpoint", "BAAI/bge-m3", t_emb["id"])
            ).embed_query("x")
        )
        == DIM
    )
    rr = mr.get_reranker(
        mr.ModelSpec("endpoint", "BAAI/bge-reranker-v2-m3", t_rr["id"])
    )
    s = rr.score("Miete", ["Wetter", "Die Miete"])
    assert s[1] > s[0]
    assert (
        mr.get_embedder(mr.ModelSpec("endpoint", "emb-model", inf["id"])).dimension
        == DIM
    )
    assert (
        mr.get_reranker(mr.ModelSpec("endpoint", "rr-model", inf["id"])).score(
            "Miete", ["a", "Miete"]
        )[1]
        > 0
    )
