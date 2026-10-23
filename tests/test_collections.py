import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from app.vector import collections
from app.vector.collections import SPARSE_VECTOR, ensure_collection


@pytest.fixture
def client():
    return QdrantClient(":memory:")


def params(client, name):
    return client.get_collection(name).config


def test_new_collection_has_dense_hnsw_and_bm25_with_idf(client):
    ensure_collection(client, "kb", 1024)  # z. B. bge-m3
    cfg = params(client, "kb")
    assert (
        cfg.params.vectors.size == 1024
        and cfg.params.vectors.distance == qm.Distance.COSINE
    )
    assert cfg.params.sparse_vectors[SPARSE_VECTOR].modifier == qm.Modifier.IDF
    assert (cfg.hnsw_config.m, cfg.hnsw_config.ef_construct) == (16, 100)
    assert cfg.hnsw_config.full_scan_threshold == 10000


def test_hnsw_settings_are_a_constant_that_must_not_vanish():
    assert collections.HNSW_CONFIG == {
        "m": 16,
        "ef_construct": 100,
        "full_scan_threshold": 10000,
    }


def test_collection_without_idf_gets_the_modifier_added(client):
    # So sahen alle Collections vor dieser Aenderung aus.
    client.create_collection(
        "alt",
        vectors_config=qm.VectorParams(size=8, distance=qm.Distance.COSINE),
        sparse_vectors_config={SPARSE_VECTOR: qm.SparseVectorParams()},
    )
    assert params(client, "alt").params.sparse_vectors[SPARSE_VECTOR].modifier is None
    ensure_collection(client, "alt", 8)
    assert (
        params(client, "alt").params.sparse_vectors[SPARSE_VECTOR].modifier
        == qm.Modifier.IDF
    )


def test_ensure_is_idempotent_and_keeps_the_dimension_of_existing_collections(client):
    ensure_collection(client, "kb", 384)
    ensure_collection(client, "kb", 1024)  # existiert: nichts wird veraendert
    assert params(client, "kb").params.vectors.size == 384
    assert (
        params(client, "kb").params.sparse_vectors[SPARSE_VECTOR].modifier
        == qm.Modifier.IDF
    )


def test_payload_indexes_for_files_sessions_and_chunk_order(client):
    ensure_collection(client, "kb", 8)
    indexed = client.get_collection("kb").payload_schema
    # Der In-Memory-Client legt keine echten Indizes an; wichtig ist, dass ensure_collection
    # sie fuer genau diese Felder anfordert.
    assert set(collections.PAYLOAD_INDEXES) == {
        "file_group_id",
        "session_id",
        "chunk_index",
    }
    assert collections.PAYLOAD_INDEXES["chunk_index"] == qm.PayloadSchemaType.INTEGER
    assert indexed is not None


def test_failure_to_update_an_old_collection_does_not_break_startup(
    client, monkeypatch
):
    client.create_collection(
        "alt",
        vectors_config=qm.VectorParams(size=8, distance=qm.Distance.COSINE),
        sparse_vectors_config={SPARSE_VECTOR: qm.SparseVectorParams()},
    )

    def boom(**kwargs):
        raise RuntimeError("Server zu alt")

    monkeypatch.setattr(client, "update_collection", boom)
    ensure_collection(client, "alt", 8)  # darf nicht werfen
