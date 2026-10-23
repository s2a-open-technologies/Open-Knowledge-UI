import json

import pytest
from flask import Flask
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from app.auth import access
from app.rag import knowledge
from app.rag import models_runtime as mr
from app.rag.retrieval import MultiCollectionRetriever, RetrievalScope
from app.vector import sources
from app.web.admin_routes import admin_bp
from app.web.api_routes import api_bp
from app.web.routes import bp as main_bp
from tests.test_access import add_user, client_for
from tests.test_rag import FakeDense as KeywordDense
from tests.test_rag import FakeSparse, cfg, with_scope

ADMIN = {"id": 1, "role": "admin", "username": "admin"}
URL = "http://fremd.example:6333"
SPEC = {"source": "hf", "model": "fremd/modell"}


class Embedder(KeywordDense):
    dimension = 7

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


class Shared:
    """Fremder In-Memory-Server, der close() ueberlebt und nur Lesezugriffe erlaubt."""

    def __init__(self, client):
        self._c = client
        self.writes = []

    def close(self):
        pass

    def __getattr__(self, name):
        if name in {
            "upsert",
            "delete",
            "delete_collection",
            "create_collection",
            "update_collection",
        }:

            def refuse(*a, **k):
                self.writes.append(name)
                raise AssertionError(f"Schreibzugriff auf eine externe Quelle: {name}")

            return refuse
        return getattr(self._c, name)


def fill(client, collection, points, *, dense_name="", sparse_name=None, size=7):
    vectors = {dense_name: qm.VectorParams(size=size, distance=qm.Distance.COSINE)}
    sparse = {sparse_name: qm.SparseVectorParams()} if sparse_name else None
    client.create_collection(
        collection,
        vectors_config=vectors if dense_name else vectors[""],
        sparse_vectors_config=sparse,
    )
    dense, sp = KeywordDense(), FakeSparse()
    pts = []
    for i, (text, payload) in enumerate(points, 1):
        vec = dense.embed_query(text)
        vector = {dense_name: vec} if dense_name else vec
        if sparse_name:
            sv = sp.embed_query(text)
            vector = {
                **(vector if isinstance(vector, dict) else {"": vec}),
                sparse_name: qm.SparseVector(indices=sv.indices, values=sv.values),
            }
        pts.append(qm.PointStruct(id=i, vector=vector, payload=payload))
    client.upsert(collection, pts)


@pytest.fixture
def remote(monkeypatch):
    """Zwei fremde Collections: A ohne Sparse mit anderen Feldnamen, B mit benanntem Dense + Sparse."""
    server = Shared(QdrantClient(":memory:"))
    fill(
        server._c,
        "fremd_a",
        [
            (
                "Die Kaution betraegt drei Monatsmieten",
                {
                    "content": "Die Kaution betraegt drei Monatsmieten",
                    "datei": "mietvertrag.pdf",
                    "seite": 3,
                },
            ),
            (
                "Das Dach wurde saniert",
                {"content": "Das Dach wurde saniert", "datei": "bau.pdf", "seite": 1},
            ),
        ],
    )
    fill(
        server._c,
        "fremd_b",
        [
            (
                "Die Kuendigung erfolgt mit Frist",
                {
                    "page_content": "Die Kuendigung erfolgt mit Frist",
                    "filename": "frist.md",
                },
            ),
            (
                "Strom wird monatlich abgerechnet",
                {
                    "page_content": "Strom wird monatlich abgerechnet",
                    "filename": "strom.md",
                },
            ),
        ],
        dense_name="emb",
        sparse_name="bm",
    )
    monkeypatch.setattr(
        sources, "connect", lambda cfg: (sources._check_url(cfg.url), server)[1]
    )
    monkeypatch.setattr(mr, "get_embedder", lambda spec: Embedder())
    return server


def body_a(**over):
    src = {
        "kind": "qdrant",
        "url": URL,
        "api_key": "geheim",
        "collection": "fremd_a",
        "fields": {"text": "content", "filename": "datei", "page_start": "seite"},
    }
    src.update(over)
    return {"name": "Fremd A", "kind": "external", "source": src, "embedding": SPEC}


def body_b():
    return {
        "name": "Fremd B",
        "kind": "external",
        "embedding": SPEC,
        "source": {
            "url": URL,
            "collection": "fremd_b",
            "dense_vector": "emb",
            "sparse_vector": "bm",
        },
    }


@pytest.fixture
def app(db):
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    for bp in (main_bp, api_bp, admin_bp):
        app.register_blueprint(bp)
    return app


def create(app, body, user=ADMIN):
    return client_for(app, user).post("/api/knowledge", json=body)


# --- Anlegen und Pruefen ------------------------------------------------------


def test_admin_creates_an_external_knowledge_and_the_key_never_leaks(app, db, remote):
    r = create(app, body_a())
    assert r.status_code == 201, r.json
    k = r.json
    assert k["kind"] == "external" and k["collection_name"].startswith("ext_")
    assert k["source_config"]["has_key"] is True and "api_key" not in k["source_config"]
    assert "geheim" not in r.get_data(as_text=True)
    listed = client_for(app, ADMIN).get("/api/knowledge").get_data(as_text=True)
    assert "geheim" not in listed and "ext_" in listed
    assert (
        knowledge.get_knowledge(k["id"])["kind"] == "external"
    )  # intern ist der Key gespeichert
    assert (
        json.loads(knowledge.get_knowledge(k["id"])["source_config"])["api_key"]
        == "geheim"
    )


def test_only_admins_create_external_sources(app, db, remote):
    ed = add_user(db, "ed", "editor")
    assert create(app, body_a(), ed).status_code == 403
    assert (
        client_for(app, ed)
        .post("/api/knowledge/external/test", json={"source": body_a()["source"]})
        .status_code
        == 403
    )


@pytest.mark.parametrize(
    "change,message",
    [
        ({"collection": "gibt_es_nicht"}, "gibt es auf dem Server nicht"),
        ({"dense_vector": "falsch"}, "Vektor 'falsch'"),
        ({"fields": {"text": "nicht_da"}}, "Textfeld"),
        ({"sparse_vector": "nope"}, "Sparse-Vektor"),
        ({"url": "ftp://x"}, "http://"),
        ({"url": "http://169.254.169.254:6333"}, "Link-Local"),
    ],
)
def test_wrong_configuration_is_refused_with_a_reason(app, db, remote, change, message):
    r = create(app, body_a(**change))
    assert r.status_code == 400 and message in r.json["error"]
    assert [k["kind"] for k in knowledge.list_knowledge()] == ["local"]


def test_a_model_with_the_wrong_dimension_is_refused(app, db, remote, monkeypatch):
    class Wrong(Embedder):
        dimension = 384

    monkeypatch.setattr(mr, "get_embedder", lambda spec: Wrong())
    r = create(app, body_a())
    assert r.status_code == 400 and "384" in r.json["error"] and "7" in r.json["error"]


def test_the_embedding_model_is_mandatory(app, db, remote):
    body = body_a()
    del body["embedding"]
    assert create(app, body).status_code == 400


def test_probe_shows_facts_and_reuses_the_stored_key_only_for_the_same_url(
    app, db, remote, monkeypatch
):
    k = create(app, body_a()).json
    seen = []
    real = sources.probe_source
    monkeypatch.setattr(
        sources,
        "probe_source",
        lambda cfg, client=None: seen.append(cfg.api_key) or real(cfg, client),
    )
    admin = client_for(app, ADMIN)
    r = admin.post(
        "/api/knowledge/external/test",
        json={"knowledge_id": k["id"], "source": {"url": URL, "collection": "fremd_a"}},
    )
    facts = r.json
    assert (
        r.status_code == 200
        and facts["points"] == 2
        and facts["dense_vectors"] == {"": 7}
    )
    assert "fremd_a" in facts["collections"] and facts["sample_payload"]["datei"]
    admin.post(
        "/api/knowledge/external/test",
        json={
            "knowledge_id": k["id"],
            "source": {"url": "http://anderer:6333", "collection": "x"},
        },
    )
    assert seen == ["geheim", ""]


# --- Suche ----------------------------------------------------------------------


def scope_for(*knowledge_dicts, profile=None):
    return RetrievalScope(
        collections=tuple(k["collection_name"] for k in knowledge_dicts),
        profiles=tuple(
            (k["collection_name"], profile or cfg(search={"mode": "hybrid"}))
            for k in knowledge_dicts
        ),
        externals=tuple(
            (k["collection_name"], k["source_config"]) for k in knowledge_dicts
        ),
        specs=tuple((k["collection_name"], json.dumps(SPEC)) for k in knowledge_dicts),
    )


@pytest.fixture
def retriever(db, remote):
    return MultiCollectionRetriever(
        client=QdrantClient(":memory:"),
        dense_embeddings=Embedder(),
        sparse_embeddings=FakeSparse(),
    )


def test_search_maps_foreign_field_names_and_falls_back_to_dense(
    app, db, remote, retriever
):
    k = knowledge.get_knowledge(create(app, body_a()).json["id"])
    with_scope(scope_for(k))
    docs = retriever.invoke("Kaution")
    assert docs[0].metadata["filename"] == "mietvertrag.pdf"
    assert docs[0].page_content.startswith("Die Kaution")
    assert docs[0].metadata["page_start"] == 3  # Seitenfeld der Quelle
    assert remote.writes == []


def test_hybrid_search_uses_the_foreign_sparse_vector(app, db, remote, retriever):
    k = knowledge.get_knowledge(create(app, body_b()).json["id"])
    with_scope(scope_for(k))
    docs = retriever.invoke("Kuendigung Frist")
    assert docs[0].metadata["filename"] == "frist.md"


def test_two_external_sources_and_a_down_source_degrade_gracefully(
    app, db, remote, retriever, monkeypatch
):
    a = knowledge.get_knowledge(create(app, body_a()).json["id"])
    b = knowledge.get_knowledge(create(app, body_b()).json["id"])
    scope = scope_for(a, b)
    with_scope(scope)
    names = {
        d.metadata["filename"] for d in retriever.invoke("Kaution Kuendigung Frist")
    }
    assert {"mietvertrag.pdf", "frist.md"} <= names

    def down(cfg):
        if cfg.collection == "fremd_a":
            raise ConnectionError("weg")
        return remote

    monkeypatch.setattr(sources, "connect", down)
    scope2 = scope_for(a, b)
    with_scope(scope2)
    docs = retriever.invoke("Kaution Kuendigung Frist")
    assert docs and all(d.metadata["filename"] != "mietvertrag.pdf" for d in docs)
    assert scope2.warnings and "nicht erreichbar" in scope2.warnings[0]


# --- Schreibschutz und Rechte -------------------------------------------------------


def test_external_knowledge_is_read_only(app, db, remote):
    k = create(app, body_a()).json
    admin = client_for(app, ADMIN)
    q = f"?knowledge_id={k['id']}"
    assert admin.post(f"/api/documents/ingest{q}").status_code == 409
    assert admin.post(f"/api/documents/upload{q}", data={}).status_code == 409
    assert admin.post(f"/api/knowledge/{k['id']}/reindex").status_code == 409
    assert admin.get(f"/api/vectordb{q}").status_code == 409
    assert (
        admin.post(
            f"/api/vectordb/clear{q}", json={"confirmation": "DELETE_ALL"}
        ).status_code
        == 409
    )
    assert (
        admin.get(f"/api/documents{q}").status_code == 200
    )  # Listen bleiben lesbar (leer)


def test_deleting_the_knowledge_never_touches_the_foreign_data(app, db, remote):
    k = create(app, body_a()).json
    assert client_for(app, ADMIN).delete(f"/api/knowledge/{k['id']}").status_code == 200
    assert remote.writes == [] and remote._c.collection_exists("fremd_a")
    assert [x["kind"] for x in knowledge.list_knowledge()] == ["local"]


def test_visibility_follows_the_usual_knowledge_grants(app, db, remote):
    k = create(app, body_a()).json
    ed = add_user(db, "ed", "editor")
    assert k["id"] not in [
        x["id"] for x in client_for(app, ed).get("/api/knowledge").json
    ]
    access.add_grant("knowledge", k["id"], "user", ed["id"], "read")
    got = client_for(app, ed).get("/api/knowledge").json
    assert k["id"] in [x["id"] for x in got]
    assert all("geheim" not in json.dumps(x) for x in got)


def test_changing_the_source_revalidates_and_keeps_the_key(app, db, remote):
    k = create(app, body_a()).json
    admin = client_for(app, ADMIN)
    bad = admin.patch(
        f"/api/knowledge/{k['id']}",
        json={
            "source": {**body_a()["source"], "api_key": "", "collection": "weg"},
            "embedding": SPEC,
        },
    )
    assert bad.status_code == 400
    ok = admin.patch(
        f"/api/knowledge/{k['id']}",
        json={
            "name": "Neu",
            "source": {**body_a()["source"], "api_key": ""},
            "embedding": SPEC,
        },
    )
    assert ok.status_code == 200 and ok.json["name"] == "Neu"
    assert (
        json.loads(knowledge.get_knowledge(k["id"])["source_config"])["api_key"]
        == "geheim"
    )
    assert knowledge.is_external(knowledge.get_knowledge(k["id"]))


def test_never_stale_and_no_local_collection(app, db, remote):
    from app.rag import model_settings

    k = knowledge.get_knowledge(create(app, body_a()).json["id"])
    model_settings.save_specs(mr.ModelSpec("hf", "ganz/anderes"))
    assert model_settings.is_stale(k) is False


def test_same_dimension_but_wrong_model_is_caught_by_the_sample_probe(
    app, db, remote, monkeypatch
):
    class Other(Embedder):
        def embed_query(self, text):  # gleiche Dimension, aber ganz andere Vektoren
            return [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]

    monkeypatch.setattr(mr, "get_embedder", lambda spec: Other())
    r = create(app, body_a())
    assert r.status_code == 400 and "passt nicht" in r.json["error"]
    monkeypatch.setattr(mr, "get_embedder", lambda spec: Embedder())
    assert create(app, body_a()).status_code == 201


# --- Zugriffsfilter (Benutzer und Gruppen) ---------------------------------------------------


@pytest.fixture
def secured(remote):
    """Fremde Collection mit Zugriffsfeldern: Gruppen, Besitzer, Mandant."""
    rows = [
        (
            "Die Kaution betraegt drei Monatsmieten",
            {
                "allowed_groups": ["Vertrieb"],
                "owner": "alice",
                "mandant": "A",
                "chunk_index": 0,
                "file_group_id": "d1",
            },
        ),
        (
            "Die Kuendigung erfolgt mit Frist",
            {
                "allowed_groups": ["Recht", "Vertrieb"],
                "owner": "bob",
                "mandant": "A",
                "chunk_index": 1,
                "file_group_id": "d1",
            },
        ),
        (
            "Das Dach wurde saniert",
            {
                "allowed_groups": ["Technik"],
                "owner": "alice",
                "mandant": "B",
                "chunk_index": 0,
                "file_group_id": "d2",
            },
        ),
        (
            "Strom wird abgerechnet",
            {
                "allowed_groups": ["Technik"],
                "owner": "carol",
                "mandant": "A",
                "chunk_index": 0,
                "file_group_id": "d3",
            },
        ),
    ]
    fill(
        remote._c,
        "gesichert",
        [
            (t, {"page_content": t, "filename": f"datei{i}.md", **p})
            for i, (t, p) in enumerate(rows)
        ],
    )
    return remote


def body_secured(**access):
    base = {"enabled": True, "group_field": "allowed_groups", "groups_from": "local"}
    return {
        "name": "Gesichert",
        "kind": "external",
        "embedding": SPEC,
        "source": {"url": URL, "collection": "gesichert", "access": {**base, **access}},
    }


def ask(retriever, k, query, **identity):
    scope = scope_for(k)
    scope.identity.update(identity)
    with_scope(scope)
    return {d.metadata["filename"] for d in retriever.invoke(query)}, scope


ALL = "Kaution Kuendigung Frist Dach Strom"


def test_group_filter_shows_only_matching_chunks(app, db, secured, retriever):
    k = knowledge.get_knowledge(create(app, body_secured()).json["id"])
    sales, _ = ask(retriever, k, ALL, username="x", groups=["Vertrieb"])
    assert sales == {"datei0.md", "datei1.md"}
    tech, _ = ask(
        retriever, k, ALL, username="x", groups=["technik"]
    )  # Schreibweise egal
    assert tech == {"datei2.md", "datei3.md"}
    both, _ = ask(retriever, k, ALL, username="x", groups=["Recht", "Technik"])
    assert both == {"datei1.md", "datei2.md", "datei3.md"}


def test_user_without_a_matching_identity_gets_nothing_not_everything(
    app, db, secured, retriever
):
    k = knowledge.get_knowledge(create(app, body_secured()).json["id"])
    none, _ = ask(retriever, k, ALL, username="x", groups=[])
    assert none == set()
    other, _ = ask(retriever, k, ALL, username="x", groups=["Marketing"])
    assert other == set()


def test_user_field_group_field_and_static_conditions_combine(
    app, db, secured, retriever
):
    only_owner = knowledge.get_knowledge(
        create(app, body_secured(group_field="", user_field="owner")).json["id"]
    )
    mine, _ = ask(retriever, only_owner, ALL, username="alice", groups=[])
    assert mine == {"datei0.md", "datei2.md"}
    combined = knowledge.get_knowledge(
        create(
            app,
            {
                **body_secured(
                    user_field="owner", static=[{"field": "mandant", "values": ["A"]}]
                ),
                "name": "Kombi",
            },
        ).json["id"]
    )
    # alice (Besitzerin) ODER Gruppe Technik, aber nur Mandant A: datei0 (alice, A) + datei3 (Technik, A), nicht datei2 (B)
    got, _ = ask(retriever, combined, ALL, username="alice", groups=["Technik"])
    assert got == {"datei0.md", "datei3.md"}


def test_filter_also_applies_to_neighbor_chunks_and_hybrid_search(
    app, db, secured, retriever
):
    k = knowledge.get_knowledge(create(app, body_secured()).json["id"])
    # Nachbar-Chunks (Standardfelder chunk_index/file_group_id) duerfen nichts Fremdes mitbringen
    scope = scope_for(k, profile=cfg(neighbors=1, search={"mode": "dense"}))
    scope.identity.update({"username": "x", "groups": ["Recht"]})
    with_scope(scope)
    texts = " ".join(d.page_content for d in retriever.invoke("Kuendigung Frist"))
    assert (
        "Kuendigung" in texts and "Kaution" not in texts
    )  # Chunk 0 von d1 ist nur fuer Vertrieb


def test_admins_can_be_exempt_only_when_switched_on(app, db, secured, retriever):
    k = knowledge.get_knowledge(create(app, body_secured()).json["id"])
    strict, _ = ask(retriever, k, ALL, username="admin", groups=[], is_admin=True)
    assert strict == set()
    k2 = knowledge.get_knowledge(
        create(app, {**body_secured(admins_unfiltered=True), "name": "Offen"}).json[
            "id"
        ]
    )
    open_, _ = ask(retriever, k2, ALL, username="admin", groups=[], is_admin=True)
    assert len(open_) == 4


def test_external_groups_can_feed_the_filter(app, db, secured, retriever):
    k = knowledge.get_knowledge(
        create(app, body_secured(groups_from="external")).json["id"]
    )
    got, _ = ask(
        retriever, k, ALL, username="x", groups=["Vertrieb"], external_groups=["Recht"]
    )
    assert got == {"datei1.md"}  # nur die externe Gruppe zaehlt
    both = knowledge.get_knowledge(
        create(app, {**body_secured(groups_from="both"), "name": "Beide"}).json["id"]
    )
    got2, _ = ask(
        retriever,
        both,
        ALL,
        username="x",
        groups=["Technik"],
        external_groups=["Recht"],
    )
    assert got2 == {"datei1.md", "datei2.md", "datei3.md"}


def test_access_configuration_is_validated(app, db, secured):
    r = create(app, body_secured(group_field="", user_field="", static=[]))
    assert r.status_code == 400 and "Zugriffsfilter" in r.json["error"]
    r = create(app, body_secured(groups_from="alle"))
    assert r.status_code == 400
    r = create(
        app, body_secured(group_field="", static=[{"field": "mandant", "values": []}])
    )
    assert r.status_code == 400 and "Werte" in r.json["error"]


def test_the_filter_test_shows_counts_without_saving_and_only_for_admins(
    app, db, secured
):
    admin = client_for(app, ADMIN)
    src = body_secured()["source"]
    r = admin.post(
        "/api/knowledge/external/filter-test",
        json={"source": src, "identity": {"username": "x", "groups": ["Vertrieb"]}},
    )
    assert (
        r.status_code == 200
        and r.json["count"] == 2
        and r.json["blocked"] is False
        and len(r.json["samples"]) == 2
    )
    blocked = admin.post(
        "/api/knowledge/external/filter-test",
        json={"source": src, "identity": {"username": "x"}},
    ).json
    assert blocked["blocked"] is True and blocked["count"] == 0
    assert [k["kind"] for k in knowledge.list_knowledge()] == [
        "local"
    ]  # nichts gespeichert
    ed = add_user(db, "ed2", "editor")
    assert (
        client_for(app, ed)
        .post("/api/knowledge/external/filter-test", json={"source": src})
        .status_code
        == 403
    )


def test_chat_scope_carries_the_users_identity(app, db, remote):
    from app.auth.access import user_group_names

    ed = add_user(db, "ed3", "editor")
    s = db.get_session()
    from app.core.models import Group, GroupMember

    g = Group(name="Vertrieb")
    s.add(g)
    s.flush()
    s.add(GroupMember(group_id=g.id, user_id=ed["id"]))
    s.commit()
    s.close()
    assert user_group_names(ed["id"]) == ["Vertrieb"]
