import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.llm import connections
from app.rag import models_runtime as mr


@pytest.fixture(autouse=True)
def clean():
    mr.reset_for_tests()
    yield
    mr.reset_for_tests()


@pytest.fixture
def server():
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body, self.headers.get("Authorization")))
            if self.path.endswith("/embeddings"):
                data = [
                    {"index": i, "embedding": [float(len(t)), 1.0, 0.0]}
                    for i, t in reversed(list(enumerate(body["input"])))
                ]
                out = {"data": data}
            elif self.path.endswith("/rerank"):
                docs = body["documents"]
                out = {
                    "results": [
                        {"index": i, "relevance_score": float(len(d))}
                        for i, d in reversed(list(enumerate(docs)))
                    ]
                }
            else:
                self.send_response(404)
                self.end_headers()
                return
            raw = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", calls
    srv.shutdown()


def test_spec_validation():
    with pytest.raises(ValueError):
        mr.ModelSpec("endpoint", "m")
    with pytest.raises(ValueError):
        mr.ModelSpec("nope", "m")
    s = mr.ModelSpec.from_dict('{"source":"endpoint","model":"m","connection_id":3}')
    assert s.connection_id == 3 and mr.ModelSpec.from_dict(s.to_dict()) == s


def test_endpoint_embedder_order_and_dimension(db, server):
    url, calls = server
    conn = connections.create_connection("emb", url, "k")
    spec = mr.ModelSpec("endpoint", "bge-m3", conn["id"])
    emb = mr.get_embedder(spec)
    assert emb.dimension == 3
    vecs = emb.embed_documents(["a", "bbb", "cc"])
    assert [v[0] for v in vecs] == [1.0, 3.0, 2.0]  # Reihenfolge nach index
    assert calls[-1][2] == "Bearer k"
    assert mr.get_embedder(spec) is emb
    assert mr.status("embedding", spec)["state"] == "ready"
    assert mr.status("embedding", spec)["dimension"] == 3


def test_endpoint_reranker_maps_scores_back(db, server):
    url, _ = server
    conn = connections.create_connection("rr", url, "")
    rr = mr.get_reranker(mr.ModelSpec("endpoint", "r", conn["id"]))
    assert rr.score("q", ["aa", "b", "cccc"]) == [2.0, 1.0, 4.0]


def test_endpoint_unreachable_sets_error_and_caches(db):
    conn = connections.create_connection("x", "http://127.0.0.1:9/v1", "")
    spec = mr.ModelSpec("endpoint", "m", conn["id"])
    with pytest.raises(mr.ModelLoadError):
        mr.get_embedder(spec)
    assert mr.status("embedding", spec)["state"] == "error"
    with pytest.raises(mr.ModelLoadError):  # zweiter Versuch sofort aus dem Cache
        mr.get_embedder(spec)


def test_link_local_blocked(db):
    conn = connections.create_connection("meta", "http://169.254.169.254/v1", "")
    with pytest.raises(mr.ModelLoadError):
        mr.get_embedder(mr.ModelSpec("endpoint", "m", conn["id"]))


def test_hf_models_with_fake_sentence_transformers(monkeypatch):
    class FakeST:
        loads = 0

        def __init__(self, name, **kw):
            FakeST.loads += 1

        def get_sentence_embedding_dimension(self):
            return 5

        def encode_query(self, texts, **kw):
            return [[1.0] * 5 for _ in texts]

        def encode_document(self, texts, **kw):
            return [[2.0] * 5 for _ in texts]

    class FakeCE:
        def __init__(self, name, **kw):
            pass

        def predict(self, pairs, **kw):
            return [len(p[1]) for p in pairs]

    mod = types.SimpleNamespace(SentenceTransformer=FakeST, CrossEncoder=FakeCE)
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    spec = mr.ModelSpec("hf", "x/y")
    emb = mr.get_embedder(spec)
    mr.get_embedder(spec)
    assert FakeST.loads == 1 and emb.dimension == 5
    assert emb.embed_query("a") == [1.0] * 5 and emb.embed_documents(["a"]) == [
        [2.0] * 5
    ]
    assert mr.get_reranker(mr.ModelSpec("hf", "r")).score("q", ["ab", "c"]) == [
        2.0,
        1.0,
    ]


def test_hf_load_failure_message(monkeypatch):
    class Boom:
        def __init__(self, name, **kw):
            raise OSError("kein Netz")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=Boom),
    )
    with pytest.raises(mr.ModelLoadError, match="kein Netz"):
        mr.get_embedder(mr.ModelSpec("hf", "x/y"))


def test_signature_and_uploads_collection():
    a = mr.ModelSpec("hf", "a")
    b = mr.ModelSpec("hf", "b")
    assert mr.index_signature(a, 1000, 150, 1, 1) == mr.index_signature(
        a.to_dict(), 1000, 150, 1, 1
    )
    assert mr.index_signature(a, 1000, 150, 1, 1) != mr.index_signature(
        b, 1000, 150, 1, 1
    )
    assert mr.index_signature(a, 1000, 150, 1, 1) != mr.index_signature(
        a, 800, 150, 1, 1
    )
    assert mr.uploads_collection(a) != mr.uploads_collection(b)


def test_redirects_are_not_followed(db):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    hit = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(307)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()

        def do_GET(self):
            hit.append(self.path)
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/")
            self.end_headers()

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        conn = connections.create_connection(
            "r", f"http://127.0.0.1:{srv.server_port}/v1", ""
        )
        with pytest.raises(mr.ModelLoadError, match="Weiterleitung"):
            mr.get_embedder(mr.ModelSpec("endpoint", "m", conn["id"]))

        from app.llm import registry

        with pytest.raises(registry.RemoteModelsError, match="Weiterleitung"):
            registry.fetch_remote_models({**conn, "api_key": ""}, force=True)
    finally:
        srv.shutdown()
