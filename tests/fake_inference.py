"""Fake-Server fuer Text Embeddings Inference (TEI) und Infinity, die ihr echtes Format sprechen."""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

DIM = 4


def vector(text: str) -> list[float]:
    words = set(re.findall(r"\w+", text.lower()))
    return [
        float(len(words)),
        float(sum(len(w) for w in words)),
        float("miete" in words),
        float("wetter" in words),
    ]


def relevance(query: str, text: str) -> float:
    q = set(re.findall(r"\w+", query.lower()))
    return float(len(q & set(re.findall(r"\w+", text.lower()))))


class _Base(BaseHTTPRequestHandler):
    api_key: str | None = None
    prefix = ""  # Infinity/TEI laufen ohne /v1; mit prefix="/v1" nur dort
    seen: list = []

    def log_message(self, *a):
        pass

    def _send(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self) -> bool:
        if (
            self.api_key
            and self.headers.get("Authorization") != f"Bearer {self.api_key}"
        ):
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def _path(self):
        path = self.path.split("?")[0]
        if self.prefix:
            return path[len(self.prefix) :] if path.startswith(self.prefix) else None
        return path

    def _body(self):
        return json.loads(
            self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
        )

    def do_GET(self):
        if self._authorized():
            self.get(self._path())

    def do_POST(self):
        if self._authorized():
            path, body = self._path(), self._body()
            type(self).seen.append((path, body))
            self.post(path, body)

    def get(self, path):
        self._send(404, {"error": "not found"})

    def post(self, path, body):
        self._send(404, {"error": "not found"})


class TEI(_Base):
    """Ein Modell pro Server. mode = "embedding" | "reranker". Kennt kein /v1/models."""

    mode = "embedding"
    model_id = "BAAI/bge-m3"

    def get(self, path):
        if path == "/info":
            kind = (
                {"embedding": {"pooling": "cls"}}
                if self.mode == "embedding"
                else {"reranker": {}}
            )
            return self._send(200, {"model_id": self.model_id, "model_type": kind})
        if path == "/health":
            return self._send(200, {})
        self._send(404, {"error": "not found"})

    def post(self, path, body):
        if path == "/embed" and self.mode == "embedding":
            return self._send(200, [vector(t) for t in body["inputs"]])
        if path == "/rerank" and self.mode == "reranker":
            if "texts" not in body:  # TEI kennt "documents" nicht
                return self._send(422, {"error": "missing field `texts`"})
            rows = [
                {"index": i, "score": relevance(body["query"], t)}
                for i, t in enumerate(body["texts"])
            ]
            return self._send(
                200, sorted(rows, key=lambda r: -r["score"])
            )  # TEI sortiert nach Score
        self._send(
            404 if path not in ("/embed", "/rerank") else 424,
            {"error": "wrong model type"},
        )


class Infinity(_Base):
    """Mehrere Modelle; /models mit Faehigkeiten, /embeddings und /rerank."""

    def get(self, path):
        if path == "/models":
            return self._send(
                200,
                {
                    "data": [
                        {
                            "id": "emb-model",
                            "object": "model",
                            "capabilities": ["embed"],
                        },
                        {
                            "id": "rr-model",
                            "object": "model",
                            "capabilities": ["rerank"],
                        },
                    ]
                },
            )
        if path == "/health":
            return self._send(200, {})
        self._send(404, {"error": "not found"})

    def post(self, path, body):
        if path == "/embeddings" and body.get("model") == "emb-model":
            data = [
                {"object": "embedding", "index": i, "embedding": vector(t)}
                for i, t in enumerate(body["input"])
            ]
            return self._send(
                200, {"object": "list", "data": data, "model": body["model"]}
            )
        if path == "/rerank" and body.get("model") == "rr-model":
            rows = [
                {"index": i, "relevance_score": relevance(body["query"], t)}
                for i, t in enumerate(body["documents"])
            ]
            return self._send(
                200,
                {
                    "object": "rerank",
                    "results": sorted(rows, key=lambda r: -r["relevance_score"]),
                },
            )
        self._send(404, {"error": "model not found"})


def serve(handler_cls, **attrs):
    """Startet einen Server; liefert (base_url, stop). ``attrs`` werden an einer Unterklasse gesetzt."""
    cls = type(handler_cls.__name__, (handler_cls,), {"seen": [], **attrs})
    srv = HTTPServer(("127.0.0.1", 0), cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}", srv.shutdown, cls
