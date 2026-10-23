# Open Knowledge UI - A privacy-focused, self-hosted RAG system
# SPDX-FileCopyrightText: 2026 S2A Open Technologies gGmbH
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 S2A Open Technologies gGmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from flask import Flask

from app.core.settings_service import SettingsService
from app.stt import stt_config
from app.web.admin_routes import admin_bp
from app.web.api_routes import api_bp
from app.web.routes import bp as main_bp
from tests.test_access import add_user, client_for

ADMIN = {"id": 1, "role": "admin", "username": "admin"}


class FakeWhisper(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        FakeWhisper.seen.append(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "size": len(body),
                "body": body,
            }
        )
        if b"fehler" in body:
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        raw = json.dumps({"text": " Hallo Welt "}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def whisper():
    FakeWhisper.seen = []
    srv = HTTPServer(("127.0.0.1", 0), FakeWhisper)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


@pytest.fixture
def app(db):
    stt_config.clear_probe_cache()
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    for bp in (main_bp, api_bp, admin_bp):
        app.register_blueprint(bp)
    return app


def set_stt(db, **values):
    s = db.get_session()
    SettingsService(s).set_batch(values)
    s.close()
    stt_config.clear_probe_cache()


def audio(content=b"RIFFxxxxWAVEdata"):
    return {"audio": (io.BytesIO(content), "recording.webm")}


def test_stream_url_is_built_from_the_base_url():
    assert (
        stt_config.stream_url("http://parakeet:5001/v1")
        == "ws://parakeet:5001/v1/audio/stream"
    )
    assert (
        stt_config.stream_url("https://stt.firma.de/v1/")
        == "wss://stt.firma.de/v1/audio/stream"
    )


def test_auto_mode_uses_upload_when_the_endpoint_has_no_stream(db, monkeypatch):
    set_stt(db, STT_BASEURL="http://stt:9/v1", STT_MODE="auto")
    monkeypatch.setattr(stt_config, "probe_stream", lambda base, key="": False)
    assert stt_config.resolve_mode(stt_config.load()) == "upload"
    monkeypatch.setattr(stt_config, "probe_stream", lambda base, key="": True)
    assert stt_config.resolve_mode(stt_config.load()) == "upload"  # gemerkt (5 Minuten)
    stt_config.clear_probe_cache()
    assert stt_config.resolve_mode(stt_config.load()) == "stream"


def test_an_explicit_mode_wins_over_the_probe(db, monkeypatch):
    monkeypatch.setattr(
        stt_config, "probe_stream", lambda *a, **k: pytest.fail("keine Probe nötig")
    )
    set_stt(db, STT_BASEURL="http://stt:9/v1", STT_MODE="stream")
    assert stt_config.resolve_mode(stt_config.load()) == "stream"
    set_stt(db, STT_MODE="upload")
    assert stt_config.resolve_mode(stt_config.load()) == "upload"


def test_config_endpoint_tells_the_browser_what_to_do_without_secrets(
    app, db, monkeypatch
):
    c = client_for(app, add_user(db, "ed", "editor"))
    set_stt(db, STT_BASEURL="", STT_MODE="auto")
    assert c.get("/api/stt/config").json == {"available": False, "mode": "upload"}
    set_stt(db, STT_BASEURL="http://stt:9/v1", STT_MODE="upload", STT_API_KEY="geheim")
    got = c.get("/api/stt/config")
    assert got.json == {
        "available": True,
        "mode": "upload",
    } and "geheim" not in got.get_data(as_text=True)
    assert app.test_client().get("/api/stt/config").status_code == 401


def test_upload_sends_the_audio_to_an_openai_compatible_endpoint(app, db, whisper):
    set_stt(db, STT_BASEURL=whisper, STT_MODEL="whisper-large-v3", STT_API_KEY="k1")
    r = client_for(app, add_user(db, "ed", "editor")).post(
        "/api/stt", data=audio(), content_type="multipart/form-data"
    )
    assert r.status_code == 200 and r.json == {"text": "Hallo Welt"}
    call = FakeWhisper.seen[0]
    assert (
        call["path"].endswith("/audio/transcriptions") and call["auth"] == "Bearer k1"
    )
    assert b"whisper-large-v3" in call["body"]


def test_upload_rejects_missing_empty_and_oversized_audio(
    app, db, whisper, monkeypatch
):
    set_stt(db, STT_BASEURL=whisper)
    c = client_for(app, add_user(db, "ed", "editor"))
    assert (
        c.post("/api/stt", data={}, content_type="multipart/form-data").status_code
        == 400
    )
    assert (
        c.post(
            "/api/stt", data=audio(b""), content_type="multipart/form-data"
        ).status_code
        == 400
    )
    monkeypatch.setattr(stt_config, "MAX_AUDIO_BYTES", 10)
    assert (
        c.post(
            "/api/stt", data=audio(b"x" * 11), content_type="multipart/form-data"
        ).status_code
        == 413
    )
    assert FakeWhisper.seen == []  # nichts weitergereicht


def test_upload_errors_do_not_leak_internals(app, db, whisper):
    set_stt(db, STT_BASEURL=whisper)
    c = client_for(app, add_user(db, "ed", "editor"))
    r = c.post("/api/stt", data=audio(b"fehler"), content_type="multipart/form-data")
    assert (
        r.status_code == 502
        and "HTTP 500" in r.json["error"]
        and "Traceback" not in r.get_data(as_text=True)
    )
    set_stt(db, STT_BASEURL="http://127.0.0.1:9/v1")
    gone = c.post("/api/stt", data=audio(), content_type="multipart/form-data")
    assert gone.status_code == 502 and "nicht erreichbar" in gone.json["error"]


def test_upload_requires_a_login_and_refuses_link_local_targets(app, db):
    assert (
        app.test_client()
        .post("/api/stt", data=audio(), content_type="multipart/form-data")
        .status_code
        == 401
    )
    set_stt(db, STT_BASEURL="http://169.254.169.254/v1")
    r = client_for(app, add_user(db, "ed", "editor")).post(
        "/api/stt", data=audio(), content_type="multipart/form-data"
    )
    assert r.status_code == 502


def test_tools_api_saves_and_validates_the_mode(app, db):
    admin = client_for(app, ADMIN)
    assert admin.get("/api/admin/tools").json["stt"]["mode"] == "auto"
    ok = admin.put(
        "/api/admin/tools",
        json={"stt": {"base_url": "http://stt:9/v1", "mode": "upload"}},
    )
    assert ok.status_code == 200 and ok.json["stt"]["mode"] == "upload"
    assert (
        admin.put(
            "/api/admin/tools",
            json={"stt": {"base_url": "http://stt:9/v1", "mode": "nope"}},
        ).status_code
        == 400
    )
    ed = client_for(app, add_user(db, "ed2", "editor"))
    assert (
        ed.put("/api/admin/tools", json={"stt": {"mode": "stream"}}).status_code == 403
    )


def test_the_test_button_does_a_real_transcription_and_reports_both_modes(
    app, db, whisper, monkeypatch
):
    monkeypatch.setattr(stt_config, "probe_stream", lambda *a, **k: False)
    r = client_for(app, ADMIN).post(
        "/api/admin/tools/test",
        json={"kind": "stt", "values": {"base_url": whisper, "model": "m"}},
    )
    assert (
        r.status_code == 200
        and "Aufnehmen & senden: ok" in r.json["details"]
        and "nicht angeboten" in r.json["details"]
    )
    assert FakeWhisper.seen and FakeWhisper.seen[0]["path"].endswith(
        "/audio/transcriptions"
    )
    monkeypatch.setattr(stt_config, "probe_stream", lambda *a, **k: True)
    both = client_for(app, ADMIN).post(
        "/api/admin/tools/test", json={"kind": "stt", "values": {"base_url": whisper}}
    )
    assert "Live-Streaming" in both.json["message"]


def test_the_test_button_keeps_ports_ending_in_one_intact(app, db, monkeypatch):
    seen = []
    monkeypatch.setattr(
        stt_config, "transcribe", lambda cfg, *a, **k: seen.append(cfg.base_url) or ""
    )
    monkeypatch.setattr(
        stt_config, "probe_stream", lambda base, key="": seen.append(base) or False
    )
    client_for(app, ADMIN).post(
        "/api/admin/tools/test",
        json={"kind": "stt", "values": {"base_url": "http://stt.example:5001"}},
    )
    assert seen == ["http://stt.example:5001", "http://stt.example:5001"]


def test_silent_wav_is_a_valid_one_second_wav():
    import wave

    with wave.open(io.BytesIO(stt_config.silent_wav(1.0))) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (1, 16000, 16000)


def test_stream_proxy_sends_the_key_to_the_upstream(monkeypatch):
    import app.stt.stt_stream as stream

    captured = {}
    monkeypatch.setattr(
        stream.websocket,
        "create_connection",
        lambda url, timeout=10, header=None: captured.update(url=url, header=header),
    )
    stream._connect_upstream("ws://stt:9/v1/audio/stream", "geheim")
    assert captured["header"] == ["Authorization: Bearer geheim"]
    stream._connect_upstream("ws://stt:9/v1/audio/stream", "")
    assert captured["header"] == []


def test_probe_detects_a_real_websocket_stream_endpoint_and_a_plain_http_one(whisper):
    from flask_sock import Sock
    from werkzeug.serving import make_server

    flask_app = Flask("fake-parakeet")
    sock = Sock(flask_app)

    @sock.route("/v1/audio/stream")
    def stream(ws):
        ws.close()

    server = make_server("127.0.0.1", 0, flask_app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert (
            stt_config.probe_stream(f"http://127.0.0.1:{server.server_port}/v1") is True
        )
    finally:
        server.shutdown()
    assert (
        stt_config.probe_stream(whisper) is False
    )  # Whisper-Server ohne /audio/stream
    assert stt_config.probe_stream("http://127.0.0.1:9/v1") is False
