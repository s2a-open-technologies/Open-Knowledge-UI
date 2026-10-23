# ruff: noqa: F811  (Fixtures aus anderen Testmodulen werden importiert und als Argument genutzt)
"""Bilder im Chat: als Text lesen (app/ingest/image_text.py) und direkt an Modelle mit Bildverstaendnis."""

import base64
import importlib.util
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app.chat import attachments
from app.ingest import image_text, prompts, tasks
from app.knowledge import service as knowledge  # noqa: F401
from app.llm import connections as store
from tests.test_access import add_user, app, client_for, conn  # noqa: F401
from tests.test_ingest import FakeDense, FakeSparse, no_enrichment, qd  # noqa: F401
from tests.test_web_and_vision import ADMIN, put_tasks


def png_bytes(size=(60, 30), color="white", mode="RGB"):
    buf = BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


class FakeModel:
    def __init__(self, answers=("Rechnung 4.812,30 EUR",), error=None):
        self.answers, self.error, self.calls, self.instruction = (
            list(answers),
            error,
            0,
            None,
        )

    def generate_image_message(self, b64, instruction=None):
        self.instruction = instruction
        return b64

    def generate(self, message):
        self.calls += 1
        if self.error:
            raise self.error
        return self.answers[min(self.calls, len(self.answers)) - 1]


# --- Bild lesen ---------------------------------------------------------------------------------


def test_encode_image_shrinks_flattens_transparency_and_applies_exif(tmp_path):
    big = tmp_path / "gross.png"
    Image.new("RGBA", (4000, 1000), (255, 0, 0, 0)).save(big)  # voll transparent
    out = Image.open(BytesIO(base64.b64decode(image_text.encode_image(big))))
    assert max(out.size) == image_text.MAX_SIDE and out.mode == "RGB"
    assert out.getpixel((10, 10)) == (255, 255, 255)  # Transparenz wird Weiss

    photo = tmp_path / "foto.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # um 90 Grad im Uhrzeigersinn drehen
    Image.new("RGB", (200, 100), "blue").save(photo, exif=exif)
    assert Image.open(
        BytesIO(base64.b64decode(image_text.encode_image(photo)))
    ).size == (100, 200)


def test_a_broken_image_is_reported_in_german(tmp_path):
    bad = tmp_path / "kaputt.png"
    bad.write_bytes(b"kein bild")
    with pytest.raises(image_text.ImageReadError, match="nicht öffnen"):
        image_text.encode_image(bad)


def test_image_to_markdown_uses_the_task_prompt_and_retries(db, tmp_path):
    img = tmp_path / "r.png"
    img.write_bytes(png_bytes())
    model = FakeModel(answers=["", "Rechnung 4.812,30 EUR"])
    md = image_text.image_to_markdown(img, "r.png", model=model, delay=0)
    assert md.startswith("# r.png\n\n") and "4.812,30" in md and model.calls == 2
    assert model.instruction == prompts.IMAGES_PROMPT


def test_image_to_markdown_gives_up_with_a_clear_message(db, tmp_path):
    img = tmp_path / "r.png"
    img.write_bytes(png_bytes())
    model = FakeModel(error=RuntimeError("Endpunkt nicht erreichbar"))
    with pytest.raises(
        image_text.ImageReadError, match="Bildmodell konnte das Bild nicht lesen"
    ):
        image_text.image_to_markdown(img, "r.png", model=model, retries=2, delay=0)
    assert model.calls == 2


def test_a_disabled_task_never_calls_the_model(db, app, tmp_path):
    put_tasks(client_for(app, ADMIN), images={"enabled": False})
    img = tmp_path / "r.png"
    img.write_bytes(png_bytes())
    model = FakeModel()
    with pytest.raises(image_text.ImageReadError, match="abgeschaltet"):
        image_text.image_to_markdown(img, "r.png", model=model)
    assert model.calls == 0


def test_the_images_task_falls_back_to_the_vision_model(db, app, conn, monkeypatch):
    from app.llm import registry
    from app.llm.local_llm import TaskLLM

    monkeypatch.setattr(
        registry, "fetch_remote_models", lambda c, force=False: ["m1", "m2"]
    )
    admin = client_for(app, ADMIN)
    ref2 = store.make_base_ref(conn["id"], "m2")
    put_tasks(admin, vision={"model_ref": ref2})
    llm = TaskLLM("images")
    llm.llm
    assert llm.vision_model == "m2"  # leer = Modell der Bildbeschreibung


# --- Anhang-Ingestion eines Bildes -----------------------------------------------------------------


def test_an_image_attachment_becomes_searchable_text(
    db, qd, no_enrichment, monkeypatch, tmp_path
):
    from tests.test_ingest import make_ingestion, scroll_all

    monkeypatch.setattr(
        "app.llm.local_llm.TaskLLM",
        lambda task="vision": FakeModel(
            ["Rechnung Nr. 2026-0147\n\nBetrag: 4.812,30 EUR"]
        ),
    )
    from app.rag.collections import ensure_collection

    ensure_collection(qd, "user_uploads", 7)
    folder = Path("data/attachments/s-1/a-1")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "scan.png"
    path.write_bytes(png_bytes((200, 100)))
    with make_ingestion(
        qd,
        collection_name="user_uploads",
        track_documents=False,
        enrich_context=False,
        extra_metadata={"session_id": "s-1"},
        files_dir=str(folder),
    ) as ing:
        assert ing._process_file(str(path), ing._shared_sparse) is True
    text = " ".join(c.payload["page_content"] for c in scroll_all(qd, "user_uploads"))
    assert "4.812,30" in text  # Text aus dem Bild ist durchsuchbar


def test_a_model_failure_reaches_the_user_as_the_attachment_error(
    db, qd, no_enrichment, monkeypatch
):
    from tests.test_ingest import make_ingestion

    monkeypatch.setattr(image_text, "RETRIES", 1)
    monkeypatch.setattr(
        "app.llm.local_llm.TaskLLM",
        lambda task="vision": FakeModel(error=RuntimeError("down")),
    )
    monkeypatch.setattr(image_text.time, "sleep", lambda s: None)
    folder = Path("data/attachments/s-2/a-2")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "scan.png"
    path.write_bytes(png_bytes())
    with make_ingestion(
        qd,
        collection_name="user_uploads",
        track_documents=False,
        enrich_context=False,
        files_dir=str(folder),
    ) as ing:
        assert ing._process_file(str(path), ing._shared_sparse) is False
        assert "Bildmodell konnte das Bild nicht lesen" in ing.errors[0]["error"]


# --- Upload-Route ----------------------------------------------------------------------------------


@pytest.fixture
def upload_env(app, db, monkeypatch):
    import importlib

    app_pkg = importlib.import_module("app")

    started = []
    monkeypatch.setattr(app_pkg, "resources_ready", lambda: True)
    monkeypatch.setattr(attachments, "_ingest", lambda *a, **k: started.append(a))
    app.extensions["sparse_embeddings"] = object()
    return started


def upload(client, name, data):
    return client.post(
        "/api/chat/attachments",
        data={"file": (BytesIO(data), name)},
        content_type="multipart/form-data",
    )


def test_images_are_accepted_and_keep_a_shrunk_copy(app, db, upload_env):
    alice = add_user(db, "alice", "user")
    res = upload(client_for(app, alice), "foto.png", png_bytes((3000, 500)))
    assert res.status_code == 202
    body = res.json
    copy = attachments.image_copy_path(body["session_id"], body["id"])
    assert (
        Image.open(copy).size[0] == 2048
    )  # verkleinert, schon vor dem Lesen vorhanden
    assert attachments.image_data_url(body["session_id"], body["id"]).startswith(
        "data:image/jpeg;base64,"
    )
    assert upload_env  # Einlesen wurde gestartet


def test_images_are_refused_when_the_task_is_switched_off(app, db, upload_env):
    put_tasks(client_for(app, ADMIN), images={"enabled": False})
    alice = add_user(db, "alice", "user")
    res = upload(client_for(app, alice), "foto.png", png_bytes())
    assert res.status_code == 415 and "abgeschaltet" in res.json["error"]
    assert upload(client_for(app, alice), "x.exe", b"MZ").status_code == 415


# --- Bild direkt an das Chat-Modell -----------------------------------------------------------------


class CapturingAgent:
    sent: list = []

    async def astream_events(self, input_data, *a, **k):
        CapturingAgent.sent.append(input_data["messages"])
        if False:
            yield None


@pytest.fixture
def chat_images(app, db, conn, upload_env, monkeypatch):
    from app.api import chat as routes

    CapturingAgent.sent = []
    monkeypatch.setattr(routes, "get_agent", lambda resolved, tools: CapturingAgent())
    monkeypatch.setattr(routes, "resources_ready", lambda: True)
    app.extensions["agent_tools"] = []
    return store.make_base_ref(conn["id"], "m")


def attach_image(app, user):
    body = upload(client_for(app, user), "foto.png", png_bytes()).json
    return body["session_id"], body["id"]


def ask(app, user, ref, session_id, ids):
    client_for(app, user).post(
        "/chat",
        json={
            "message": "Was steht drauf?",
            "model": ref,
            "session_id": session_id,
            "attachment_ids": ids,
        },
    ).get_data()
    return CapturingAgent.sent[-1][-1]["content"]


def test_a_model_without_vision_only_gets_text(app, db, chat_images):
    alice = add_user(db, "alice", "user")
    sid, aid = attach_image(app, alice)
    assert isinstance(ask(app, alice, chat_images, sid, [aid]), str)


def test_a_model_with_vision_also_gets_the_image(app, db, chat_images):
    store.save_model_config(chat_images, {"vision": True})
    alice = add_user(db, "alice", "user")
    sid, aid = attach_image(app, alice)
    content = ask(app, alice, chat_images, sid, [aid])
    assert content[0]["type"] == "text" and content[0]["text"].startswith(
        "Was steht drauf?"
    )
    assert content[1]["type"] == "image_url" and content[1]["image_url"][
        "url"
    ].startswith("data:image/jpeg;base64,")
    assert isinstance(
        ask(app, alice, chat_images, sid, []), str
    )  # Folgefrage ohne neue IDs: nur Text


def test_foreign_or_unknown_attachment_ids_are_ignored(app, db, chat_images):
    store.save_model_config(chat_images, {"vision": True})
    alice, bob = add_user(db, "alice", "user"), add_user(db, "bob", "user")
    sid_a, _aid_a = attach_image(app, alice)
    _sid_b, aid_b = attach_image(app, bob)
    assert isinstance(
        ask(app, alice, chat_images, sid_a, [aid_b, "gibt-es-nicht"]), str
    )  # Bobs Bild, falsche Session


def test_the_pii_filter_masks_the_text_but_the_image_is_sent_unchanged(
    app, db, chat_images, monkeypatch
):
    from app.api import chat as routes

    store.save_model_config(chat_images, {"vision": True})

    class Filter:
        def demask(self, text, mapping):
            return text

    monkeypatch.setattr(routes, "pii_client", lambda: Filter())
    monkeypatch.setattr(
        routes,
        "mask_conversation",
        lambda client, texts: ([t.replace("Was", "[X]") for t in texts], []),
    )
    alice = add_user(db, "alice", "user")
    sid, aid = attach_image(app, alice)
    content = ask(app, alice, chat_images, sid, [aid])
    assert (
        content[0]["text"].startswith("[X] steht drauf?")
        and content[1]["type"] == "image_url"
    )


def test_vision_is_a_model_property_that_clones_inherit_and_the_model_list_shows(
    app, db, conn, monkeypatch
):
    from app.llm import registry

    monkeypatch.setattr(registry, "fetch_remote_models", lambda c, force=False: ["m"])
    ref = store.make_base_ref(conn["id"], "m")
    store.save_model_config(ref, {"vision": True})
    assert registry.resolve_model(ref).vision is True
    alice = add_user(db, "alice", "user")
    for user in (alice, ADMIN):
        m = next(
            x
            for x in client_for(app, user).get("/api/models?visible=1").json["models"]
            if x["ref"] == ref
        )
        assert m["vision"] is True
    clone = store.clone_model(ref, "Klon")
    assert store.get_model_config(clone["ref"])["vision"] is True


# --- Migrationen ---------------------------------------------------------------------------------


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_0032_matches_the_seed_and_0033_runs_up_down_up(tmp_path):
    rows = {
        k: v for k, v, _d in _load("alembic/versions/0032_chat_images.py", "m0032").ROWS
    }
    assert (
        rows["IMAGES_PROMPT"] == prompts.IMAGES_PROMPT
        and rows["INGEST_CHAT_IMAGES"] == "true"
    )
    assert set(tasks.TASKS) == {"vision", "context", "tables", "images"}

    env = {**os.environ, "DATABASE_URL": f"sqlite:///{tmp_path / 'm.db'}"}
    run = lambda *a: subprocess.run(
        [sys.executable, "-m", "alembic", *a], env=env, check=True, capture_output=True
    )
    run("upgrade", "head")
    run("downgrade", "0031")
    run("upgrade", "head")
