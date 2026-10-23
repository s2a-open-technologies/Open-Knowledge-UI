import json
from pathlib import Path

from app.knowledge import service as knowledge


def test_default_knowledge_is_seeded_once(db):
    db.init_default_data()  # idempotent
    items = knowledge.list_knowledge()
    assert [k["name"] for k in items] == ["Standard"]
    assert items[0]["is_default"] is True
    assert items[0]["collection_name"] == "open_knowledge_ui"
    assert knowledge.get_knowledge(None)["id"] == items[0]["id"]


def test_create_knowledge_gets_own_collection_and_embedding_config(db):
    k = knowledge.create_knowledge("Verträge", "Alle Verträge", 384)
    assert k["collection_name"] == f"kb_{k['id']}"
    assert k["is_default"] is False
    assert k["embedding_dimension"] == 384
    assert json.loads(k["embedding_spec"])["source"] == "hf"
    assert k["index_signature"]
    assert [x["name"] for x in knowledge.list_knowledge()] == ["Standard", "Verträge"]


def test_paths_default_vs_custom(db):
    default = knowledge.knowledge_paths(knowledge.get_knowledge(None))
    assert default.files_dir == Path("data/files")
    assert default.name_prefix == ""

    k = knowledge.create_knowledge("FAQ")
    paths = knowledge.knowledge_paths(k)
    assert paths.files_dir == Path(f"data/knowledge/{k['id']}/files")
    assert paths.processed_dir == Path(f"data/knowledge/{k['id']}/processed_files")
    assert paths.data_dir.name == f"knowledge_{k['id']}"
    assert paths.name_prefix == f"knowledge_{k['id']}/"


def test_default_knowledge_cannot_be_deleted(db):
    default = knowledge.get_knowledge(None)
    assert knowledge.delete_knowledge(default["id"]) is None
    assert knowledge.get_knowledge(None) is not None


def test_delete_removes_documents_and_folders(db):
    k = knowledge.create_knowledge("Weg damit")
    paths = knowledge.knowledge_paths(k)
    paths.files_dir.mkdir(parents=True)
    (paths.files_dir / "a.pdf").write_text("x")
    db.add_document("grp-1", "a.pdf", knowledge_id=k["id"])
    db.add_document("grp-2", "b.pdf", knowledge_id=knowledge.get_knowledge(None)["id"])

    deleted = knowledge.delete_knowledge(k["id"])

    assert deleted["collection_name"] == f"kb_{k['id']}"
    assert not paths.files_dir.parent.exists()
    assert knowledge.get_knowledge(k["id"]) is None
    assert [d.filename for d in db.get_all_documents()] == ["b.pdf"]


def test_documents_filtered_by_knowledge(db):
    default_id = knowledge.get_knowledge(None)["id"]
    other = knowledge.create_knowledge("Anderes")
    db.add_document("g1", "eins.pdf", knowledge_id=default_id)
    db.add_document("g2", "zwei.pdf", knowledge_id=other["id"])

    assert [d.filename for d in db.get_all_documents(knowledge_id=other["id"])] == [
        "zwei.pdf"
    ]
    assert len(db.get_all_documents()) == 2


def test_attachment_lifecycle(db):
    sess = db.create_session()
    att = db.create_attachment(sess.id, "bericht.pdf")
    assert db.get_attachment(att.id).status == "processing"

    db.set_attachment_status(att.id, "ready")
    assert db.get_attachment(att.id).status == "ready"

    db.set_attachment_status(att.id, "error", "kaputt")
    fetched = db.get_attachment(att.id)
    assert (fetched.status, fetched.error) == ("error", "kaputt")

    db.delete_session_attachments(sess.id)
    assert db.get_attachment(att.id) is None
