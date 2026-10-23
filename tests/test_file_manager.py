from pathlib import Path

import pytest

from app.knowledge.files import delete_associated_file


@pytest.mark.parametrize("ext", [".pdf", ".docx", ".doc", ".odt", ".md"])
def test_delete_archived_file_of_knowledge(tmp_path, monkeypatch, ext):
    monkeypatch.chdir(tmp_path)
    archived = Path(f"data/knowledge/2/processed_files/vertrag{ext}")
    archived.parent.mkdir(parents=True)
    archived.write_text("x")

    payload = {
        "filename": f"vertrag{ext}",
        "source": f"data/knowledge/2/files/vertrag{ext}",
    }
    assert delete_associated_file(payload) is True
    assert not archived.exists()


def test_delete_archived_file_of_default_knowledge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    archived = Path("data/processed_files/bericht.pdf")
    archived.parent.mkdir(parents=True)
    archived.write_text("x")

    payload = {"filename": "bericht.pdf", "source": "data/files/bericht.pdf"}
    assert delete_associated_file(payload) is True
    assert not archived.exists()
