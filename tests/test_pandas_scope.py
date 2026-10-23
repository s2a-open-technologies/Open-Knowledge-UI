import os
import time

import pytest

from app.agent.tools.custom_tools import CustomTools
from app.rag import retrieval
from app.rag.retrieval import RetrievalScope


@pytest.fixture
def tools(tmp_path, worker):
    data = tmp_path / "data"
    (data / "knowledge_2").mkdir(parents=True)
    (data / "knowledge_3").mkdir()
    (data / "umsatz.csv").write_text("monat,betrag\njan,100\nfeb,250\n")
    (data / "knowledge_2" / "vertrag--ab12--tabelle-1.csv").write_text(
        "posten,betrag\nmiete,950\nkaution,2850\n"
    )
    (data / "knowledge_3" / "geheim.csv").write_text("name,gehalt\nanna,9999\n")
    (tmp_path / "ausserhalb.csv").write_text("x,y\n1,2\n")
    return CustomTools(llm=None, retriever=None, data_dir=str(data)), data


def use_scope(data, *dirs):
    pairs = tuple((prefix, str(data / sub)) for prefix, sub in dirs)
    return retrieval.set_scope(RetrievalScope(("kb",), None, data_dirs=pairs))


@pytest.fixture(autouse=True)
def reset_scope():
    token = retrieval.set_scope(None)
    yield
    retrieval._scope.reset(token)


def test_without_a_scope_the_flat_data_dir_is_used(tools):
    t, _ = tools
    assert t.list_files() == ["umsatz.csv"]  # Unterordner bleiben aussen vor


def test_only_the_data_of_the_selected_knowledge_is_listed(tools):
    t, data = tools
    use_scope(data, ("", "."), ("knowledge_2/", "knowledge_2"))
    assert sorted(t.list_files()) == [
        "knowledge_2/vertrag--ab12--tabelle-1.csv",
        "umsatz.csv",
    ]


def test_other_knowledge_is_neither_listed_nor_readable(tools):
    t, data = tools
    use_scope(data, ("knowledge_2/", "knowledge_2"))
    assert t.list_files() == ["knowledge_2/vertrag--ab12--tabelle-1.csv"]
    for name in ("knowledge_3/geheim.csv", "geheim.csv", "umsatz.csv"):
        result = t.run_pandas(name, "result = df.shape")
        assert result.startswith("FEHLER") and "9999" not in result
        assert "nicht verfügbar" in result
    assert "9999" not in t.preview_data("knowledge_3/geheim.csv")


@pytest.mark.parametrize(
    "name",
    [
        "../ausserhalb.csv",
        "knowledge_2/../knowledge_3/geheim.csv",
        "/etc/passwd.csv",
        "../../data/umsatz.csv",
        "knowledge_2/../umsatz.csv",
    ],
)
def test_path_tricks_do_not_escape_the_allowed_folders(tools, name):
    t, data = tools
    use_scope(data, ("knowledge_2/", "knowledge_2"))
    assert t.run_pandas(name, "result = df.shape").startswith("FEHLER")


def test_a_chat_without_knowledge_has_no_files_at_all(tools):
    t, data = tools
    use_scope(data)  # nichts ausgewaehlt
    assert "Keine" in t.list_files()
    assert t.run_pandas("umsatz.csv", "result = 1").startswith("FEHLER")


def test_allowed_files_can_be_analysed(tools):
    t, data = tools
    use_scope(data, ("", "."), ("knowledge_2/", "knowledge_2"))
    assert "350" in t.run_pandas("umsatz.csv", "result = df['betrag'].sum()")
    assert "3800" in t.run_pandas(
        "knowledge_2/vertrag--ab12--tabelle-1.csv", "result = df['betrag'].sum()"
    )


def test_changed_files_are_reloaded(tools):
    t, data = tools
    use_scope(data, ("", "."))
    assert "350" in t.run_pandas("umsatz.csv", "result = df['betrag'].sum()")
    path = data / "umsatz.csv"
    path.write_text("monat,betrag\njan,100\nfeb,250\nmar,1000\n")
    os.utime(path, (time.time() + 5, time.time() + 5))
    assert "1350" in t.run_pandas("umsatz.csv", "result = df['betrag'].sum()")


def test_unknown_file_types_are_not_listed(tools):
    t, data = tools
    (data / "notizen.txt").write_text("x")
    (data / "bild.png").write_bytes(b"x")
    use_scope(data, ("", "."))
    assert t.list_files() == ["umsatz.csv"]


def test_other_knowledge_files_cannot_be_read_by_the_code(tools):
    t, data = tools
    use_scope(data, ("", "."))  # nur die flache Wurzel
    out = t.run_pandas("umsatz.csv", "result = pd.read_csv('/x/geheim.csv')")
    assert out.startswith("FEHLER") and "blockiert" in out


def test_without_a_worker_there_is_no_fallback(tools, monkeypatch):
    from app.agent.toolworker import client as worker_client

    t, _ = tools

    def down(*a, **k):
        raise ConnectionError("weg")

    worker_client.set_override(worker_client.ToolWorkerClient("http://x:1", "t", down))
    out = t.run_pandas("umsatz.csv", "result = 1")
    assert out.startswith("FEHLER: Codeausführung nicht verfügbar") and "x:1" not in out


def test_exec_is_gone_from_the_app():
    import inspect

    from app.agent.tools import custom_tools

    assert "exec(" not in inspect.getsource(custom_tools.CustomTools.run_pandas)
