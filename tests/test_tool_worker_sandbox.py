import sys
import textwrap

import pytest

from tool_worker import sandbox


@pytest.fixture
def csv(tmp_path):
    p = tmp_path / "umsatz.csv"
    p.write_text("monat,betrag\njan,100\nfeb,250\n", encoding="utf-8")
    return str(p)


def run(csv, code, **kw):
    kw.setdefault("timeout_s", 20)
    kw.setdefault("memory_mb", 1024)
    kw.setdefault("max_output_chars", 2000)
    return sandbox.run_job(csv, code, **kw)


# --- AST-Prüfung -----------------------------------------------------------
@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "from os import path",
        "__import__('os')",
        "eval('1')",
        "exec('x=1')",
        "compile('1','x','eval')",
        "open('/etc/passwd')",
        "getattr(df, 'x')",
        "df.__class__",
        "x = df.__dict__",
        "'{0.__class__}'.format(df)",
        "pd.read_csv('/data/geheim.csv')",
        "pd.read_pickle('x')",
        "np.load('x.npy')",
        "np.fromfile('x')",
        "np.genfromtxt('x')",
        "df.to_csv('x.csv')",
        "df.to_excel('x.xlsx')",
        "df.to_pickle('x')",
        "pd.eval('1+1')",
        "globals()",
        "type(df)",
    ],
)
def test_dangerous_code_is_rejected(code):
    assert sandbox.check_code(code) is not None


@pytest.mark.parametrize(
    "code",
    [
        "result = df['betrag'].sum()",
        "result = df.groupby('monat')['betrag'].mean()",
        "print(df.describe())",
        "result = df.to_string()",
        "result = df.head().to_dict()",
        "result = np.mean(df['betrag'])",
        "result = df.query('betrag > 100')",
    ],
)
def test_ordinary_analysis_is_allowed(code):
    assert sandbox.check_code(code) is None


def test_syntax_errors_are_reported_not_raised():
    assert "Syntax" in sandbox.check_code("result = (")


def test_substring_names_are_not_blocked():
    # alte Teilstring-Prüfung traf "open" in "reopen" und "os." in "pos."
    assert sandbox.check_code("reopened = 1\npos = 2\nresult = reopened + pos") is None


# --- Ausführung ------------------------------------------------------------
def test_result_and_console_output_are_reported(csv):
    out = run(csv, "print('hallo')\nresult = df['betrag'].sum()")
    assert out["ok"] is True
    assert "Konsolenausgabe:\nhallo" in out["output"]
    assert "Ergebnis:\n350" in out["output"]


def test_no_result_gives_hint(csv):
    out = run(csv, "x = 1")
    assert out["ok"] is True
    assert "kein Ergebnis zurückgegeben" in out["output"]


def test_user_exception_is_returned_as_text(csv):
    out = run(csv, "result = df['gibt_es_nicht']")
    assert out["ok"] is False
    assert "KeyError" in out["error"]


def test_blocked_code_never_starts_a_process(csv):
    out = run(csv, "import os")
    assert out["ok"] is False and "blockiert" in out["error"]


def test_cp1252_umlauts_load(tmp_path):
    p = tmp_path / "k.csv"
    p.write_bytes("ort,wert\nKöln,1\nMünchen,2\n".encode("cp1252"))
    out = run(str(p), "result = ', '.join(df['ort'])")
    assert "Köln" in out["output"] and "München" in out["output"]


def test_excel_loads(tmp_path):
    import pandas as pd

    p = tmp_path / "x.xlsx"
    pd.DataFrame({"a": [1, 2, 3]}).to_excel(p, index=False)
    assert "6" in run(str(p), "result = df['a'].sum()")["output"]


def test_empty_file_is_reported(tmp_path):
    p = tmp_path / "leer.csv"
    p.write_text("a,b\n", encoding="utf-8")
    out = run(str(p), "result = 1")
    assert out["ok"] is True and "leer" in out["output"]


def test_endless_loop_is_killed(csv):
    out = run(csv, "while True:\n    pass", timeout_s=2)
    assert out["ok"] is False and "Zeitlimit" in out["error"]


def test_memory_hog_is_stopped(csv):
    out = run(csv, "x = [0] * (10**10)", memory_mb=512)
    assert out["ok"] is False


def test_output_is_truncated(csv):
    out = run(csv, "print('x' * 100000)\nresult = 1", max_output_chars=500)
    assert (
        out["ok"] is True and len(out["output"]) <= 700 and "gekürzt" in out["output"]
    )


# --- Audit-Hook: das, was die AST-Prüfung nicht sieht -----------------------
def test_other_files_cannot_be_read_via_numpy_helpers(csv, tmp_path):
    secret = tmp_path / "geheim.txt"
    secret.write_text("TOPSECRET")
    code = textwrap.dedent(
        f"""
        f = np.lib.npyio.DataSource().open({str(secret)!r})
        result = f.read()
        """
    )
    # np.DataSource ist kein verbotener Name; der Audit-Hook muss greifen
    out = run(csv, code)
    assert out["ok"] is False and "TOPSECRET" not in str(out)


def test_network_is_blocked(csv):
    code = "import_free = np.frompyfunc(lambda: 1, 0, 1)\nresult = 1"
    assert run(csv, code)["ok"] is True  # Sanity: Lambda/frompyfunc erlaubt


def test_files_cannot_be_written(csv, tmp_path):
    target = tmp_path / "out.txt"
    code = f"df.to_string(buf={str(target)!r})"
    out = run(csv, code)
    assert not target.exists()
    assert out["ok"] is False


# --- Audit-Hook allein (ohne AST-Prüfung): zweite Schicht muss für sich halten --
def run_child_directly(path, code, memory_mb=1024):
    import json
    import os
    import subprocess

    child = os.path.join(os.path.dirname(sandbox.__file__), "child.py")
    job = json.dumps({"path": path, "code": code, "memory_mb": memory_mb, "cpu_s": 20})
    out = subprocess.run(
        [sys.executable, "-I", child],
        input=job.encode(),
        capture_output=True,
        timeout=30,
        check=False,
    ).stdout
    return json.loads(out)


def test_hook_blocks_reading_other_files_without_ast(csv, tmp_path):
    secret = tmp_path / "geheim.txt"
    secret.write_text("TOPSECRET")
    out = run_child_directly(
        csv, f"result = np.lib.npyio.DataSource().open({str(secret)!r}).read()"
    )
    assert (
        out["ok"] is False
        and "blockiert" in out["error"]
        and "TOPSECRET" not in str(out)
    )


def test_hook_blocks_writing_without_ast(csv, tmp_path):
    target = tmp_path / "out.txt"
    out = run_child_directly(csv, f"df.to_csv({str(target)!r})")
    assert out["ok"] is False and not target.exists()


def test_hook_blocks_network_without_ast(csv):
    out = run_child_directly(csv, "pd.io.common.urlopen('http://127.0.0.1:9')")
    assert out["ok"] is False and "blockiert" in out["error"]
