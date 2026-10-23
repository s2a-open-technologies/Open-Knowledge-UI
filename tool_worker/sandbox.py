"""Pruefung und Start von Pandas-Code. Die Pruefung ist eine Schicht; die andere ist der Kindprozess
(Ressourcenlimits, Audit-Hook) im Container ohne Schluessel."""

import ast
import json
import os
import signal
import subprocess
import sys
import tempfile

BLOCKED_NAMES = {
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "input",
    "breakpoint",
    "memoryview",
    "type",
    "help",
    "dir",
    "exit",
    "quit",
}
# Attribute, die Dateien lesen/schreiben oder Code ausfuehren (ausser den erlaubten to_*-Formen)
BLOCKED_ATTRS = {
    "load",
    "loads",
    "loadtxt",
    "genfromtxt",
    "fromfile",
    "fromregex",
    "memmap",
    "save",
    "savez",
    "savez_compressed",
    "savetxt",
    "eval",
    "system",
    "popen",
    "DataSource",
    "open_memmap",
}
ALLOWED_TO = {
    "to_string",
    "to_dict",
    "to_list",
    "to_numpy",
    "to_frame",
    "to_markdown",
    "to_datetime",
    "to_numeric",
    "to_timedelta",
    "to_period",
    "to_timestamp",
    "to_series",
    "to_records",
}


def check_code(code: str) -> str | None:
    """Fehlertext, wenn der Code nicht ausgefuehrt werden darf, sonst ``None``."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"FEHLER: Syntaxfehler in Zeile {e.lineno}: {e.msg}"
    for node in ast.walk(tree):
        bad = None
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bad = "import"
        elif isinstance(node, ast.Name) and (
            node.id in BLOCKED_NAMES or node.id.startswith("__")
        ):
            bad = node.id
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if (
                a.startswith(("__", "read_"))
                or a in BLOCKED_ATTRS
                or a.startswith("to_")
                and a not in ALLOWED_TO
            ):
                bad = a
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "__" in node.value
        ):
            bad = "__"  # verhindert "{0.__class__}".format(...)
        if bad:
            return f"FEHLER: '{bad}' ist aus Sicherheitsgründen blockiert."
    return None


def format_report(printed: str, result) -> str:
    report = []
    if printed:
        report.append(f"Konsolenausgabe:\n{printed}")
    if result is not None:
        report.append(f"Ergebnis:\n{result}")
    if not report:
        return (
            "Code wurde ausgeführt, aber kein Ergebnis zurückgegeben.\n"
            "Tipp: Weise das Ergebnis der Variable 'result' zu, z. B.: result = df['Spalte'].sum()"
        )
    return "\n\n".join(report)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… (gekürzt auf {limit} Zeichen)"


def run_job(
    path: str, code: str, *, timeout_s: float, memory_mb: int, max_output_chars: int
) -> dict:
    blocked = check_code(code)
    if blocked:
        return {"ok": False, "error": blocked}
    job = json.dumps(
        {
            "path": path,
            "code": code,
            "memory_mb": memory_mb,
            "cpu_s": int(timeout_s) + 5,
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": tmp,
            "TMPDIR": tmp,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        child = os.path.join(os.path.dirname(__file__), "child.py")
        proc = subprocess.Popen(
            [sys.executable, "-I", child],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=tmp,
            env=env,
            close_fds=True,
            start_new_session=True,
        )
        try:
            out, _ = proc.communicate(job.encode(), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
            return {
                "ok": False,
                "error": f"FEHLER: Zeitlimit von {timeout_s:g} s überschritten.",
            }
    try:
        data = json.loads(out.decode("utf-8", "replace"))
    except ValueError:
        return {
            "ok": False,
            "error": "FEHLER: Ausführung abgebrochen (Speicher- oder Rechenlimit erreicht).",
        }
    if data.get("ok"):
        return {
            "ok": True,
            "output": _truncate(data.get("output", ""), max_output_chars),
        }
    return {
        "ok": False,
        "error": _truncate(data.get("error", "FEHLER: unbekannt"), max_output_chars),
    }
