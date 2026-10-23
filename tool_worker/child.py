"""Kindprozess: laedt die Datei, schaltet Beschraenkungen scharf und fuehrt den Code aus.

Laeuft mit ``python -I``. Ergebnis als eine JSON-Zeile auf dem echten stdout.
"""

import contextlib
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_REAL_STDOUT = sys.__stdout__

READ_ALLOWED = (
    "/proc/cpuinfo",
    "/proc/meminfo",
    "/proc/self/",
    "/sys/devices/system/cpu",
    "/etc/localtime",
    "/usr/share/zoneinfo",
    "/dev/urandom",
    "/dev/null",
    "/etc/mime.types",
)
DENY_PREFIXES = (
    "socket.",
    "ctypes.",
    "subprocess.",
    "os.exec",
    "os.spawn",
    "os.posix_spawn",
    "os.fork",
)
DENY_EVENTS = {
    "os.system",
    "os.remove",
    "os.rename",
    "os.mkdir",
    "os.rmdir",
    "os.chmod",
    "os.chown",
    "os.symlink",
    "os.link",
    "os.truncate",
    "os.putenv",
    "os.kill",
    "pickle.find_class",
    "shutil.copyfile",
    "shutil.rmtree",
    "shutil.move",
}
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _roots() -> list[str]:
    import sysconfig

    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    roots.update(v for v in sysconfig.get_paths().values() if v)
    roots.update(p for p in sys.path if p)
    return [os.path.realpath(r) for r in roots if r and os.path.isdir(r)]


def _install_hook(roots: list[str]) -> None:
    def under(path: str) -> bool:
        real = os.path.realpath(path)
        if any(real == r or real.startswith(r + os.sep) for r in roots):
            return True
        return any(real.startswith(a) for a in READ_ALLOWED)

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            path, mode, flags = args[0], args[1], args[2]
            if isinstance(path, int):
                return
            if isinstance(path, bytes):
                path = os.fsdecode(path)
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & WRITE_FLAGS
            )
            if writing or not under(str(path)):
                raise PermissionError("Dateizugriff blockiert")
        elif event in ("os.listdir", "os.scandir"):
            p = args[0]
            if p is not None and not under(
                os.fsdecode(p) if not isinstance(p, int) else "/"
            ):
                raise PermissionError("Dateizugriff blockiert")
        elif event in DENY_EVENTS or event.startswith(DENY_PREFIXES):
            raise PermissionError("Aktion blockiert")

    sys.addaudithook(hook)


def _limits(memory_mb: int, cpu_s: int) -> None:
    import resource

    mem = int(memory_mb) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


SAFE_BUILTINS = {
    n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
    for n in (
        "range",
        "len",
        "sum",
        "min",
        "max",
        "abs",
        "round",
        "sorted",
        "enumerate",
        "zip",
        "map",
        "filter",
        "list",
        "dict",
        "set",
        "tuple",
        "str",
        "int",
        "float",
        "bool",
        "print",
        "isinstance",
        "all",
        "any",
        "reversed",
        "divmod",
        "pow",
        "repr",
        "Exception",
        "ValueError",
        "KeyError",
        "TypeError",
        "RuntimeError",
        "IndexError",
        "AttributeError",
        "ZeroDivisionError",
    )
}
SAFE_BUILTINS.update({"True": True, "False": False, "None": None})


def main() -> None:
    job = json.loads(sys.stdin.read())
    out: dict
    try:
        _limits(job["memory_mb"], job["cpu_s"])
        import numpy as np
        import pandas as pd

        from tool_worker.frames import FrameError, load_frame
        from tool_worker.sandbox import format_report

        try:
            df = load_frame(job["path"])
        except FrameError as e:
            raise SystemExit(_emit({"ok": False, "error": f"FEHLER: {e}"}))
        if df.empty:
            raise SystemExit(
                _emit(
                    {
                        "ok": True,
                        "output": "Die Datei ist leer oder enthält keine Daten.",
                    }
                )
            )
        code = compile(job["code"], "<analyse>", "exec")
        _install_hook(_roots())
        scope = {
            "__builtins__": SAFE_BUILTINS,
            "df": df,
            "pd": pd,
            "np": np,
            "result": None,
        }
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code, scope)  # noqa: S102 - bewusst: geprueft, begrenzt, isoliert
        out = {
            "ok": True,
            "output": format_report(buf.getvalue().strip(), scope.get("result")),
        }
    except SystemExit:
        raise
    except MemoryError:
        out = {"ok": False, "error": "FEHLER: Speicherlimit erreicht."}
    except BaseException as e:  # noqa: BLE001 - alles dem Modell als Text melden
        out = {
            "ok": False,
            "error": f"FEHLER bei der Ausführung: {type(e).__name__}: {e}\n\nKorrigiere den Code und versuche erneut.",
        }
    _emit(out)


def _emit(obj: dict) -> int:
    _REAL_STDOUT.write(json.dumps(obj))
    _REAL_STDOUT.flush()
    return 0


if __name__ == "__main__":
    main()
