"""API-Vertrag: Struktur der JSON-Antworten je Endpunkt aufzeichnen und vergleichen.

Aufzeichnen (``RECORD_API_CONTRACT=<datei>``): ``tests/conftest.py`` meldet jede Antwort eines
Test-Clients an :func:`record`; am Ende der Suite schreibt :func:`dump` je ``(Methode, Pfad, Status)``
die vereinigte Struktur (Schluessel und Typen, rekursiv) als JSON.

Vergleichen: ``python -m tests.contract alt.json neu.json`` meldet Endpunkte, die fehlen, und
Felder, die in der neuen Aufnahme fehlen oder ihren Typ geaendert haben. So fallen Felder auf, die
ein zu knappes Response-Modell stillschweigend verwirft.
"""

import json
import re
import sys
from pathlib import Path

_recorded: dict[str, object] = {}


def normalize_path(rule: str) -> str:
    """``/api/x/<int:id>`` (Flask) und ``/api/x/{id}`` (FastAPI) auf dieselbe Form bringen."""
    return re.sub(r"<(?:[^:>]+:)?([^>]+)>", r"{\1}", rule)


def shape(value):
    """Struktur eines JSON-Werts: dict -> {key: shape}, list -> [shape der Elemente], sonst Typname."""
    if isinstance(value, dict):
        return {k: shape(v) for k, v in value.items()}
    if isinstance(value, list):
        merged = None
        for item in value:
            merged = merge(merged, shape(item))
        return [merged] if merged is not None else []
    if value is None:
        return "null"
    return type(value).__name__


def merge(a, b):
    """Zwei Strukturen vereinigen (Schluessel beider Seiten, Typen als sortierte Liste)."""
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: merge(a.get(k), b.get(k)) for k in sorted(a.keys() | b.keys())}
    if isinstance(a, list) and isinstance(b, list):
        inner = merge(a[0] if a else None, b[0] if b else None)
        return [inner] if inner is not None else []
    if a == b:
        return a
    # Objekt/Liste neben null oder einem Skalar: die Struktur zaehlt (null heisst nur "optional")
    if isinstance(a, (dict, list)):
        return a
    if isinstance(b, (dict, list)):
        return b
    types = set(a if isinstance(a, tuple) else (a,)) | set(
        b if isinstance(b, tuple) else (b,)
    )
    return tuple(sorted(types))


def record(method: str, rule: str, status: int, payload) -> None:
    key = f"{method} {normalize_path(rule)} {status}"
    _recorded[key] = merge(_recorded.get(key), shape(payload))


def dump(path: str) -> None:
    def plain(value):
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, list):
            return [plain(v) for v in value]
        if isinstance(value, tuple):
            return "|".join(value)
        return value

    Path(path).write_text(
        json.dumps(
            {k: plain(_recorded[k]) for k in sorted(_recorded)},
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def compare(old, new, where="") -> list[str]:
    """Abweichungen: Felder, die in ``new`` fehlen, und geaenderte Typen (neue Felder sind erlaubt)."""
    problems = []
    if isinstance(old, dict):
        if not isinstance(new, dict):
            return [f"{where or '/'}: war Objekt, jetzt {new!r}"]
        for key, value in old.items():
            if key not in new:
                problems.append(f"{where}.{key}: fehlt")
            else:
                problems += compare(value, new[key], f"{where}.{key}")
    elif isinstance(old, list):
        if not isinstance(new, list):
            return [f"{where or '/'}: war Liste, jetzt {new!r}"]
        if old and new:
            problems += compare(old[0], new[0], f"{where}[]")
    elif isinstance(new, (dict, list)):
        if old != "null":
            problems.append(
                f"{where or '/'}: Typ {old} -> {'Objekt' if isinstance(new, dict) else 'Liste'}"
            )
    elif not (set(str(old).split("|")) - {"null"}) <= set(str(new).split("|")):
        problems.append(f"{where or '/'}: Typ {old} -> {new}")
    return problems


def main(old_file: str, new_file: str) -> int:
    old = json.loads(Path(old_file).read_text(encoding="utf-8"))
    new = json.loads(Path(new_file).read_text(encoding="utf-8"))
    missing = [k for k in old if k not in new]
    changed = {k: compare(old[k], new[k]) for k in old if k in new}
    changed = {k: v for k, v in changed.items() if v}
    for k in missing:
        print(f"FEHLT   {k}")
    for k, problems in changed.items():
        for p in problems:
            print(f"ANDERS  {k}  {p}")
    print(
        f"{len(old)} Endpunkte im Vertrag, {len(missing)} fehlen, {len(changed)} mit Abweichungen"
    )
    return 1 if missing or changed else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
