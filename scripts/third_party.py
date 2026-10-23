"""Erzeugt THIRD_PARTY_NOTICES.md aus den installierten Distributions-Metadaten.

Nur Standardbibliothek wie scripts/release.py. Liefert die Laufzeit-Komponenten,
die sich im Projektumfang installieren lassen (uv-Abhaengigkeiten inkl. transitiv),
sowie die npm-Pakete aus frontend/node_modules. Bewusst eine Momentaufnahme:
regenerieren mit ``uv run python scripts/third_party.py`` nach Dependency-Aenderungen.
"""

import json
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "THIRD_PARTY_NOTICES.md"

HEADER = """# Third Party Notices

Erzeugte Liste der Drittkomponenten (Momentaufnahme; neu erzeugen mit
`uv run python scripts/third_party.py`). Python-Seite entspricht der uv.lock-Aufloesung,
Frontend-Seite `frontend/package-lock.json`. Lizenzen binden ihre eigenen Texte; Links
fuer Volltexte. Eigenes Lizenzgut: [LICENSE](./LICENSE) (AGPL-3.0-or-later). Schriften:
Inter (SIL Open Font License 1.1), siehe [LICENSES/OFL-1.1.txt](./LICENSES/OFL-1.1.txt).
"""


def _normal_license(value) -> str:
    if not value:
        return "n/a"
    if isinstance(value, list):  # METADATA License: kann Mehrfachtext sein
        value = value[0] if value else ""
    if value.lower().startswith("license file"):  # klassifizierender Platzhalter
        return "n/a"
    return value.replace(":: ", " :: ").replace("\n", " ")


def python_dists() -> tuple[list[dict], set[str]]:
    rows: dict[str, dict] = {}
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").strip()
        if not name:
            continue
        key = name.lower()
        license_text = _normal_license(dist.metadata.get("License"))
        urls = [u for u in (dist.metadata.get_all("Project-URL") or []) if u]
        homepage = next(
            (u.split(", ")[-1] for u in urls if u.lower().startswith("homepage,")), None
        )
        repo = next(
            (u.split(", ")[-1] for u in urls if u.lower().startswith("repository,")),
            homepage,
        )
        rows[key] = {
            "name": name,
            "version": dist.version,
            "license": license_text,
            "url": repo or homepage or "",
        }
    # Nur Pakete, die die Projektumgebung real zieht: filter auf uv.lock-Namen
    lock = ROOT / "uv.lock"
    known = set()
    if lock.is_file():
        for line in lock.read_text().splitlines():
            line = line.strip()
            if line.startswith("name = "):
                known.add(line[8:].strip('"').lower())
    if known:
        known |= {n for n in known}
        rows = {k: v for k, v in rows.items() if k in known}
    return sorted(rows.values(), key=lambda r: r["name"].lower()), set()


def npm_packages() -> list[dict]:
    rows: dict[str, dict] = {}
    node_modules = ROOT / "frontend" / "node_modules"
    for pkg_json in node_modules.glob("**/package.json"):
        if "/@types/" in pkg_json.as_posix():
            continue
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8"))
            name, version = data.get("name"), str(data.get("version", ""))
            license_text = data.get("license")
            if not name or name.startswith("_"):
                continue
            license_str = license_text if isinstance(license_text, str) else "n/a"
            license_str = str(license_str)
            if license_str.startswith("("):  # Dual-License "("x OR y")" kuerzen
                license_str = license_str.replace("(", "").replace(")", "")
            key = f"{name}@{version}"
            rows[key] = {
                "name": name,
                "version": version,
                "license": license_str or "n/a",
            }
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(
        {(r["name"], r["version"], r["license"]): r for r in rows.values()}.values(),
        key=lambda r: r["name"].lower(),
    )


def main() -> int:
    py_rows, _ = python_dists()
    if not py_rows:
        print(
            "Keine Python-Distributionen gefunden — uv sync laufen lassen.",
            file=sys.stderr,
        )
        return 1
    npm_rows = npm_packages()

    lines = [
        HEADER,
        f"## Python-Abhängigkeiten ({len(py_rows)} Pakete, uv.lock)",
        "",
        "| Paket | Version | Lizenz | Projekt |",
        "|---|---|---|---|",
    ]
    for r in py_rows:
        lines.append(f"| {r['name']} | {r['version']} | {r['license']} | {r['url']} |")
    lines += [
        "",
        f"## Frontend-Abhängigkeiten ({len(npm_rows)} Pakete, frontend/package-lock.json)",
        "",
        "| Paket | Version | Lizenz |",
        "|---|---|---|---|",
    ]
    for r in npm_rows:
        lines.append(f"| {r['name']} | {r['version']} | {r['license']} |")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"{OUT.relative_to(ROOT)}: {len(py_rows)} Python- + {len(npm_rows)} npm-Pakete"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
