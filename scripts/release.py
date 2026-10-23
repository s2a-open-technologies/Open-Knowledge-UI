#!/usr/bin/env python3

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

"""
Release-Werkzeug: Version, Changelog, Tag.

Einzige Quelle der Version ist ``pyproject.toml``; ``frontend/package.json``,
``frontend/package-lock.json`` und ``uv.lock`` werden mitgezogen. Das
GitHub-Release erstellt die Action ``.github/workflows/release.yml`` beim Push
eines ``v*.*.*``-Tags, dieses Skript ruft ``gh release`` nicht selbst auf.

Usage:
    python3 scripts/release.py current
    python3 scripts/release.py check [--tag v0.2.0]
    python3 scripts/release.py bump patch|minor|major|X.Y.Z [--dry-run]
    python3 scripts/release.py notes [--from TAG] [--to REF]
    python3 scripts/release.py notes --version 0.2.0
    python3 scripts/release.py publish [--yes]

Ablauf eines Releases:
    1. Aenderungen mit Conventional Commits (feat:, fix:, ...) committen und
       optional unter "## [Unreleased]" in CHANGELOG.md von Hand ergaenzen.
    2. ``bump minor``  -> Version in allen Dateien, Changelog, Commit, Tag (lokal).
    3. ``publish --yes`` -> Branch und Tag pushen; die Action erstellt das Release.

Erstes Release: Existiert noch kein ``v*``-Tag, ist ``bump 0.1.0`` (gleiche
Version wie aktuell) erlaubt und setzt nur Changelog, Commit und Tag.
"""

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
TAG_PREFIX = "v"
CHANGELOG = "CHANGELOG.md"
UNRELEASED = "## [Unreleased]"
DEFAULT_BRANCH = "master"

CHANGELOG_HEADER = """# Changelog

Alle nennenswerten Änderungen an diesem Projekt werden hier festgehalten.
Das Format orientiert sich an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/),
die Versionierung folgt [Semantic Versioning](https://semver.org/lang/de/).

"""

# Conventional-Commit-Typ -> Abschnitt im Changelog (Reihenfolge = Ausgabe).
SECTIONS = [
    ("breaking", "Inkompatible Änderungen"),
    ("feat", "Neu"),
    ("fix", "Fehlerbehebungen"),
    ("change", "Änderungen"),
    ("docs", "Dokumentation"),
    ("chore", "Wartung"),
    ("other", "Sonstiges"),
]
TYPE_TO_SECTION = {
    "feat": "feat",
    "fix": "fix",
    "refactor": "change",
    "perf": "change",
    "docs": "docs",
    "chore": "chore",
    "ci": "chore",
    "test": "chore",
    "build": "chore",
    "style": "chore",
}
COMMIT_RE = re.compile(
    r"^(?P<type>\w+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?:\s*(?P<text>.+)$"
)


class ReleaseError(Exception):
    """Fachlicher Fehler; wird als Meldung ausgegeben (Exit 1)."""


# --------------------------------------------------------------------------
# Version
# --------------------------------------------------------------------------


def parse_version(text: str) -> tuple[int, int, int]:
    m = SEMVER.match(text or "")
    if not m:
        raise ReleaseError(f"Ungültige Version: {text!r} (erwartet X.Y.Z)")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def next_version(current: str, kind: str) -> str:
    """Berechnet die neue Version aus ``patch|minor|major`` oder ``X.Y.Z``."""
    major, minor, patch = parse_version(current)
    if kind == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    if kind == "major":
        return f"{major + 1}.0.0"
    parse_version(kind)
    return kind


# Je Datei: (Pfad, Regex mit Gruppen 1=Praefix, 2=Version, 3=Suffix). Es wird
# nur das erste Vorkommen ersetzt; formatiert wird nichts um.
def _patterns(old: str | None = None) -> list[tuple[str, re.Pattern]]:
    v = re.escape(old) if old else r"\d+\.\d+\.\d+"
    return [
        (
            "pyproject.toml",
            re.compile(
                rf'(\[project\][^\[]*?^version\s*=\s*")({v})(")',
                re.MULTILINE | re.DOTALL,
            ),
        ),
        (
            "frontend/package.json",
            re.compile(rf'^(\s*"version"\s*:\s*")({v})(")', re.MULTILINE),
        ),
        # Zwei Stellen: Wurzel und packages[""]
        (
            "frontend/package-lock.json",
            re.compile(rf'^(\s*"version"\s*:\s*")({v})(")', re.MULTILINE),
        ),
        (
            "frontend/package-lock.json#packages",
            re.compile(
                rf'("packages"\s*:\s*\{{\s*""\s*:\s*\{{[^}}]*?"version"\s*:\s*")({v})(")',
                re.DOTALL,
            ),
        ),
        (
            "uv.lock",
            re.compile(
                rf'(\[\[package\]\]\nname = "open-knowledge-ui"\nversion = ")({v})(")'
            ),
        ),
    ]


def read_versions(root: Path) -> dict[str, str | None]:
    """Version je Datei (``None`` = Datei fehlt bzw. Muster nicht gefunden)."""
    found: dict[str, str | None] = {}
    for name, pattern in _patterns():
        path = root / name.split("#")[0]
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        m = pattern.search(text)
        found[name] = m.group(2) if m else None
    return found


def current_version(root: Path) -> str:
    """Aktuelle Version; Fehler, wenn die Dateien nicht uebereinstimmen."""
    versions = read_versions(root)
    distinct = {v for v in versions.values() if v is not None}
    missing = [n for n, v in versions.items() if v is None]
    if missing:
        raise ReleaseError(f"Version nicht gefunden in: {', '.join(missing)}")
    if len(distinct) != 1:
        detail = ", ".join(f"{n}={v}" for n, v in versions.items())
        raise ReleaseError(f"Versionen sind uneinheitlich: {detail}")
    return distinct.pop()


def write_version(root: Path, old: str, new: str) -> list[str]:
    """Ersetzt ``old`` durch ``new`` in allen Dateien; liefert die geaenderten Dateien."""
    changed: list[str] = []
    for name, pattern in _patterns(old):
        path = root / name.split("#")[0]
        text = path.read_text(encoding="utf-8")
        new_text, count = pattern.subn(rf"\g<1>{new}\g<3>", text, count=1)
        if count != 1:
            raise ReleaseError(f"Version {old} in {name} nicht gefunden")
        path.write_text(new_text, encoding="utf-8")
        rel = name.split("#")[0]
        if rel not in changed:
            changed.append(rel)
    return changed


# --------------------------------------------------------------------------
# Git
# --------------------------------------------------------------------------


def git(root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if check and proc.returncode != 0:
        raise ReleaseError(
            f"git {' '.join(args)} fehlgeschlagen: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def release_tags(root: Path) -> list[str]:
    out = git(root, "tag", "--list", f"{TAG_PREFIX}*", "--sort=-v:refname")
    return [t for t in out.splitlines() if SEMVER.match(t[len(TAG_PREFIX) :])]


def repo_url(root: Path) -> str | None:
    """``https://github.com/owner/repo`` aus dem Remote ``origin`` (oder ``None``)."""
    url = git(root, "remote", "get-url", "origin", check=False)
    m = re.match(r"^(?:git@|ssh://git@|https?://)([^:/]+)[:/](.+?)(?:\.git)?/?$", url)
    return f"https://{m.group(1)}/{m.group(2)}" if m else None


def commits_since(root: Path, since: str | None, to: str = "HEAD") -> list[dict]:
    rng = f"{since}..{to}" if since else to
    raw = git(root, "log", "--no-merges", "--format=%H%x1f%s%x1f%b%x1e", rng)
    commits = []
    for entry in raw.split("\x1e"):
        entry = entry.strip()
        if not entry:
            continue
        sha, subject, body = (entry.split("\x1f") + ["", ""])[:3]
        commits.append({"sha": sha, "subject": subject.strip(), "body": body})
    return commits


# --------------------------------------------------------------------------
# Notizen und Changelog
# --------------------------------------------------------------------------


def classify(subject: str, body: str = "") -> tuple[str, str] | None:
    """(Abschnitt, Text) eines Commits; ``None`` = nicht aufnehmen."""
    m = COMMIT_RE.match(subject)
    if m and m.group("type") == "chore" and (m.group("scope") or "") == "release":
        return None
    breaking = bool(m and m.group("bang")) or "BREAKING CHANGE" in body
    if m and m.group("type") in TYPE_TO_SECTION:
        text = m.group("text").strip()
        scope = m.group("scope")
        if scope:
            text = f"{scope}: {text}"
        return ("breaking" if breaking else TYPE_TO_SECTION[m.group("type")], text)
    return ("breaking" if breaking else "other", subject.strip())


def render_notes(commits: list[dict]) -> str:
    grouped: dict[str, list[str]] = {}
    for c in commits:
        item = classify(c["subject"], c["body"])
        if item:
            grouped.setdefault(item[0], []).append(item[1])
    lines: list[str] = []
    for key, title in SECTIONS:
        if grouped.get(key):
            lines += [f"### {title}", ""] + [f"- {t}" for t in grouped[key]] + [""]
    return "\n".join(lines).strip()


_LINKDEF = re.compile(r"^\[[^\]]+\]:\s+\S+", re.MULTILINE)


def changelog_section(text: str, version: str) -> str:
    """Inhalt des Abschnitts ``## [version]`` (leer, wenn nicht vorhanden)."""
    m = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n", text, re.MULTILINE)
    if not m:
        return ""
    rest = text[m.end() :]
    end = re.search(r"^## \[|^\[[^\]]+\]:\s+\S+", rest, re.MULTILINE)
    return (rest[: end.start()] if end else rest).strip()


def update_changelog(
    text: str,
    version: str,
    date: str,
    generated_notes: str,
    previous: str | None,
    url: str | None,
) -> str:
    """Verschiebt ``[Unreleased]`` nach ``[version]`` und pflegt die Vergleichs-Links.

    Ist ``[Unreleased]`` leer, werden ``generated_notes`` (aus den Commits)
    verwendet. Ohne beides entsteht ein Hinweis, dass nichts erfasst wurde.
    """
    if not text.strip():
        text = CHANGELOG_HEADER + UNRELEASED + "\n"
    if UNRELEASED not in text:
        head_end = re.search(r"^## \[", text, re.MULTILINE)
        pos = head_end.start() if head_end else len(text)
        text = text[:pos] + UNRELEASED + "\n\n" + text[pos:]

    if re.search(rf"^## \[{re.escape(version)}\]", text, re.MULTILINE):
        raise ReleaseError(
            f"CHANGELOG.md enthält bereits einen Abschnitt für {version}"
        )

    start = text.index(UNRELEASED) + len(UNRELEASED)
    rest = text[start:]
    end = re.search(r"^## \[|^\[[^\]]+\]:\s+\S+", rest, re.MULTILINE)
    body = (rest[: end.start()] if end else rest).strip()
    tail = rest[end.start() :] if end else ""

    body = body or generated_notes or "- Keine Änderungen erfasst."
    new_text = (
        text[:start].rstrip()
        + f"\n\n## [{version}] - {date}\n\n{body}\n\n"
        + tail.lstrip("\n")
    )
    return _update_links(new_text.rstrip() + "\n", version, previous, url)


def _update_links(
    text: str, version: str, previous: str | None, url: str | None
) -> str:
    if not url:
        return text
    links = {
        m.group(1): m.group(0)
        for m in re.finditer(r"^\[([^\]]+)\]:\s+\S+.*$", text, re.MULTILINE)
    }
    links["Unreleased"] = f"[Unreleased]: {url}/compare/{TAG_PREFIX}{version}...HEAD"
    links[version] = (
        f"[{version}]: {url}/compare/{TAG_PREFIX}{previous}...{TAG_PREFIX}{version}"
        if previous
        else f"[{version}]: {url}/releases/tag/{TAG_PREFIX}{version}"
    )
    body = _LINKDEF.sub("", text).rstrip()

    def order(name: str) -> tuple:
        return (
            (0, ())
            if name == "Unreleased"
            else (1, tuple(-n for n in parse_version(name)))
        )

    ordered = sorted(links, key=order)
    return body + "\n\n" + "\n".join(links[n] for n in ordered) + "\n"


# --------------------------------------------------------------------------
# Kommandos
# --------------------------------------------------------------------------


def cmd_current(root: Path, args) -> int:
    print(current_version(root))
    return 0


def cmd_check(root: Path, args) -> int:
    version = current_version(root)
    if args.tag and args.tag != f"{TAG_PREFIX}{version}":
        raise ReleaseError(
            f"Tag {args.tag} passt nicht zur Version {version} in den Dateien"
        )
    if not args.no_changelog:
        path = root / CHANGELOG
        if not path.exists() or not changelog_section(
            path.read_text(encoding="utf-8"), version
        ):
            raise ReleaseError(f"CHANGELOG.md hat keinen Abschnitt für {version}")
    print(f"OK: Version {version}")
    return 0


def cmd_notes(root: Path, args) -> int:
    if args.version:
        path = root / CHANGELOG
        section = (
            changelog_section(path.read_text(encoding="utf-8"), args.version)
            if path.exists()
            else ""
        )
        if not section:
            raise ReleaseError(f"CHANGELOG.md hat keinen Abschnitt für {args.version}")
        print(section)
        return 0
    since = args.from_ref
    if since is None:
        tags = release_tags(root)
        since = tags[0] if tags else None
    print(render_notes(commits_since(root, since, args.to)))
    return 0


def cmd_bump(root: Path, args) -> int:
    current = current_version(root)
    tags = release_tags(root)
    new = next_version(current, args.kind)

    first_release = not tags and new == current
    if not first_release and parse_version(new) <= parse_version(current):
        raise ReleaseError(f"Neue Version {new} muss größer sein als {current}")
    tag = f"{TAG_PREFIX}{new}"
    if tag in tags:
        raise ReleaseError(f"Tag {tag} existiert bereits")

    problems: list[str] = []
    if not args.allow_dirty and git(root, "status", "--porcelain"):
        problems.append(
            "Arbeitsverzeichnis ist nicht sauber. Erst committen (oder --allow-dirty)."
        )
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if branch != args.branch:
        problems.append(
            f"Release von Branch '{branch}', erwartet '{args.branch}' (Override: --branch)"
        )
    if problems and not args.dry_run:
        raise ReleaseError(" ".join(problems))

    previous = tags[0][len(TAG_PREFIX) :] if tags else None
    changelog_path = root / CHANGELOG
    old_text = (
        changelog_path.read_text(encoding="utf-8") if changelog_path.exists() else ""
    )
    generated = render_notes(commits_since(root, tags[0] if tags else None))
    new_text = update_changelog(
        old_text,
        new,
        datetime.date.today().isoformat(),
        generated,
        previous,
        repo_url(root),
    )

    print(f"Version {current} -> {new} (Tag {tag}, Branch {branch})")
    notes = changelog_section(new_text, new)
    if args.dry_run:
        print("\n--- Trockenlauf: nichts wurde geändert ---")
        for problem in problems:
            print(f"WÜRDE ABBRECHEN: {problem}")
        print("Changelog-Abschnitt:\n")
        print(notes)
        return 0

    changed = write_version(root, current, new) if new != current else []
    changelog_path.write_text(new_text, encoding="utf-8")
    git(root, "add", CHANGELOG, *changed)
    git(root, "commit", "-m", f"chore(release): {tag}")
    if not args.no_tag:
        git(root, "tag", "-a", tag, "-m", f"Release {new}\n\n{notes}")
        print(f"Commit und Tag {tag} lokal angelegt. Nächster Schritt: publish --yes")
    else:
        print("Commit angelegt (ohne Tag).")
    return 0


def cmd_publish(root: Path, args) -> int:
    tags = release_tags(root)
    if not tags:
        raise ReleaseError("Kein Release-Tag vorhanden (erst bump)")
    tag = tags[0]
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD")
    if not git(root, "remote", "get-url", "origin", check=False):
        raise ReleaseError("Kein Remote 'origin' konfiguriert")
    commands = [["push", "origin", branch], ["push", "origin", tag]]
    for c in commands:
        print("git " + " ".join(c))
    if not args.yes:
        print(
            "\nNur Anzeige. Mit --yes wirklich pushen; danach erstellt die Action das Release."
        )
        return 0
    if args.dry_run:
        return 0
    for c in commands:
        git(root, *c)
    print(f"Gepusht. Die Action 'release' erstellt das GitHub-Release für {tag}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Release-Werkzeug (Version, Changelog, Tag)"
    )
    p.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("current", help="Aktuelle Version anzeigen").set_defaults(
        func=cmd_current
    )

    c = sub.add_parser("check", help="Versionen und Changelog prüfen")
    c.add_argument("--tag", help="Erwarteter Tag, z. B. v0.2.0")
    c.add_argument("--no-changelog", action="store_true")
    c.set_defaults(func=cmd_check)

    n = sub.add_parser("notes", help="Release-Notizen ausgeben")
    n.add_argument("--version", help="Abschnitt aus CHANGELOG.md")
    n.add_argument("--from", dest="from_ref", help="Start-Tag (Default: letzter Tag)")
    n.add_argument("--to", default="HEAD")
    n.set_defaults(func=cmd_notes)

    b = sub.add_parser(
        "bump", help="Version erhöhen, Changelog pflegen, committen, taggen"
    )
    b.add_argument("kind", help="patch | minor | major | X.Y.Z")
    b.add_argument("--dry-run", action="store_true")
    b.add_argument("--allow-dirty", action="store_true")
    b.add_argument("--branch", default=DEFAULT_BRANCH, help="Erlaubter Branch")
    b.add_argument("--no-tag", action="store_true")
    b.set_defaults(func=cmd_bump)

    pub = sub.add_parser(
        "publish", help="Branch und Tag pushen (Release erstellt die Action)"
    )
    pub.add_argument("--yes", action="store_true", help="wirklich pushen")
    pub.add_argument("--dry-run", action="store_true")
    pub.set_defaults(func=cmd_publish)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args.root, args)
    except ReleaseError as e:
        print(f"Fehler: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
