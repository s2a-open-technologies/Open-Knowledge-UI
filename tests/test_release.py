import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "release.py"
spec = importlib.util.spec_from_file_location("release_script", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)

PYPROJECT = """[project]
name = "open-knowledge-ui"
version = "0.1.0"
dependencies = ["x[y]>=1.0"]

[tool.other]
version = "9.9.9"
"""
PACKAGE_JSON = '{\n  "name": "frontend",\n  "private": true,\n  "version": "0.1.0",\n  "type": "module"\n}\n'
PACKAGE_LOCK = (
    json.dumps(
        {
            "name": "frontend",
            "version": "0.1.0",
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "frontend", "version": "0.1.0", "dependencies": {}},
                "node_modules/dep": {"version": "0.1.0"},
            },
        },
        indent=2,
    )
    + "\n"
)
UV_LOCK = """version = 1

[[package]]
name = "other"
version = "0.1.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "open-knowledge-ui"
version = "0.1.0"
source = { virtual = "." }
"""


@pytest.fixture(autouse=True)
def git_identity(monkeypatch):
    """Frische CI-Runner haben keine Git-Identitaet; das Release-Skript committet im Testrepo."""
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "test@example.com")


def sh(root, *args, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=check,
        env={
            "GIT_AUTHOR_NAME": "T",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "T",
            "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(root),
        },
    ).stdout.strip()


def commit(root, message, filename="x.txt"):
    (root / filename).write_text(
        message + "\n" + (root / filename).read_text()
        if (root / filename).exists()
        else message
    )
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "frontend").mkdir(parents=True)
    (root / "pyproject.toml").write_text(PYPROJECT)
    (root / "frontend" / "package.json").write_text(PACKAGE_JSON)
    (root / "frontend" / "package-lock.json").write_text(PACKAGE_LOCK)
    (root / "uv.lock").write_text(UV_LOCK)
    sh(root, "init", "-q", "-b", "master")
    sh(root, "add", "-A")
    sh(root, "commit", "-q", "-m", "chore: initial")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    sh(root, "remote", "add", "origin", str(remote))
    return root


def run(root, *argv):
    return release.main(["--root", str(root), *argv])


# --- Version ---------------------------------------------------------------


@pytest.mark.parametrize(
    "current,kind,expected",
    [
        ("0.1.0", "patch", "0.1.1"),
        ("0.1.9", "minor", "0.2.0"),
        ("1.2.3", "major", "2.0.0"),
        ("0.1.0", "1.4.2", "1.4.2"),
    ],
)
def test_next_version(current, kind, expected):
    assert release.next_version(current, kind) == expected


@pytest.mark.parametrize("bad", ["1.2", "v1.2.3", "1.2.3-rc1", "", "abc"])
def test_invalid_versions_are_rejected(bad):
    with pytest.raises(release.ReleaseError):
        release.next_version("0.1.0", bad)


def test_versions_are_read_from_all_four_files(repo):
    versions = release.read_versions(repo)
    assert set(versions.values()) == {"0.1.0"} and len(versions) == 5
    assert release.current_version(repo) == "0.1.0"


def test_inconsistent_versions_are_reported(repo):
    (repo / "frontend" / "package.json").write_text(
        PACKAGE_JSON.replace("0.1.0", "0.2.0")
    )
    with pytest.raises(release.ReleaseError, match="uneinheitlich"):
        release.current_version(repo)


def test_write_version_changes_only_the_projects_own_entries(repo):
    changed = release.write_version(repo, "0.1.0", "0.2.0")
    assert sorted(changed) == [
        "frontend/package-lock.json",
        "frontend/package.json",
        "pyproject.toml",
        "uv.lock",
    ]
    assert 'version = "0.2.0"' in (repo / "pyproject.toml").read_text()
    assert (
        'version = "9.9.9"' in (repo / "pyproject.toml").read_text()
    )  # anderer Abschnitt unberuehrt
    lock = json.loads((repo / "frontend" / "package-lock.json").read_text())
    assert lock["version"] == "0.2.0" and lock["packages"][""]["version"] == "0.2.0"
    assert (
        lock["packages"]["node_modules/dep"]["version"] == "0.1.0"
    )  # Fremdpaket unberuehrt
    uv = (repo / "uv.lock").read_text()
    assert 'name = "other"\nversion = "0.1.0"' in uv
    assert 'name = "open-knowledge-ui"\nversion = "0.2.0"' in uv
    assert release.current_version(repo) == "0.2.0"
    # Formatierung bleibt: nur die Versionen unterscheiden sich
    assert (repo / "frontend" / "package.json").read_text() == PACKAGE_JSON.replace(
        "0.1.0", "0.2.0"
    )


# --- Commits und Changelog -------------------------------------------------


@pytest.mark.parametrize(
    "subject,body,expected",
    [
        ("feat: neue Sache", "", ("feat", "neue Sache")),
        ("fix(chat): kaputt", "", ("fix", "chat: kaputt")),
        ("refactor: aufgeraeumt", "", ("change", "aufgeraeumt")),
        ("ci: workflow", "", ("chore", "workflow")),
        ("feat!: umbau", "", ("breaking", "umbau")),
        ("fix: x", "BREAKING CHANGE: api", ("breaking", "x")),
        ("Refactor code structure", "", ("other", "Refactor code structure")),
        ("chore(release): v0.2.0", "", None),
    ],
)
def test_classify(subject, body, expected):
    assert release.classify(subject, body) == expected


def test_render_notes_groups_in_fixed_order():
    notes = release.render_notes(
        [
            {"subject": "fix: b", "body": ""},
            {"subject": "feat: a", "body": ""},
            {"subject": "irgendwas", "body": ""},
            {"subject": "chore(release): v0.1.0", "body": ""},
        ]
    )
    assert (
        notes.index("### Neu")
        < notes.index("### Fehlerbehebungen")
        < notes.index("### Sonstiges")
    )
    assert "- a" in notes and "- b" in notes and "release" not in notes


BASE = "# Changelog\n\nText\n\n## [Unreleased]\n\n"


def test_unreleased_body_is_moved_to_the_new_version():
    text = BASE + "### Neu\n\n- Handgeschrieben\n"
    out = release.update_changelog(
        text, "0.1.0", "2026-10-12", "GENERIERT", None, "https://github.com/o/r"
    )
    assert (
        "## [Unreleased]\n\n## [0.1.0] - 2026-10-12\n\n### Neu\n\n- Handgeschrieben"
        in out
    )
    assert "GENERIERT" not in out
    assert "[Unreleased]: https://github.com/o/r/compare/v0.1.0...HEAD" in out
    assert "[0.1.0]: https://github.com/o/r/releases/tag/v0.1.0" in out


def test_empty_unreleased_uses_generated_notes_and_links_are_ordered():
    first = release.update_changelog(
        BASE + "- eins\n", "0.1.0", "2026-10-01", "", None, "https://github.com/o/r"
    )
    second = release.update_changelog(
        first,
        "0.2.0",
        "2026-10-12",
        "### Neu\n\n- aus Commits",
        "0.1.0",
        "https://github.com/o/r",
    )
    assert "## [0.2.0] - 2026-10-12\n\n### Neu\n\n- aus Commits" in second
    assert second.index("## [0.2.0]") < second.index("## [0.1.0]")
    links = second[second.index("[Unreleased]:") :]
    assert links.splitlines()[0].startswith(
        "[Unreleased]: https://github.com/o/r/compare/v0.2.0"
    )
    assert "[0.2.0]: https://github.com/o/r/compare/v0.1.0...v0.2.0" in links
    assert links.index("[0.2.0]") < links.index("[0.1.0]")
    assert release.changelog_section(second, "0.1.0") == "- eins"
    assert release.changelog_section(second, "0.2.0") == "### Neu\n\n- aus Commits"
    assert release.changelog_section(second, "9.9.9") == ""


def test_changelog_is_created_when_missing_and_duplicates_are_rejected():
    out = release.update_changelog("", "0.1.0", "2026-10-12", "- x", None, None)
    assert (
        out.startswith("# Changelog")
        and "## [0.1.0] - 2026-10-12" in out
        and "[0.1.0]:" not in out
    )
    with pytest.raises(release.ReleaseError, match="bereits"):
        release.update_changelog(out, "0.1.0", "2026-10-12", "", None, None)


# --- bump ------------------------------------------------------------------


def test_first_release_keeps_the_version_and_creates_commit_and_tag(repo):
    commit(repo, "feat: erstes Feature", "a.txt")
    assert run(repo, "bump", "0.1.0") == 0

    assert release.current_version(repo) == "0.1.0"
    assert release.release_tags(repo) == ["v0.1.0"]
    assert sh(repo, "log", "-1", "--format=%s") == "chore(release): v0.1.0"
    assert sh(repo, "status", "--porcelain") == ""
    changelog = (repo / "CHANGELOG.md").read_text()
    assert "## [0.1.0]" in changelog and "erstes Feature" in changelog
    assert "erstes Feature" in sh(
        repo, "tag", "-n99", "v0.1.0"
    )  # annotierter Tag mit Notizen
    assert sh(repo, "cat-file", "-t", "v0.1.0") == "tag"


def test_second_release_bumps_files_and_uses_only_new_commits(repo):
    commit(repo, "feat: alt", "a.txt")
    run(repo, "bump", "0.1.0")
    commit(repo, "fix: neu behoben", "b.txt")
    commit(repo, "docs: erklaert", "c.txt")

    assert run(repo, "bump", "minor") == 0

    assert release.current_version(repo) == "0.2.0"
    assert set(release.read_versions(repo).values()) == {"0.2.0"}
    assert release.release_tags(repo) == ["v0.2.0", "v0.1.0"]
    section = release.changelog_section((repo / "CHANGELOG.md").read_text(), "0.2.0")
    assert "neu behoben" in section and "erklaert" in section and "alt" not in section
    assert "### Fehlerbehebungen" in section and "### Dokumentation" in section


def test_dry_run_changes_nothing(repo, capsys):
    before = sh(repo, "rev-parse", "HEAD")
    assert run(repo, "bump", "minor", "--dry-run") == 0
    assert sh(repo, "rev-parse", "HEAD") == before
    assert sh(repo, "status", "--porcelain") == ""
    assert release.release_tags(repo) == []
    assert not (repo / "CHANGELOG.md").exists()
    assert release.current_version(repo) == "0.1.0"
    assert "Trockenlauf" in capsys.readouterr().out


def test_dirty_tree_blocks_the_release_even_for_untracked_files(repo, capsys):
    (repo / "neu.txt").write_text("x")
    assert run(repo, "bump", "minor") == 1
    assert "nicht sauber" in capsys.readouterr().err
    assert release.release_tags(repo) == []
    assert release.current_version(repo) == "0.1.0"
    assert run(repo, "bump", "minor", "--allow-dirty") == 0  # bewusste Ausnahme


def test_wrong_branch_blocks_the_release(repo, capsys):
    sh(repo, "checkout", "-q", "-b", "feature")
    assert run(repo, "bump", "minor") == 1
    assert "master" in capsys.readouterr().err
    assert run(repo, "bump", "minor", "--branch", "feature") == 0


def test_version_must_increase_and_tag_must_not_exist(repo, capsys):
    run(repo, "bump", "0.1.0")
    commit(repo, "fix: x", "x.txt")
    assert (
        run(repo, "bump", "0.1.0") == 1
    )  # gleiche Version, aber es gibt schon einen Tag
    assert run(repo, "bump", "patch") == 0
    assert run(repo, "bump", "0.0.5") == 1
    err = capsys.readouterr().err
    assert "größer" in err
    sh(repo, "tag", "v0.5.0")
    commit(repo, "fix: y", "y.txt")
    assert run(repo, "bump", "0.5.0") == 1
    assert "existiert bereits" in capsys.readouterr().err


def test_failed_bump_leaves_no_partial_changes(repo):
    sh(repo, "tag", "v0.2.0")
    assert run(repo, "bump", "0.2.0") == 1
    assert sh(repo, "status", "--porcelain") == ""
    assert release.current_version(repo) == "0.1.0"


# --- notes / check / publish -----------------------------------------------


def test_notes_for_a_version_come_from_the_changelog(repo, capsys):
    commit(repo, "feat: x", "a.txt")
    run(repo, "bump", "0.1.0")
    capsys.readouterr()
    assert run(repo, "notes", "--version", "0.1.0") == 0
    assert "x" in capsys.readouterr().out
    assert run(repo, "notes", "--version", "7.7.7") == 1


def test_check_validates_tag_and_changelog(repo, capsys):
    assert run(repo, "check") == 1  # kein Changelog
    assert "Abschnitt" in capsys.readouterr().err
    commit(repo, "feat: x", "a.txt")
    run(repo, "bump", "0.1.0")
    assert run(repo, "check", "--tag", "v0.1.0") == 0
    assert run(repo, "check", "--tag", "v0.2.0") == 1
    assert "passt nicht" in capsys.readouterr().err


def test_publish_only_shows_commands_without_yes(repo, tmp_path, capsys):
    commit(repo, "feat: x", "a.txt")
    run(repo, "bump", "0.1.0")
    remote = tmp_path / "remote.git"

    assert run(repo, "publish") == 0
    assert "git push origin v0.1.0" in capsys.readouterr().out
    assert (
        subprocess.run(
            ["git", "tag"], cwd=remote, capture_output=True, text=True
        ).stdout.strip()
        == ""
    )

    assert run(repo, "publish", "--yes") == 0
    assert subprocess.run(
        ["git", "tag"], cwd=remote, capture_output=True, text=True
    ).stdout.split() == ["v0.1.0"]
    assert (
        "master"
        in subprocess.run(
            ["git", "branch"], cwd=remote, capture_output=True, text=True
        ).stdout
    )


def test_publish_without_a_tag_fails(repo):
    assert run(repo, "publish", "--yes") == 1


def test_the_real_repo_has_consistent_versions():
    assert release.current_version(release.ROOT)
