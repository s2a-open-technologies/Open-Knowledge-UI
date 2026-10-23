import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = sorted(
    (Path(__file__).resolve().parent.parent / ".github" / "workflows").glob("*.yml")
)
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def load(path):
    return yaml.safe_load(path.read_text())


def uses_refs(workflow):
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if "uses" in step:
                yield step["uses"]


def test_workflows_exist():
    assert {p.name for p in WORKFLOWS} >= {"ci.yml", "release.yml"}


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_full_commit_sha(path):
    for ref in uses_refs(load(path)):
        name, _, version = ref.partition("@")
        assert FULL_SHA.match(version), (
            f"{name} ist nicht auf eine Commit-SHA gepinnt: @{version}"
        )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_checkouts_do_not_leave_credentials_behind(path):
    for job in load(path)["jobs"].values():
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step.get("with", {}).get("persist-credentials") is False


def test_only_the_final_release_job_may_write():
    wf = load(next(p for p in WORKFLOWS if p.name == "release.yml"))
    assert wf["permissions"] == {"contents": "read"}
    writers = [
        n
        for n, j in wf["jobs"].items()
        if j.get("permissions", {}).get("contents") == "write"
    ]
    assert writers == ["release"]
    # Es werden keine Fremd-Actions im Job mit Schreibrechten ausgefuehrt (nur eigener Code und gh).
    external = [
        u
        for u in (s.get("uses", "") for s in wf["jobs"]["release"]["steps"])
        if u and not u.startswith("actions/")
    ]
    assert external == []
    assert wf["jobs"]["release"]["needs"] == "verify"


def test_ci_has_read_only_permissions():
    wf = load(next(p for p in WORKFLOWS if p.name == "ci.yml"))
    assert wf["permissions"] == {"contents": "read"}


def test_dependabot_keeps_the_pins_current():
    cfg = yaml.safe_load((WORKFLOWS[0].parent.parent / "dependabot.yml").read_text())
    assert any(u["package-ecosystem"] == "github-actions" for u in cfg["updates"])
