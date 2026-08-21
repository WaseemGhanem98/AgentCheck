"""CI must not become a way to run other people's code on someone's computer.

While this repository is private, normal CI runs on a self-hosted runner --
a maintainer's own workstation -- because every branch here is written by the
owner and GitHub-hosted minutes are billed against the account.

That arrangement is safe only because of the `if:` guard on each self-hosted
job, which admits pushes and same-repository pull requests and nothing else.
A job-level `if` is not inherited, so one unguarded job is enough to undo it.

These tests check the guard rather than trusting review, and they check it by
evaluating the expression against synthetic trigger contexts, because the
mistakes worth catching are logical: an `||` that should be `&&`, a deny-list
that admits a trigger nobody thought about.

`test_publication_checklist_is_discoverable` is the one that matters most and
tests the least code. The guard makes a fork's pull request *skip*, which is
safe but leaves external contributors with no CI -- so this configuration has
to be revisited before the repository goes public. That is precisely the kind
of decision that gets forgotten, so the reminder is asserted to exist.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOWS = REPOSITORY_ROOT / ".github" / "workflows"
TRUST_MODEL_DOC = REPOSITORY_ROOT / "docs" / "ci-trust-model.md"


def _workflow_files() -> list[pathlib.Path]:
    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert files, f"no workflows found under {WORKFLOWS}"
    return files


def test_no_self_hosted_job_is_reachable_from_an_untrusted_context() -> None:
    """Delegates to the checker CI runs, so both see the same verdict."""

    result = subprocess.run(
        [sys.executable, "scripts/check_workflow_safety.py"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("workflow", _workflow_files(), ids=lambda path: path.name)
def test_no_workflow_uses_pull_request_target(workflow: pathlib.Path) -> None:
    """The trigger that runs a fork's code with the base repository's secrets."""

    offending = [
        number
        for number, line in enumerate(workflow.read_text().splitlines(), start=1)
        if "pull_request_target" in line and not line.lstrip().startswith("#")
    ]
    assert not offending, (
        f"{workflow.name} uses pull_request_target at line(s) {offending}. "
        "It runs with the base repository's token and secrets, so it must never "
        "be combined with checking out a fork's code -- and never at all on a "
        "self-hosted runner."
    )


@pytest.mark.parametrize("workflow", _workflow_files(), ids=lambda path: path.name)
def test_workflows_declare_least_privilege_token(workflow: pathlib.Path) -> None:
    import re

    text = workflow.read_text()
    assert re.search(r"^permissions:", text, re.MULTILINE), (
        f"{workflow.name} does not declare a top-level `permissions:` block. "
        "Without one it inherits the repository default, which may be write."
    )


def test_publication_checklist_is_discoverable() -> None:
    """A self-hosted runner plus a public repository is the mistake to prevent."""

    assert TRUST_MODEL_DOC.exists(), (
        f"{TRUST_MODEL_DOC.relative_to(REPOSITORY_ROOT)} is missing. It records why "
        "CI is self-hosted while this repository is private and what has to change "
        "before it goes public; without it that decision is only in someone's head."
    )
    doc = TRUST_MODEL_DOC.read_text()
    for required in ("private", "public", "self-hosted", "fork"):
        assert required in doc.lower(), f"trust-model doc never mentions {required!r}"

    # The workflow itself must carry the warning too. Someone changing runner
    # labels is reading the YAML, not the docs directory.
    ci = (WORKFLOWS / "ci.yml").read_text()
    assert "BEFORE MAKING THIS REPOSITORY PUBLIC" in ci, (
        "ci.yml must carry the pre-publication warning where a maintainer "
        "editing runner labels will actually see it"
    )
