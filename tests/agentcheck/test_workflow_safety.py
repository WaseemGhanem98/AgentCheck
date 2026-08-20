"""A public repository must not run contributor code on a trusted machine.

This is the one CI mistake in an open-source project that costs more than a red
build. `pull_request` runs code from a fork. If such a workflow is ever pointed
at a self-hosted runner -- someone's workstation, with their filesystem, their
network, and their credentials -- then opening a pull request is remote code
execution on that machine.

`pull_request_target` is the same hazard by a different route: it runs with the
base repository's token and secrets, so checking out and executing the fork's
code under it hands those over.

Neither is something to catch in review. This test reads the workflows.
"""

from __future__ import annotations

import pathlib
import re

import pytest


WORKFLOWS = pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _workflow_files() -> list[pathlib.Path]:
    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert files, f"no workflows found under {WORKFLOWS}"
    return files


@pytest.mark.parametrize("workflow", _workflow_files(), ids=lambda path: path.name)
def test_no_workflow_uses_a_self_hosted_runner(workflow: pathlib.Path) -> None:
    text = workflow.read_text(encoding="utf-8")
    offending = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(text.splitlines(), start=1)
        # Match the runner label wherever it appears in a runs-on value, including
        # the list form `runs-on: [self-hosted, Linux, X64]`.
        if re.search(r"self-hosted", line) and not line.lstrip().startswith("#")
    ]
    assert not offending, (
        f"{workflow.name} references a self-hosted runner: {offending}. "
        "Public pull requests execute untrusted code; they must run on "
        "GitHub-hosted ephemeral runners only."
    )


@pytest.mark.parametrize("workflow", _workflow_files(), ids=lambda path: path.name)
def test_no_workflow_uses_pull_request_target(workflow: pathlib.Path) -> None:
    text = workflow.read_text(encoding="utf-8")
    offending = [
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if "pull_request_target" in line and not line.lstrip().startswith("#")
    ]
    assert not offending, (
        f"{workflow.name} uses pull_request_target at line(s) {offending}. "
        "That trigger runs with the base repository's token and secrets, so it "
        "must never be combined with checking out a fork's code."
    )


@pytest.mark.parametrize("workflow", _workflow_files(), ids=lambda path: path.name)
def test_workflows_declare_least_privilege_token(workflow: pathlib.Path) -> None:
    text = workflow.read_text(encoding="utf-8")
    assert re.search(r"^permissions:", text, re.MULTILINE), (
        f"{workflow.name} does not declare a top-level `permissions:` block. "
        "Without one the workflow inherits the repository default, which may be "
        "write. State the minimum explicitly."
    )
