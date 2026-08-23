"""Public workflows must not expose privileged execution to pull requests."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
from typing import Any

import pytest
import yaml


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOWS = REPOSITORY_ROOT / ".github" / "workflows"
TRUST_MODEL_DOC = REPOSITORY_ROOT / "docs" / "ci-trust-model.md"
FULL_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _workflow_files() -> list[pathlib.Path]:
    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert files, f"no workflows found under {WORKFLOWS}"
    return files


def _load(path: pathlib.Path) -> dict[Any, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _triggers(workflow: dict[Any, Any]) -> set[str]:
    raw = workflow.get("on", workflow.get(True))
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {str(item) for item in raw}
    if isinstance(raw, dict):
        return {str(item) for item in raw}
    return set()


def test_workflow_safety_checker_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_workflow_safety.py"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("workflow_path", _workflow_files(), ids=lambda path: path.name)
def test_workflows_use_hosted_runners_and_read_only_defaults(
    workflow_path: pathlib.Path,
) -> None:
    workflow = _load(workflow_path)
    text = workflow_path.read_text(encoding="utf-8")
    assert workflow.get("permissions") == {"contents": "read"}
    assert "pull_request_target" not in _triggers(workflow)
    assert "self-hosted" not in text.lower()
    assert "agentcheck-local" not in text.lower()
    for job_name, job in (workflow.get("jobs") or {}).items():
        runner = job.get("runs-on")
        assert runner == "ubuntu-latest", (
            f"{workflow_path.name}:{job_name} uses unexpected runner {runner!r}"
        )


@pytest.mark.parametrize("workflow_path", _workflow_files(), ids=lambda path: path.name)
def test_pull_request_workflows_have_no_secrets_or_environments(
    workflow_path: pathlib.Path,
) -> None:
    workflow = _load(workflow_path)
    text = workflow_path.read_text(encoding="utf-8")
    assert re.search(r"\$\{\{\s*secrets\.", text) is None
    for job_name, job in (workflow.get("jobs") or {}).items():
        is_publish = workflow_path.name == "release.yml" and job_name == "publish"
        if is_publish:
            assert job.get("permissions") == {"id-token": "write"}
            assert (job.get("environment") or {}).get("name") == "pypi"
            continue
        assert "environment" not in job
        permissions = job.get("permissions")
        if permissions is not None:
            assert isinstance(permissions, dict)
            assert all(value in {"read", "none"} for value in permissions.values())


@pytest.mark.parametrize("workflow_path", _workflow_files(), ids=lambda path: path.name)
def test_external_actions_are_pinned_to_full_commit_shas(
    workflow_path: pathlib.Path,
) -> None:
    workflow = _load(workflow_path)
    for job_name, job in (workflow.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            uses = step.get("uses") if isinstance(step, dict) else None
            if not isinstance(uses, str) or uses.startswith("./"):
                continue
            _, separator, ref = uses.rpartition("@")
            assert separator and FULL_COMMIT_SHA.fullmatch(ref), (
                f"{workflow_path.name}:{job_name} action {uses!r} is mutable"
            )
            if uses.startswith("actions/checkout@"):
                options = step.get("with") or {}
                assert options.get("persist-credentials") is False


def test_ci_scope_is_fail_closed_and_skips_only_docs_and_media() -> None:
    workflow = _load(WORKFLOWS / "ci.yml")
    jobs = workflow["jobs"]
    scope = jobs["scope"]
    assert scope["outputs"] == {
        "expensive": "${{ steps.classify.outputs.expensive }}"
    }
    assert jobs["tests"]["needs"] == "scope"
    assert jobs["tests"]["if"] == "needs.scope.outputs.expensive == 'true'"
    assert jobs["checks"]["needs"] == "scope"

    classify = next(step for step in scope["steps"] if step.get("id") == "classify")
    script = classify["run"]
    for required in (
        "expensive=true",
        "git cat-file -e",
        "git diff --no-renames --name-only -z",
        "*.md|docs/assets/*",
        "GITHUB_OUTPUT",
    ):
        assert required in script

    checks_text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert "needs.scope.outputs.expensive == 'false'" in checks_text
    assert "Documentation consistency" in checks_text


def test_public_trust_model_is_discoverable() -> None:
    assert TRUST_MODEL_DOC.is_file()
    text = TRUST_MODEL_DOC.read_text(encoding="utf-8").lower()
    for required in (
        "github-hosted",
        "fork",
        "self-hosted runner",
        "remove",
        "read-only",
        "no secrets",
        "documentation-only",
    ):
        assert required in text, f"CI trust model does not mention {required!r}"


def test_full_pytest_commands_keep_two_workers() -> None:
    ci = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert "tests -q -n 2" in ci
    assert '"${compat[@]}" -q -n 2' in ci
