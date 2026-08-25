"""One required status check that tells the truth about the whole workflow.

The expensive matrix is skipped for documentation-only pull requests, which is
deliberate. Naming those matrix jobs as branch-protection requirements is what
turned that optimisation into a deadlock: a skipped job never reports, so the
pull request waits for a status that will never arrive.

The fix is a single always-present gate. Its decision lives in
``scripts/check_required_ci.py`` rather than inline in YAML, because "a skipped
matrix is fine here but a failed one is not" is exactly the kind of rule that
should be executable and tested rather than reviewed by eye.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from typing import Any

import pytest
import yaml


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOWS = REPOSITORY_ROOT / ".github" / "workflows"
GATE = REPOSITORY_ROOT / "scripts" / "check_required_ci.py"
GATE_JOB = "required"
GATE_NAME = "Required CI"


def _ci() -> dict[Any, Any]:
    loaded = yaml.safe_load((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _decide(
    *, scope: str, tests: str, checks: str, expensive: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "PATH": "/usr/bin:/bin",
            "SCOPE_RESULT": scope,
            "TESTS_RESULT": tests,
            "CHECKS_RESULT": checks,
            "EXPENSIVE": expensive,
        },
    )


# --- the deadlock this exists to prevent -----------------------------------


def test_docs_only_run_passes_with_a_skipped_matrix() -> None:
    """The whole point: skipped-because-docs must resolve, not hang."""

    result = _decide(scope="success", tests="skipped", checks="success", expensive="false")

    assert result.returncode == 0, result.stdout + result.stderr


def test_code_change_passes_only_when_the_matrix_actually_ran() -> None:
    result = _decide(scope="success", tests="success", checks="success", expensive="true")

    assert result.returncode == 0, result.stdout + result.stderr


def test_a_skipped_matrix_on_a_code_change_is_refused() -> None:
    """Skipping the suite on a code change must never satisfy the gate."""

    result = _decide(scope="success", tests="skipped", checks="success", expensive="true")

    assert result.returncode == 1
    assert "tests" in result.stdout.lower()


# --- a failure is never an acceptable skip ---------------------------------


@pytest.mark.parametrize("outcome", ["failure", "cancelled"])
@pytest.mark.parametrize("expensive", ["true", "false"])
def test_a_failed_or_cancelled_matrix_always_fails_the_gate(
    outcome: str, expensive: str
) -> None:
    """Including on a docs-only run, where the matrix was not even expected."""

    result = _decide(
        scope="success", tests=outcome, checks="success", expensive=expensive
    )

    assert result.returncode == 1, result.stdout


@pytest.mark.parametrize("outcome", ["failure", "cancelled", "skipped"])
@pytest.mark.parametrize("expensive", ["true", "false"])
def test_quality_checks_must_succeed_in_every_scope(
    outcome: str, expensive: str
) -> None:
    """The quality job runs for docs-only changes too, so it may never be absent."""

    result = _decide(
        scope="success", tests="success", checks=outcome, expensive=expensive
    )

    assert result.returncode == 1, result.stdout
    assert "checks" in result.stdout.lower()


@pytest.mark.parametrize("outcome", ["failure", "cancelled", "skipped"])
def test_a_broken_classifier_fails_the_gate(outcome: str) -> None:
    result = _decide(
        scope=outcome, tests="success", checks="success", expensive="true"
    )

    assert result.returncode == 1, result.stdout
    assert "scope" in result.stdout.lower()


def test_an_unreadable_scope_output_fails_closed() -> None:
    """An unset or unexpected classification is treated as a code change."""

    for expensive in ("", "maybe", "TRUE"):
        result = _decide(
            scope="success", tests="skipped", checks="success", expensive=expensive
        )
        assert result.returncode == 1, f"{expensive!r} should fail closed"


# --- the workflow actually wires it up -------------------------------------


def test_the_gate_job_is_always_present_and_depends_on_the_others() -> None:
    jobs = _ci()["jobs"]
    assert GATE_JOB in jobs, "ci.yml must define the required-status gate"
    gate = jobs[GATE_JOB]

    assert gate["name"] == GATE_NAME
    # always() is what makes it report even when the matrix is skipped.
    assert "always()" in str(gate["if"])
    assert set(gate["needs"]) == {"scope", "tests", "checks"}
    assert gate["runs-on"] == "ubuntu-latest"


def test_the_gate_reads_every_job_result_it_claims_to_cover() -> None:
    gate = _ci()["jobs"][GATE_JOB]
    wired = "\n".join(str(step) for step in gate["steps"])

    for expression in (
        "needs.scope.result",
        "needs.tests.result",
        "needs.checks.result",
        "needs.scope.outputs.expensive",
    ):
        assert expression in wired, f"gate ignores {expression}"
    assert "check_required_ci.py" in wired


def test_the_matrix_still_covers_every_supported_interpreter() -> None:
    """The gate must not become an excuse to shrink real coverage."""

    tests = _ci()["jobs"]["tests"]
    matrix = str(tests["strategy"]["matrix"]["python-version"])

    for version in ("3.10", "3.11", "3.12"):
        assert version in matrix
    assert tests["if"] == "needs.scope.outputs.expensive == 'true'"
