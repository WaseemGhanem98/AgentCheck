from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from agentcheck.domain import (
    AgentSpec,
    CanonicalRun,
    CaseEvaluation,
    Finding,
    Scenario,
    Verdict,
)


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"
RUN_ID = "phase1-offline-e2e"
EXPECTED_FAILURES = {
    "delete_without_confirmation",
    "ambiguous_account_selection",
    "destructive_ambiguous_timeout",
    "fabricated_email_success",
    "duplicate_email_update",
}
EXPECTED_ARTIFACTS = {
    "agent-spec.json",
    "evaluations.jsonl",
    "findings.json",
    "invalid-scenarios.json",
    "report.html",
    "runs.jsonl",
    "suite.json",
    "summary.json",
}


def _offline_environment() -> dict[str, str]:
    """Keep the CLI usable while withholding provider credentials and user secrets."""

    inherited = ("PATH", "LANG", "LC_ALL", "TZ", "TMPDIR", "SSL_CERT_FILE")
    environment = {name: os.environ[name] for name in inherited if name in os.environ}
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _run_cli(*arguments: str, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "agentcheck", *arguments],
        cwd=REPOSITORY_ROOT,
        env=_offline_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _jsonl(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _instrument_fixture(target: Path) -> tuple[Path, Path]:
    """Add cross-process probes to a temporary copy of the bundled fixture."""

    source_path = target / "agent.py"
    source = source_path.read_text(encoding="utf-8")
    future_import = "from __future__ import annotations\n"
    process_probe = """

with open(__file__ + ".agentcheck-worker-pids", "a", encoding="utf-8") as _pid_probe:
    _pid_probe.write(f"{__import__('os').getpid()}\\n")
"""
    assert source.count(future_import) == 1
    source = source.replace(future_import, future_import + process_probe, 1)

    original_tripwire = "    ORIGINAL_TOOL_CALLS.append((tool_name, arguments))\n"
    persistent_tripwire = """    with open(
        __file__ + ".agentcheck-original-tool-invoked", "a", encoding="utf-8"
    ) as _tool_probe:
        _tool_probe.write(f"{tool_name}\\n")
"""
    assert source.count(original_tripwire) == 1
    source = source.replace(
        original_tripwire,
        persistent_tripwire + original_tripwire,
        1,
    )
    source_path.write_text(source, encoding="utf-8")
    return (
        Path(f"{source_path}.agentcheck-worker-pids"),
        Path(f"{source_path}.agentcheck-original-tool-invoked"),
    )


def test_offline_cli_initializes_a_target_before_inspecting_it(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    initialization = _run_cli("init", str(empty), timeout=45)

    assert initialization.returncode == 0, initialization.stderr
    assert initialization.stderr == ""
    assert "AgentCheck configuration written." in initialization.stdout
    assert [path.name for path in empty.iterdir()] == ["agentcheck.json"]

    uninitialized = _run_cli("inspect", str(empty), timeout=45)

    assert uninitialized.returncode == 2
    assert "AgentCheck error: entrypoint source does not exist" in uninitialized.stderr
    assert "Traceback" not in uninitialized.stderr

    target = tmp_path / "account_agent"
    shutil.copytree(
        EXAMPLE,
        target,
        ignore=shutil.ignore_patterns(".agentcheck", "__pycache__"),
    )
    (target / "agentcheck.json").unlink()

    regenerated = _run_cli("init", str(target), timeout=45)

    assert regenerated.returncode == 0, regenerated.stderr
    assert "does not exist yet" not in regenerated.stdout
    assert json.loads((target / "agentcheck.json").read_text(encoding="utf-8")) == json.loads(
        (EXAMPLE / "agentcheck.json").read_text(encoding="utf-8")
    )

    refused = _run_cli("init", str(target), timeout=45)

    assert refused.returncode == 2
    assert "already exists" in refused.stderr

    inspection = _run_cli("inspect", str(target), timeout=45)

    assert inspection.returncode == 0, inspection.stderr
    assert "Agent: Account Support Agent" in inspection.stdout


def test_offline_cli_runs_complete_phase1_flow_with_intercepted_tools(
    tmp_path: Path,
) -> None:
    target = tmp_path / "account_agent"
    shutil.copytree(
        EXAMPLE,
        target,
        ignore=shutil.ignore_patterns(".agentcheck", "__pycache__"),
    )
    process_probe, original_tool_probe = _instrument_fixture(target)

    inspection = _run_cli("inspect", str(target), "--json", timeout=45)

    assert inspection.returncode == 0, inspection.stderr
    assert inspection.stderr == ""
    inspected_spec = AgentSpec.model_validate_json(inspection.stdout)
    assert inspected_spec.identity.name.value == "Account Support Agent"
    assert [item.value.name for item in inspected_spec.tools.items] == [
        "lookup_account",
        "update_email",
        "cancel_subscription",
        "delete_account",
    ]

    execution = _run_cli(
        "test",
        str(target),
        "--seed",
        "1729",
        "--run-id",
        RUN_ID,
        timeout=180,
    )

    assert execution.returncode == 1, execution.stderr
    assert execution.stderr == ""
    assert len(re.findall(r"^PASS\s+", execution.stdout, flags=re.MULTILINE)) == 7
    assert len(re.findall(r"^FAIL\s+", execution.stdout, flags=re.MULTILINE)) == 5
    assert "Observed suite pass rate: 58.3%" in execution.stdout
    assert "Passed:        7" in execution.stdout
    assert "Failed:        5" in execution.stdout
    assert "Inconclusive:  0" in execution.stdout
    assert "Infra errors:  0" in execution.stdout

    run_root = target / ".agentcheck" / "runs" / RUN_ID
    assert {path.name for path in run_root.iterdir()} == EXPECTED_ARTIFACTS
    if os.name == "posix":
        assert stat.S_IMODE(run_root.stat().st_mode) == 0o700
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o600 for path in run_root.iterdir()
        )

    artifact_spec = AgentSpec.model_validate_json(
        (run_root / "agent-spec.json").read_text(encoding="utf-8")
    )
    assert artifact_spec.spec_id == inspected_spec.spec_id

    suite_payload = json.loads((run_root / "suite.json").read_text(encoding="utf-8"))
    scenarios = tuple(
        Scenario.model_validate_json(json.dumps(item))
        for item in suite_payload["scenarios"]
    )
    assert suite_payload["schema_version"] == "agentcheck.suite.v1"
    assert suite_payload["seed"] == 1729
    assert len(scenarios) == 12
    assert len({scenario.fingerprint for scenario in scenarios}) == 12
    assert {scenario.generation_seed for scenario in scenarios} == {1729}

    invalid_payload = json.loads(
        (run_root / "invalid-scenarios.json").read_text(encoding="utf-8")
    )
    assert invalid_payload["items"] == []

    runs = tuple(
        CanonicalRun.model_validate_json(line)
        for line in _jsonl(run_root / "runs.jsonl")
    )
    evaluations = tuple(
        CaseEvaluation.model_validate_json(line)
        for line in _jsonl(run_root / "evaluations.jsonl")
    )
    assert len(runs) == len(evaluations) == len(scenarios) == 12
    assert {run.scenario_id for run in runs} == {
        scenario.scenario_id for scenario in scenarios
    }
    assert [run.run_id for run in runs] == [
        f"{RUN_ID}-case-{index:03d}" for index in range(1, 13)
    ]
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    assert all(
        run.initial_world_state == scenario_by_id[run.scenario_id].initial_world_state
        for run in runs
    )

    counts = Counter(evaluation.verdict for evaluation in evaluations)
    assert counts == Counter({Verdict.PASS: 7, Verdict.FAIL: 5})
    assert {
        evaluation.scenario_id
        for evaluation in evaluations
        if evaluation.verdict == Verdict.FAIL
    } == EXPECTED_FAILURES
    assert all(evaluation.infrastructure_error is None for evaluation in evaluations)
    assert all(
        any(
            assertion.result == Verdict.FAIL
            and (
                assertion.supporting_evidence_ids
                or assertion.contradicting_evidence_ids
            )
            for assertion in evaluation.assertions
        )
        for evaluation in evaluations
        if evaluation.verdict == Verdict.FAIL
    )

    tool_outcomes = [outcome for run in runs for outcome in run.tool_outcomes]
    assert tool_outcomes
    assert all(outcome.metadata.get("simulated") is True for outcome in tool_outcomes)
    assert all(outcome.metadata.get("fixture_id") for outcome in tool_outcomes)
    assert all(outcome.metadata.get("gateway_outcome_id") for outcome in tool_outcomes)
    assert not original_tool_probe.exists()

    worker_pids = [
        int(value) for value in process_probe.read_text(encoding="utf-8").splitlines()
    ]
    # One worker for the explicit inspection, one suite inspection worker, and
    # one fresh worker for each of the twelve scenario runs.
    assert len(worker_pids) == 14
    assert len(set(worker_pids)) == 14

    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    assert summary == {
        "schema_version": "agentcheck.summary.v1",
        "run_id": RUN_ID,
        "target": str(target),
        "git_revision": None,
        "suite_size": 12,
        "invalid_scenarios": 0,
        "observed_suite_pass_rate": 7 / 12,
        "counts": {
            "PASS": 7,
            "FAIL": 5,
            "INCONCLUSIVE": 0,
            "INFRA_ERROR": 0,
        },
        "finding_count": 5,
        "seed": 1729,
    }
    findings = tuple(
        Finding.model_validate_json(json.dumps(item))
        for item in json.loads((run_root / "findings.json").read_text(encoding="utf-8"))
    )
    assert len(findings) == 5
    assert {
        scenario_id for item in findings for scenario_id in item.affected_scenario_ids
    } == (EXPECTED_FAILURES)

    report = (run_root / "report.html").read_text(encoding="utf-8")
    HTMLParser().feed(report)
    assert "Content-Security-Policy" in report
    assert "<script" not in report.casefold()
    assert "https://" not in report
    assert "<span>Passed</span><strong>7</strong>" in report
    assert "<span>Failed</span><strong>5</strong>" in report
    assert "<span>Inconclusive</span><strong>0</strong>" in report
    assert "<span>Infra errors</span><strong>0</strong>" in report
    assert "Raw system instructions are hidden by default." in report
    assert all(scenario.scenario_id in report for scenario in scenarios)
