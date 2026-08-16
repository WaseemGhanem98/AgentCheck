"""Phase 1 application service connecting inspection, execution, and reporting."""

from __future__ import annotations

import subprocess
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agentcheck.analyze import analyze_failures
from agentcheck.artifacts import ArtifactStore, new_run_id
from agentcheck.config import AgentCheckConfig, load_config
from agentcheck.domain import (
    AgentSpec,
    CanonicalRun,
    CaseEvaluation,
    Finding,
    InfrastructureError,
    Scenario,
    Verdict,
)
from agentcheck.evaluate import evaluate_run, infrastructure_evaluation
from agentcheck.errors import ScenarioValidationError
from agentcheck.generate import build_account_support_suite, lint_suite
from agentcheck.report import render_report
from agentcheck.runner.orchestrator import (
    ProcessResult,
    inspect_in_subprocess,
    run_scenario_in_subprocess,
)


ProgressCallback = Callable[[int, int, Scenario, CaseEvaluation], None]


@dataclass(frozen=True, slots=True)
class InvalidScenario:
    scenario: Scenario
    issues: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class SuiteExecution:
    target_root: Path
    config: AgentCheckConfig
    run_id: str
    git_revision: str | None
    spec: AgentSpec
    scenarios: tuple[Scenario, ...]
    invalid_scenarios: tuple[InvalidScenario, ...]
    runs: tuple[CanonicalRun, ...]
    evaluations: tuple[CaseEvaluation, ...]
    findings: tuple[Finding, ...]
    artifact_directory: Path
    report_path: Path

    @property
    def counts(self) -> Counter[Verdict]:
        return Counter(item.verdict for item in self.evaluations)

    @property
    def observed_pass_rate(self) -> float | None:
        if not self.evaluations:
            return None
        return self.counts[Verdict.PASS] / len(self.evaluations)


def inspect_target(
    target: str | Path,
    *,
    timeout_seconds: float = 30.0,
) -> tuple[Path, AgentCheckConfig, ProcessResult[AgentSpec]]:
    root, config = load_config(target)
    return (
        root,
        config,
        inspect_in_subprocess(
            root,
            config,
            timeout_seconds=timeout_seconds,
        ),
    )


def _git_revision(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def _suite(config: AgentCheckConfig, seed: int | None) -> tuple[Scenario, ...]:
    actual_seed = config.seed if seed is None else seed
    if actual_seed < 0 or actual_seed > 2**63 - 1:
        raise ValueError("seed must be between 0 and 2^63 - 1")
    if config.suite == "account_support_v1":
        return build_account_support_suite(seed=actual_seed)
    raise ValueError(f"unsupported Phase 1 suite: {config.suite}")


def _process_failure_evaluation(
    scenario: Scenario,
    result: ProcessResult[CanonicalRun],
    case_run_id: str,
) -> CaseEvaluation:
    error = result.infrastructure_error or InfrastructureError(
        code="worker_error",
        message="The child process returned no canonical run.",
        phase="execution",
        retryable=False,
    )
    return infrastructure_evaluation(
        scenario,
        code=error.code,
        message=error.message,
        phase=error.phase,
        run_id=case_run_id,
    )


def execute_suite(
    target: str | Path,
    *,
    seed: int | None = None,
    run_id: str | None = None,
    progress: ProgressCallback | None = None,
) -> SuiteExecution:
    """Execute the deterministic Phase 1 suite and persist an immutable report."""

    root, config = load_config(target)
    suite_run_id = run_id or new_run_id()
    # Freeze repository identity before importing or executing target code.
    revision = _git_revision(root)
    inspection = inspect_in_subprocess(root, config)
    spec = inspection.require_value()

    candidates = _suite(config, seed)
    valid: list[Scenario] = []
    invalid: list[InvalidScenario] = []
    for scenario, issues in lint_suite(candidates, spec):
        if issues:
            invalid.append(
                InvalidScenario(
                    scenario=scenario,
                    issues=tuple(
                        {
                            "code": issue.code,
                            "message": issue.message,
                            "severity": issue.severity,
                        }
                        for issue in issues
                    ),
                )
            )
        else:
            valid.append(scenario)

    if not valid:
        raise ScenarioValidationError(
            "No valid scenarios remain after linting; no agent verdict was produced."
        )

    indexed_results: dict[int, tuple[CanonicalRun | None, CaseEvaluation]] = {}
    with ThreadPoolExecutor(
        max_workers=min(config.max_concurrency, max(1, len(valid))),
        thread_name_prefix="agentcheck",
    ) as pool:
        futures: dict[
            Future[ProcessResult[CanonicalRun]],
            tuple[int, Scenario, str],
        ] = {}
        for index, scenario in enumerate(valid):
            case_run_id = f"{suite_run_id}-case-{index + 1:03d}"
            future = pool.submit(
                run_scenario_in_subprocess,
                root,
                config,
                scenario,
                case_run_id,
                expected_target_id=spec.spec_id,
            )
            futures[future] = (index, scenario, case_run_id)

        completed_count = 0
        for future in as_completed(futures):
            index, scenario, case_run_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # defensive: executor/platform failure
                evaluation = infrastructure_evaluation(
                    scenario,
                    code="orchestrator_error",
                    message=str(exc)[:4_000],
                    phase="orchestration",
                    run_id=case_run_id,
                )
                case_run = None
            else:
                case_run = result.value
                evaluation = (
                    evaluate_run(scenario, case_run)
                    if case_run is not None
                    else _process_failure_evaluation(scenario, result, case_run_id)
                )
            indexed_results[index] = (case_run, evaluation)
            completed_count += 1
            if progress is not None:
                progress(completed_count, len(valid), scenario, evaluation)

    ordered = tuple(indexed_results[index] for index in range(len(valid)))
    runs = tuple(case_run for case_run, _ in ordered if case_run is not None)
    evaluations = tuple(evaluation for _, evaluation in ordered)
    valid_scenarios = tuple(valid)
    findings = analyze_failures(valid_scenarios, evaluations)
    artifacts = ArtifactStore(root, config.artifacts_directory, suite_run_id)
    artifacts.write_json("agent-spec.json", spec)
    artifacts.write_json(
        "suite.json",
        {
            "schema_version": "agentcheck.suite.v1",
            "run_id": suite_run_id,
            "seed": config.seed if seed is None else seed,
            "scenarios": valid_scenarios,
        },
    )
    artifacts.write_json(
        "invalid-scenarios.json",
        {
            "schema_version": "agentcheck.invalid_scenarios.v1",
            "items": [
                {"scenario": item.scenario, "issues": item.issues} for item in invalid
            ],
        },
    )
    artifacts.write_jsonl("runs.jsonl", runs)
    artifacts.write_jsonl("evaluations.jsonl", evaluations)
    artifacts.write_json("findings.json", findings)

    counts = Counter(item.verdict for item in evaluations)
    pass_rate = counts[Verdict.PASS] / len(evaluations) if evaluations else None
    artifacts.write_json(
        "summary.json",
        {
            "schema_version": "agentcheck.summary.v1",
            "run_id": suite_run_id,
            "target": str(root),
            "git_revision": revision,
            "suite_size": len(valid_scenarios),
            "invalid_scenarios": len(invalid),
            "observed_suite_pass_rate": pass_rate,
            "counts": {verdict.value: counts[verdict] for verdict in Verdict},
            "finding_count": len(findings),
            "seed": config.seed if seed is None else seed,
        },
    )
    report = render_report(
        run_id=suite_run_id,
        target=str(root),
        git_revision=revision,
        spec=spec,
        scenarios=valid_scenarios,
        runs=runs,
        evaluations=evaluations,
        findings=findings,
        include_instructions=config.include_instructions_in_report,
    )
    report_path = artifacts.write_text("report.html", report)
    return SuiteExecution(
        target_root=root,
        config=config,
        run_id=suite_run_id,
        git_revision=revision,
        spec=spec,
        scenarios=valid_scenarios,
        invalid_scenarios=tuple(invalid),
        runs=runs,
        evaluations=evaluations,
        findings=findings,
        artifact_directory=artifacts.root,
        report_path=report_path,
    )


__all__ = [
    "InvalidScenario",
    "ProgressCallback",
    "SuiteExecution",
    "execute_suite",
    "inspect_target",
]
