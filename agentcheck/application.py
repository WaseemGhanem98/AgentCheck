"""Phase 1 application service connecting inspection, execution, and reporting."""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

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
from agentcheck.errors import ConfigurationError, ScenarioValidationError
from agentcheck.generate import lint_suite
from agentcheck.generate.suite import (
    FrozenSuite,
    build_frozen_suite,
    built_in_suite,
    configured_frozen_suite,
    resolve_suite_destination,
    write_frozen_suite,
)
from agentcheck.policies import (
    apply_policy_packs_to_many,
    attach_declared_policies,
    resolve_policy_packs,
)
from agentcheck.privacy import redact_log_text
from agentcheck.report import render_report
from agentcheck.store import open_evaluation_store, stored_run_from_execution
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
class SuiteGeneration:
    target_root: Path
    config: AgentCheckConfig
    spec: AgentSpec
    suite: FrozenSuite
    suite_path: Path


@dataclass(frozen=True, slots=True)
class SuiteExecution:
    target_root: Path
    config: AgentCheckConfig
    run_id: str
    seed: int
    git_revision: str | None
    spec: AgentSpec
    frozen_suite: FrozenSuite | None
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
    inspection = inspect_in_subprocess(
        root,
        config,
        timeout_seconds=timeout_seconds,
    )
    if inspection.value is not None:
        packs = resolve_policy_packs(root, config)
        inspection = ProcessResult(
            value=attach_declared_policies(inspection.value, packs),
            infrastructure_error=None,
            stdout=inspection.stdout,
            stderr=inspection.stderr,
            returncode=inspection.returncode,
            timed_out=inspection.timed_out,
            worker_pid=inspection.worker_pid,
        )
    return (
        root,
        config,
        inspection,
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


def _effective_seed(config: AgentCheckConfig, seed: int | None) -> int:
    actual_seed = config.seed if seed is None else seed
    if actual_seed < 0 or actual_seed > 2**63 - 1:
        raise ValueError("seed must be between 0 and 2^63 - 1")
    return actual_seed


def _suite(config: AgentCheckConfig, seed: int | None) -> tuple[Scenario, ...]:
    return built_in_suite(config, _effective_seed(config, seed))


def generate_suite(
    target: str | Path,
    *,
    seed: int | None = None,
    out: str | None = None,
    force: bool = False,
    include_mutations: bool = False,
    max_mutations: int | None = None,
    policy_packs: Sequence[str] | None = None,
    timeout_seconds: float = 30.0,
) -> SuiteGeneration:
    """Inspect a target, freeze its deterministic suite, and persist a reviewable file."""

    root, config = load_config(target)
    actual_seed = _effective_seed(config, seed)
    destination = resolve_suite_destination(root, config, out)
    if not force and (destination.exists() or destination.is_symlink()):
        raise ConfigurationError(
            f"{destination.name} already exists at {destination}; "
            "re-run with --force to replace it"
        )
    inspection = inspect_in_subprocess(root, config, timeout_seconds=timeout_seconds)
    packs = resolve_policy_packs(root, config, policy_packs)
    spec = attach_declared_policies(inspection.require_value(), packs)
    suite = build_frozen_suite(
        spec,
        config,
        seed=actual_seed,
        include_mutations=include_mutations,
        max_mutations=max_mutations,
        policy_packs=packs,
    )
    write_frozen_suite(destination, suite, force=force)
    return SuiteGeneration(
        target_root=root,
        config=config,
        spec=spec,
        suite=suite,
        suite_path=destination,
    )


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
    persist_store: bool = True,
) -> SuiteExecution:
    """Execute the configured suite and persist an immutable report."""

    root, config = load_config(target)
    # A frozen suite is untrusted input: parse and integrity-check it before the
    # target is imported, so a malformed file never reaches model or tool code.
    configured = configured_frozen_suite(root, config)
    suite_run_id = run_id or new_run_id()
    # Freeze repository identity before importing or executing target code.
    revision = _git_revision(root)
    inspection = inspect_in_subprocess(root, config)
    packs = resolve_policy_packs(root, config)
    spec = attach_declared_policies(inspection.require_value(), packs)

    frozen: FrozenSuite | None = None
    if configured is None:
        effective_seed = _effective_seed(config, seed)
        candidates = apply_policy_packs_to_many(
            _suite(config, seed), packs, declared=True
        )
    else:
        suite_path, frozen = configured
        if frozen.spec_id != spec.spec_id:
            raise ConfigurationError(
                f"frozen suite {suite_path.name} was generated for target "
                f"{frozen.spec_id}, but this target inspects as {spec.spec_id}; "
                "re-run agentcheck generate"
            )
        if seed is not None and seed != frozen.seed:
            raise ConfigurationError(
                f"frozen suite {suite_path.name} was generated with seed "
                f"{frozen.seed}; re-run agentcheck generate to change it"
            )
        effective_seed = frozen.seed
        candidates = frozen.scenarios
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
            "seed": effective_seed,
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
            "seed": effective_seed,
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
        seed=effective_seed,
        frozen_suite=frozen,
    )
    report_path = artifacts.write_text("report.html", report)
    execution = SuiteExecution(
        target_root=root,
        config=config,
        run_id=suite_run_id,
        seed=effective_seed,
        git_revision=revision,
        spec=spec,
        frozen_suite=frozen,
        scenarios=valid_scenarios,
        invalid_scenarios=tuple(invalid),
        runs=runs,
        evaluations=evaluations,
        findings=findings,
        artifact_directory=artifacts.root,
        report_path=report_path,
    )
    if persist_store:
        _persist_execution(execution)
    return execution


def _persist_execution(execution: SuiteExecution) -> None:
    """Index a completed run. Storage bugs must not become agent verdicts."""

    try:
        store = open_evaluation_store(execution.target_root, execution.config)
        store.record_run(stored_run_from_execution(execution))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(
            "AgentCheck warning: evaluation store failed: "
            + redact_log_text(str(exc)),
            file=sys.stderr,
        )


__all__ = [
    "InvalidScenario",
    "ProgressCallback",
    "SuiteExecution",
    "SuiteGeneration",
    "execute_suite",
    "generate_suite",
    "inspect_target",
]
