"""Phase 1 application service connecting inspection, execution, and reporting."""

from __future__ import annotations

import os
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
from agentcheck.generate.realization import (
    REALIZATION_CREDENTIAL_ENV,
    OpenAIChatRealizer,
    realization_settings,
    require_realization_consent,
)
from agentcheck.generate.selection import (
    SelectionPlan,
    lineage_coverage_tags,
    select_scenarios,
)
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
from agentcheck.replay import (
    build_replay_manifest,
    entrypoint_digest,
    git_revision,
    load_replay_manifest,
    verify_replay_bindings,
    write_replay_manifest,
)
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
    selection: SelectionPlan | None = None
    replay_manifest_path: Path | None = None

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
    return git_revision(root)


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
    max_cases: int | None = None,
    realize: bool = False,
    realizer: object | None = None,
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
    active_realizer: object | None = None
    if realize:
        require_realization_consent(config, realize=True)
        print(
            "AgentCheck warning: LLM realization will make paid provider calls. "
            "Display text only; fingerprints and verdicts stay deterministic.",
            file=sys.stderr,
        )
        if realizer is None:
            model, _max_calls, max_retries = realization_settings(config)
            active_realizer = OpenAIChatRealizer(
                api_key=os.environ[REALIZATION_CREDENTIAL_ENV],
                model=model,
                max_retries=max_retries,
            )
        else:
            active_realizer = realizer
    suite = build_frozen_suite(
        spec,
        config,
        seed=actual_seed,
        include_mutations=include_mutations,
        max_mutations=max_mutations,
        policy_packs=packs,
        max_cases=max_cases,
        realizer=active_realizer,
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
    select: str | None = None,
) -> SuiteExecution:
    """Execute the configured suite and persist an immutable report."""

    root, config = load_config(target)
    if select not in (None, "coverage"):
        raise ConfigurationError(
            "select must be omitted or set to 'coverage'"
        )
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

    selection_plan: SelectionPlan | None = None if frozen is None else frozen.selection
    if select == "coverage":
        extra_tags: dict[str, Sequence[str]] = {}
        if frozen is not None:
            extra_tags = {
                case.scenario.scenario_id: lineage_coverage_tags(
                    origin=case.lineage.origin.value,
                    tool_name=case.lineage.tool_name,
                    boundary_kind=case.lineage.boundary_kind,
                    mutation_kind=case.lineage.mutation_kind,
                )
                for case in frozen.cases
                if case.scenario.scenario_id in {item.scenario_id for item in valid}
            }
        valid_selected, selection_plan = select_scenarios(
            valid,
            max_cases=config.max_cases,
            spec=spec,
            extra_tags_by_id=extra_tags,
        )
        valid = list(valid_selected)
        if not valid:
            raise ScenarioValidationError(
                "No valid scenarios remain after coverage selection; "
                "no agent verdict was produced."
            )

    return _execute_valid_scenarios(
        root=root,
        config=config,
        spec=spec,
        suite_run_id=suite_run_id,
        effective_seed=effective_seed,
        revision=revision,
        valid=tuple(valid),
        invalid=tuple(invalid),
        frozen=frozen,
        selection_plan=selection_plan,
        progress=progress,
        persist_store=persist_store,
        policy_pack_ids=tuple(pack.pack_id for pack in packs),
    )


def replay_suite(
    target: str | Path,
    manifest_path: str,
    *,
    run_id: str | None = None,
    persist_store: bool = True,
    progress: ProgressCallback | None = None,
) -> SuiteExecution:
    """Re-execute a replay manifest through the existing isolated runtime."""

    root, config = load_config(target)
    # Untrusted manifest is parsed before the target is imported.
    manifest = load_replay_manifest(root, manifest_path)
    suite_run_id = run_id or new_run_id()
    revision = _git_revision(root)
    inspection = inspect_in_subprocess(root, config)
    packs = resolve_policy_packs(root, config)
    spec = attach_declared_policies(inspection.require_value(), packs)
    pack_ids = tuple(pack.pack_id for pack in packs)
    verify_replay_bindings(
        manifest,
        root=root,
        config=config,
        spec=spec,
        policy_pack_ids=pack_ids,
    )
    valid: list[Scenario] = []
    invalid: list[InvalidScenario] = []
    for scenario, issues in lint_suite(manifest.cases, spec):
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
    if invalid or len(valid) != len(manifest.cases):
        raise ConfigurationError(
            "replay manifest contains scenarios that fail lint against this target"
        )
    return _execute_valid_scenarios(
        root=root,
        config=config,
        spec=spec,
        suite_run_id=suite_run_id,
        effective_seed=manifest.seed,
        revision=revision,
        valid=tuple(valid),
        invalid=(),
        frozen=None,
        selection_plan=None,
        progress=progress,
        persist_store=persist_store,
        policy_pack_ids=pack_ids,
    )


def _execute_valid_scenarios(
    *,
    root: Path,
    config: AgentCheckConfig,
    spec: AgentSpec,
    suite_run_id: str,
    effective_seed: int,
    revision: str | None,
    valid: tuple[Scenario, ...],
    invalid: tuple[InvalidScenario, ...],
    frozen: FrozenSuite | None,
    selection_plan: SelectionPlan | None,
    progress: ProgressCallback | None,
    persist_store: bool,
    policy_pack_ids: tuple[str, ...],
) -> SuiteExecution:
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
    summary: dict[str, object] = {
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
    }
    if selection_plan is not None:
        summary["excluded_by_selection"] = len(selection_plan.excluded_ids)
        summary["selection"] = selection_plan.model_dump(mode="json")
        summary["coverage"] = selection_plan.coverage.model_dump(mode="json")
    artifacts.write_json(
        "summary.json",
        summary,
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
        selection_plan=selection_plan,
    )
    report_path = artifacts.write_text("report.html", report)
    replay_path = _emit_replay_manifest(
        root=root,
        config=config,
        spec=spec,
        run_id=suite_run_id,
        seed=effective_seed,
        scenarios=valid_scenarios,
        revision=revision,
        policy_pack_ids=policy_pack_ids,
    )
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
        selection=selection_plan,
        replay_manifest_path=replay_path,
    )
    if persist_store:
        _persist_execution(execution)
    return execution


def _emit_replay_manifest(
    *,
    root: Path,
    config: AgentCheckConfig,
    spec: AgentSpec,
    run_id: str,
    seed: int,
    scenarios: tuple[Scenario, ...],
    revision: str | None,
    policy_pack_ids: tuple[str, ...],
) -> Path | None:
    """Write a pre-redaction replay manifest. Failures never change a verdict."""

    try:
        digest = entrypoint_digest(root, config.entrypoint)
        manifest, omitted = build_replay_manifest(
            run_id=run_id,
            seed=seed,
            spec=spec,
            config=config,
            scenarios=scenarios,
            git_revision=revision,
            entrypoint_digest=digest,
            policy_pack_ids=policy_pack_ids,
        )
        if manifest is None:
            print(
                "AgentCheck warning: replay manifest omitted because every case "
                "failed secret screening",
                file=sys.stderr,
            )
            return None
        path = write_replay_manifest(root, config, manifest)
        if omitted:
            print(
                "AgentCheck warning: "
                f"{len(omitted)} scenario(s) omitted from the replay manifest",
                file=sys.stderr,
            )
        return path
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(
            "AgentCheck warning: replay manifest failed: "
            + redact_log_text(str(exc)),
            file=sys.stderr,
        )
        return None


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
    "replay_suite",
]
