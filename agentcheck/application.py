"""Phase 1 application service connecting inspection, execution, and reporting."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

from agentcheck.analyze import analyze_failures
from agentcheck.adapters import UnsupportedTargetError
from agentcheck.artifacts import ArtifactStore, new_run_id
from agentcheck.config import (
    AgentCheckConfig,
    apply_python_executable,
    contained_path,
    load_config,
)
from agentcheck.coverage import (
    BehavioralCoverage,
    BehavioralCoverageReferenceScope,
    analyze_behavioral_coverage,
)
from agentcheck.domain import (
    AgentSpec,
    CanonicalRun,
    CaseEvaluation,
    Finding,
    InfrastructureError,
    Scenario,
    Verdict,
)
from agentcheck import __version__
from agentcheck.evaluate import evaluate_run, infrastructure_evaluation
from agentcheck.fixtures import (
    load_prerequisite_outcomes,
    load_representative_inputs,
    load_scenario_requests,
)
from agentcheck.identity import identity_mismatch_hint, spec_identity_matches
from agentcheck.errors import (
    ConfigurationError,
    IncompatibleSuiteError,
    ScenarioValidationError,
)
from agentcheck.generate import lint_scenario, lint_suite
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
from agentcheck.generate.templates import (
    incompatible_built_in_suite_message,
    spec_matches_built_in_suite,
)
from agentcheck.policies import (
    apply_policy_packs_to_many,
    attach_declared_policies,
    resolve_policy_packs,
)
from agentcheck.privacy import redact_log_text
from agentcheck.report import render_report
from agentcheck.replay import (
    ReplayManifest,
    build_replay_manifest,
    git_revision,
    load_replay_manifest,
    secret_shaped_reason,
    verify_replay_source_bindings,
    verify_replay_spec_bindings,
    write_replay_manifest,
)
from agentcheck.replay.bind import (
    SourceSnapshot,
    capture_source_snapshot,
    verify_source_snapshot,
)
from agentcheck.replay.manifest import replay_manifest_relative_path
from agentcheck.shrink import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_ROUNDS,
    MAX_MAX_CANDIDATES,
    MAX_MAX_ROUNDS,
    MAX_SOURCE_SCANS,
    ShrinkResult,
    extract_failure_signature,
    signatures_match,
    shrink_scenario,
    unsupported_failure_reason,
    write_shrink_result,
)
from agentcheck.shrink.complexity import measure_complexity
from agentcheck.shrink.result import RejectedCount, ShrinkBudget
from agentcheck.store import open_evaluation_store, stored_run_from_execution
from agentcheck.runner.orchestrator import (
    ProcessResult,
    inspect_in_subprocess,
    run_scenario_in_subprocess,
)


ProgressCallback = Callable[[int, int, Scenario, CaseEvaluation], None]
InspectionProgressCallback = Callable[[AgentSpec], None]
PreparationProgressCallback = Callable[[FrozenSuite | None, int, int, int], None]


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
    behavioral_coverage: BehavioralCoverage


@dataclass(frozen=True, slots=True)
class ShrinkExecution:
    target_root: Path
    config: AgentCheckConfig
    source_manifest: ReplayManifest
    source_manifest_path: str
    original_scenario: Scenario
    minimized_scenario: Scenario
    original_evaluation: CaseEvaluation
    result: ShrinkResult
    result_path: Path
    minimized_manifest: ReplayManifest
    minimized_manifest_path: Path
    verification: SuiteExecution


@dataclass(frozen=True, slots=True)
class SuiteExecution:
    target_root: Path
    config: AgentCheckConfig
    run_id: str
    seed: int
    git_revision: str | None
    source_snapshot: SourceSnapshot
    spec: AgentSpec
    frozen_suite: FrozenSuite | None
    scenarios: tuple[Scenario, ...]
    invalid_scenarios: tuple[InvalidScenario, ...]
    runs: tuple[CanonicalRun, ...]
    evaluations: tuple[CaseEvaluation, ...]
    findings: tuple[Finding, ...]
    artifact_directory: Path
    report_path: Path
    behavioral_coverage: BehavioralCoverage
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
    python_executable: str | None = None,
) -> tuple[Path, AgentCheckConfig, ProcessResult[AgentSpec]]:
    root, config = load_config(target)
    config = apply_python_executable(config, python_executable)
    inspection = inspect_in_subprocess(
        root,
        config,
        timeout_seconds=timeout_seconds,
    )
    if inspection.value is not None:
        packs = resolve_policy_packs(root, config, spec=inspection.value)
        inspection = replace(
            inspection,
            value=attach_declared_policies(inspection.value, packs),
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


def _require_supported_spec(inspection: ProcessResult[AgentSpec]) -> AgentSpec:
    spec = inspection.require_value()
    if inspection.preflight_issues:
        raise UnsupportedTargetError(list(inspection.preflight_issues))
    return spec


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
    python_executable: str | None = None,
) -> SuiteGeneration:
    """Inspect a target, freeze its deterministic suite, and persist a reviewable file."""

    root, config = load_config(target)
    config = apply_python_executable(config, python_executable)
    actual_seed = _effective_seed(config, seed)
    destination = resolve_suite_destination(root, config, out)
    if not force and (destination.exists() or destination.is_symlink()):
        raise ConfigurationError(
            f"{destination.name} already exists at {destination}; "
            "re-run with --force to replace it"
        )
    inspection = inspect_in_subprocess(root, config, timeout_seconds=timeout_seconds)
    packs = resolve_policy_packs(
        root, config, policy_packs, spec=_require_supported_spec(inspection)
    )
    spec = attach_declared_policies(_require_supported_spec(inspection), packs)
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
    representative_inputs = load_representative_inputs(root, spec)
    scenario_requests = load_scenario_requests(root, spec)
    prerequisite_outcomes = load_prerequisite_outcomes(root, spec)
    coverage_reference_scenarios: list[Scenario] = []
    suite = build_frozen_suite(
        spec,
        config,
        seed=actual_seed,
        include_mutations=include_mutations,
        max_mutations=max_mutations,
        policy_packs=packs,
        max_cases=max_cases,
        realizer=active_realizer,
        representative_inputs=representative_inputs,
        scenario_requests=scenario_requests,
        prerequisite_outcomes=prerequisite_outcomes,
        _reference_scenarios=coverage_reference_scenarios,
    )
    behavioral_coverage = analyze_behavioral_coverage(
        spec,
        suite.scenarios,
        suite_fingerprint=suite.fingerprint,
        reference_scenarios=tuple(coverage_reference_scenarios),
    )
    write_frozen_suite(destination, suite, force=force)
    return SuiteGeneration(
        target_root=root,
        config=config,
        spec=spec,
        suite=suite,
        suite_path=destination,
        behavioral_coverage=behavioral_coverage,
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
    on_inspected: InspectionProgressCallback | None = None,
    on_prepared: PreparationProgressCallback | None = None,
    persist_store: bool = True,
    select: str | None = None,
    python_executable: str | None = None,
) -> SuiteExecution:
    """Execute the configured suite and persist an immutable report."""

    root, config = load_config(target)
    config = apply_python_executable(config, python_executable)
    if select not in (None, "coverage"):
        raise ConfigurationError(
            "select must be omitted or set to 'coverage'"
        )
    # A frozen suite is untrusted input: parse and integrity-check it before the
    # target is imported, so a malformed file never reaches model or tool code.
    configured = configured_frozen_suite(root, config)
    suite_run_id = run_id or new_run_id()
    # Capture bounded source bytes and named-commit eligibility before the first
    # target import. Later endpoint comparisons are ordinary trusted-code
    # controls, not hostile filesystem immutability or ABA protection.
    source_snapshot = capture_source_snapshot(root, config)
    revision = source_snapshot.git_revision
    inspection = inspect_in_subprocess(root, config)
    packs = resolve_policy_packs(
        root, config, spec=_require_supported_spec(inspection)
    )
    spec = attach_declared_policies(_require_supported_spec(inspection), packs)
    if on_inspected is not None:
        on_inspected(spec)

    frozen: FrozenSuite | None = None
    if configured is None:
        # The built-in suite is a domain-specific tool contract, not a generic
        # fallback. Linting it against an unrelated agent would drop every case
        # as nonexistent_tool and look like an empty suite rather than a mismatch.
        if not spec_matches_built_in_suite(spec, config.suite):
            raise IncompatibleSuiteError(
                incompatible_built_in_suite_message(spec, config.suite)
            )
        effective_seed = _effective_seed(config, seed)
        candidates = apply_policy_packs_to_many(
            _suite(config, seed), packs, declared=True
        )
    else:
        suite_path, frozen = configured
        if not spec_identity_matches(spec, frozen.spec_id):
            raise ConfigurationError(
                f"frozen suite {suite_path.name} was generated for target "
                f"{frozen.spec_id}, but this target inspects as {spec.spec_id}; "
                "re-run agentcheck generate"
                + identity_mismatch_hint(spec, frozen.spec_id)
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

    if frozen is not None and valid and invalid:
        # Every case persisted in a frozen suite is part of that suite's evidence
        # contract. Dropping only the cases that no longer lint would make the
        # remaining verdicts and coverage look complete while silently shrinking
        # their denominator. Refuse before selection, scenario-execution
        # workers, or artifacts can turn that incomplete run into
        # plausible-looking behavioral evidence.
        raise ScenarioValidationError(
            f"Frozen suite contains {len(invalid)} scenario(s) that fail lint "
            "against the inspected target; refusing partial execution. Re-run "
            "agentcheck generate. No agent verdict was produced."
        )

    if not valid:
        raise ScenarioValidationError(
            "No valid scenarios remain after linting; no agent verdict was produced."
        )

    reference_scenarios = tuple(valid)
    bounded_frozen = (
        frozen is not None
        and frozen.selection is not None
        and bool(frozen.selection.excluded_ids)
    )
    reference_scope = (
        BehavioralCoverageReferenceScope.AVAILABLE_SCENARIOS_ONLY
        if bounded_frozen
        else BehavioralCoverageReferenceScope.COMPLETE
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

    if reference_scope is BehavioralCoverageReferenceScope.AVAILABLE_SCENARIOS_ONLY:
        # A persisted pruned suite cannot supply the cases it discarded before
        # freezing, so the reference set can only be what actually survived.
        # Re-read it here so a further runtime selection is reflected too.
        reference_scenarios = tuple(valid)

    if on_prepared is not None:
        excluded_by_selection = (
            len(selection_plan.excluded_ids) if selection_plan is not None else 0
        )
        on_prepared(frozen, len(valid), len(invalid), excluded_by_selection)

    return _execute_valid_scenarios(
        root=root,
        config=config,
        spec=spec,
        suite_run_id=suite_run_id,
        effective_seed=effective_seed,
        revision=revision,
        source_snapshot=source_snapshot,
        valid=tuple(valid),
        invalid=tuple(invalid),
        reference_scenarios=reference_scenarios,
        reference_scope=reference_scope,
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
    python_executable: str | None = None,
) -> SuiteExecution:
    """Re-execute a replay manifest through the existing isolated runtime."""

    root, config = load_config(target)
    config = apply_python_executable(config, python_executable)
    # Untrusted manifest is parsed before the target is imported.
    manifest = load_replay_manifest(root, manifest_path)
    suite_run_id = run_id or new_run_id()
    source_snapshot = verify_replay_source_bindings(
        manifest,
        root=root,
        config=config,
    )
    _warn_legacy_source_binding(manifest)
    verify_source_snapshot(
        source_snapshot,
        root=root,
        config=config,
        phase="before replay target inspection",
    )
    revision = source_snapshot.git_revision
    inspection = inspect_in_subprocess(root, config)
    packs = resolve_policy_packs(root, config, spec=inspection.require_value())
    spec = attach_declared_policies(inspection.require_value(), packs)
    pack_ids = tuple(pack.pack_id for pack in packs)
    verify_replay_spec_bindings(
        manifest, spec=spec, policy_pack_ids=pack_ids
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
        source_snapshot=source_snapshot,
        valid=tuple(valid),
        invalid=(),
        reference_scenarios=tuple(valid),
        reference_scope=BehavioralCoverageReferenceScope.COMPLETE,
        frozen=None,
        selection_plan=None,
        progress=progress,
        persist_store=persist_store,
        policy_pack_ids=pack_ids,
    )


def shrink_suite(
    target: str | Path,
    manifest_path: str,
    *,
    scenario_id: str | None = None,
    run_id: str | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    persist_store: bool = True,
    python_executable: str | None = None,
) -> ShrinkExecution:
    """Minimize a failing replay case without weakening oracles or bindings."""

    if max_candidates < 1 or max_candidates > MAX_MAX_CANDIDATES:
        raise ConfigurationError(
            f"max-candidates must be between 1 and {MAX_MAX_CANDIDATES}"
        )
    if max_rounds < 1 or max_rounds > MAX_MAX_ROUNDS:
        raise ConfigurationError(
            f"max-rounds must be between 1 and {MAX_MAX_ROUNDS}"
        )

    root, config = load_config(target)
    config = apply_python_executable(config, python_executable)
    source_manifest = load_replay_manifest(root, manifest_path)
    source_resolved = contained_path(root, manifest_path)
    suite_run_id = run_id or new_run_id()
    destination_relative = replay_manifest_relative_path(config, suite_run_id)
    destination = contained_path(root, destination_relative)
    if destination == source_resolved:
        raise ConfigurationError(
            "refusing to overwrite the source replay manifest; pass a different --run-id"
        )

    source_snapshot = verify_replay_source_bindings(
        source_manifest,
        root=root,
        config=config,
    )
    _warn_legacy_source_binding(source_manifest)
    verify_source_snapshot(
        source_snapshot,
        root=root,
        config=config,
        phase="before shrink target inspection",
    )
    inspection = inspect_in_subprocess(root, config)
    packs = resolve_policy_packs(root, config)
    spec = attach_declared_policies(inspection.require_value(), packs)
    pack_ids = tuple(pack.pack_id for pack in packs)
    verify_replay_spec_bindings(
        source_manifest, spec=spec, policy_pack_ids=pack_ids
    )
    verify_source_snapshot(
        source_snapshot,
        root=root,
        config=config,
        phase="after shrink target inspection",
    )

    original, original_evaluation = _select_shrink_target(
        source_manifest,
        scenario_id=scenario_id,
        root=root,
        config=config,
        spec=spec,
        run_id=suite_run_id,
    )
    executions = {"count": 0}

    def execute_candidate(scenario: Scenario) -> CaseEvaluation:
        executions["count"] += 1
        return _run_and_evaluate(
            root,
            config,
            spec,
            scenario,
            f"{suite_run_id}-c{executions['count']:03d}",
        )

    outcome = shrink_scenario(
        original,
        original_evaluation,
        spec=spec,
        execute=execute_candidate,
        max_candidates=max_candidates,
        max_rounds=max_rounds,
    )
    verify_source_snapshot(
        source_snapshot,
        root=root,
        config=config,
        phase="after shrink scenario workers",
    )
    minimized = _annotate_minimized(outcome.scenario, original)
    if secret_shaped_reason(minimized) is not None:
        raise ConfigurationError(
            "minimized scenario cannot be serialized without a secret-shaped value"
        )
    minimized_manifest = ReplayManifest(
        created_from_run_id=suite_run_id,
        agentcheck_version=__version__,
        seed=source_manifest.seed,
        spec_binding=source_manifest.spec_binding,
        source_binding=source_manifest.source_binding,
        environment_requirements=source_manifest.environment_requirements,
        cases=(minimized,),
    )
    minimized_path = write_replay_manifest(root, config, minimized_manifest)
    verification = replay_suite(
        root,
        destination_relative,
        run_id=f"{suite_run_id}-verify",
        persist_store=persist_store,
    )
    if len(verification.evaluations) != 1:
        raise ConfigurationError("minimized replay did not produce exactly one evaluation")
    try:
        verified_signature = extract_failure_signature(verification.evaluations[0])
    except ValueError as exc:
        raise ConfigurationError(
            f"minimized replay did not reproduce a shrinkable failure: {exc}"
        ) from exc
    if not signatures_match(outcome.signature, verified_signature):
        raise ConfigurationError(
            "minimized replay did not reproduce the original failure signature"
        )

    result = ShrinkResult(
        source_manifest_id=source_manifest.manifest_id,
        source_manifest_fingerprint=source_manifest.fingerprint,
        source_scenario_id=original.scenario_id,
        source_scenario_fingerprint=original.fingerprint,
        failure_signature=outcome.signature,
        original_complexity=measure_complexity(original),
        minimized_complexity=outcome.complexity,
        minimized_scenario_fingerprint=minimized.fingerprint,
        minimized_manifest_id=minimized_manifest.manifest_id,
        minimized_manifest_path=destination_relative,
        candidate_executions=outcome.candidate_executions,
        accepted_reductions=outcome.accepted_reductions,
        rejected_candidates=outcome.rejected_candidates,
        skipped_invalid=outcome.skipped_invalid,
        rejected_by_reason=tuple(
            RejectedCount(reason=reason, count=count)
            for reason, count in outcome.rejected_by_reason
        ),
        budget=ShrinkBudget(max_candidates=max_candidates, max_rounds=max_rounds),
        budget_exhausted=outcome.budget_exhausted,
        minimality=outcome.minimality,
        agentcheck_version=__version__,
    )
    result_path = write_shrink_result(root, config, result, suite_run_id)
    return ShrinkExecution(
        target_root=root,
        config=config,
        source_manifest=source_manifest,
        source_manifest_path=manifest_path,
        original_scenario=original,
        minimized_scenario=minimized,
        original_evaluation=original_evaluation,
        result=result,
        result_path=result_path,
        minimized_manifest=minimized_manifest,
        minimized_manifest_path=minimized_path,
        verification=verification,
    )


def _select_shrink_target(
    manifest: ReplayManifest,
    *,
    scenario_id: str | None,
    root: Path,
    config: AgentCheckConfig,
    spec: AgentSpec,
    run_id: str,
) -> tuple[Scenario, CaseEvaluation]:
    if scenario_id is not None:
        selected = [case for case in manifest.cases if case.scenario_id == scenario_id]
        if not selected:
            raise ConfigurationError(
                f"replay manifest has no scenario {scenario_id!r}"
            )
        scenario = selected[0]
        _require_lint_clean(scenario, spec)
        evaluation = _run_and_evaluate(
            root, config, spec, scenario, f"{run_id}-original"
        )
        reason = unsupported_failure_reason(evaluation)
        if reason is not None:
            raise ConfigurationError(
                f"scenario {scenario.scenario_id} is not a shrinkable counterexample: {reason}"
            )
        return scenario, evaluation

    last_reason = "replay manifest contains no shrinkable FAIL case"
    scanned = 0
    for index, scenario in enumerate(manifest.cases, start=1):
        if scanned >= MAX_SOURCE_SCANS:
            raise ConfigurationError(
                "replay manifest exceeded the shrink source scan bound of "
                f"{MAX_SOURCE_SCANS} cases without a shrinkable FAIL; "
                "pass --scenario-id to select a case"
            )
        scanned += 1
        _require_lint_clean(scenario, spec)
        evaluation = _run_and_evaluate(
            root, config, spec, scenario, f"{run_id}-scan-{index:03d}"
        )
        reason = unsupported_failure_reason(evaluation)
        if reason is None:
            return scenario, evaluation
        last_reason = reason
    raise ConfigurationError(last_reason)


def _require_lint_clean(scenario: Scenario, spec: AgentSpec) -> None:
    issues = lint_scenario(scenario, spec)
    if issues:
        raise ConfigurationError(
            "replay manifest contains scenarios that fail lint against this target"
        )


def _annotate_minimized(scenario: Scenario, original: Scenario) -> Scenario:
    payload = json.loads(scenario.model_dump_json())
    payload["title"] = (
        original.title
        if original.title.startswith("Minimized:")
        else f"Minimized: {original.title}"
    )
    payload["description"] = (
        "Smaller replayable counterexample that preserves the original "
        "deterministic failure signature. This is not a claim about the "
        "business root cause."
    )
    payload["fingerprint"] = ""
    return Scenario.model_validate_json(json.dumps(payload, ensure_ascii=False))


def _run_and_evaluate(
    root: Path,
    config: AgentCheckConfig,
    spec: AgentSpec,
    scenario: Scenario,
    case_run_id: str,
) -> CaseEvaluation:
    result = run_scenario_in_subprocess(
        root,
        config,
        scenario,
        case_run_id,
        expected_target_id=spec.spec_id,
    )
    if result.value is None:
        return _process_failure_evaluation(scenario, result, case_run_id)
    return evaluate_run(scenario, result.value)


def _execute_valid_scenarios(
    *,
    root: Path,
    config: AgentCheckConfig,
    spec: AgentSpec,
    suite_run_id: str,
    effective_seed: int,
    revision: str | None,
    source_snapshot: SourceSnapshot,
    valid: tuple[Scenario, ...],
    invalid: tuple[InvalidScenario, ...],
    reference_scenarios: tuple[Scenario, ...],
    reference_scope: BehavioralCoverageReferenceScope,
    frozen: FrozenSuite | None,
    selection_plan: SelectionPlan | None,
    progress: ProgressCallback | None,
    persist_store: bool,
    policy_pack_ids: tuple[str, ...],
) -> SuiteExecution:
    # Reserve the run ID before scenario workers so a collision cannot replace
    # an existing replay or begin a second execution under the same identity.
    # The empty directory is deliberately not a trusted stored run: the loader
    # requires the complete artifact set written only after replay succeeds.
    artifacts = ArtifactStore(root, config.artifacts_directory, suite_run_id)
    verify_source_snapshot(
        source_snapshot,
        root=root,
        config=config,
        phase="after inspection and preparation, before scenario workers",
    )
    valid_scenarios = tuple(valid)
    behavioral_coverage = analyze_behavioral_coverage(
        spec,
        valid_scenarios,
        reference_scenarios=reference_scenarios,
        reference_scope=reference_scope,
        suite_fingerprint=frozen.fingerprint if frozen is not None else None,
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
    verify_source_snapshot(
        source_snapshot,
        root=root,
        config=config,
        phase="after scenario workers",
    )
    findings = analyze_failures(valid_scenarios, evaluations)
    replay_path = _require_complete_replay_manifest(
        root=root,
        config=config,
        spec=spec,
        run_id=suite_run_id,
        seed=effective_seed,
        scenarios=valid_scenarios,
        source_snapshot=source_snapshot,
        policy_pack_ids=policy_pack_ids,
    )
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
        "behavioral_coverage": behavioral_coverage.model_dump(mode="json"),
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
        behavioral_coverage=behavioral_coverage,
        _coverage_reference_scenarios=reference_scenarios,
    )
    report_path = artifacts.write_text("report.html", report)
    execution = SuiteExecution(
        target_root=root,
        config=config,
        run_id=suite_run_id,
        seed=effective_seed,
        git_revision=revision,
        source_snapshot=source_snapshot,
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
        behavioral_coverage=behavioral_coverage,
        replay_manifest_path=replay_path,
    )
    if persist_store:
        _persist_execution(execution)
    return execution


def _require_complete_replay_manifest(
    *,
    root: Path,
    config: AgentCheckConfig,
    spec: AgentSpec,
    run_id: str,
    seed: int,
    scenarios: tuple[Scenario, ...],
    source_snapshot: SourceSnapshot,
    policy_pack_ids: tuple[str, ...],
) -> Path:
    """Write one replay case per executed scenario or fail the execution."""

    try:
        manifest, omitted = build_replay_manifest(
            run_id=run_id,
            seed=seed,
            spec=spec,
            config=config,
            scenarios=scenarios,
            git_revision=source_snapshot.git_revision,
            entrypoint_digest=source_snapshot.entrypoint_digest,
            policy_pack_ids=policy_pack_ids,
            file_set=source_snapshot.file_set,
        )
        if manifest is None or omitted or manifest.omitted:
            omitted_count = max(
                len(omitted),
                len(manifest.omitted) if manifest is not None else 0,
            )
            raise ConfigurationError(
                "complete replay manifest required: "
                f"{omitted_count or len(scenarios)} of {len(scenarios)} "
                "scenario(s) could not be serialized safely"
            )
        if tuple(manifest.cases) != scenarios:
            raise ConfigurationError(
                "complete replay manifest required: replay cases do not match "
                "each executed scenario exactly once"
            )
        return write_replay_manifest(root, config, manifest)
    except (KeyboardInterrupt, SystemExit):
        raise
    except ConfigurationError:
        raise
    except Exception as exc:
        message = redact_log_text(str(exc)) or type(exc).__name__
        raise ConfigurationError(
            f"unable to produce complete replay manifest: {message}"
        ) from exc


def _warn_legacy_source_binding(manifest: ReplayManifest) -> None:
    if manifest.source_binding.file_set is not None:
        return
    print(
        "AgentCheck warning: replay manifest has no source file-set; only the "
        "entrypoint digest is bound. Re-run test to emit the stronger binding.",
        file=sys.stderr,
    )


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
    "ShrinkExecution",
    "SuiteExecution",
    "SuiteGeneration",
    "execute_suite",
    "generate_suite",
    "inspect_target",
    "replay_suite",
    "shrink_suite",
]
