from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agentcheck.artifacts import ArtifactStore
from agentcheck.baseline import (
    BASELINE_CONTRACT_VERSION,
    COMPARISON_CONTRACT_VERSION,
    compare_baselines,
    format_comparison,
)
from agentcheck.baseline.build import baseline_from_loaded
from agentcheck.baseline.service import check_baseline, create_baseline
from agentcheck.cli import main
from agentcheck.domain import (
    AgentProperty,
    AgentSpec,
    AssertionResult,
    CapabilitiesSpec,
    CanonicalRun,
    CaseEvaluation,
    Evidence,
    EvidenceKind,
    Finding,
    FixTarget,
    IdentitySpec,
    InfrastructureError,
    InspectionProvenance,
    InstructionsSpec,
    InterfaceSpec,
    ObservabilitySpec,
    RootCauseLayer,
    RunTermination,
    RuntimeSpec,
    Scenario,
    Severity,
    SourceKind,
    SourceReference,
    SpecEvidence,
    SuggestedFix,
    ToolsSpec,
    Verdict,
    utc_now,
)
from agentcheck.errors import ConfigurationError
from agentcheck.generate.templates import build_account_support_suite
from agentcheck.report import load_stored_run
from agentcheck.review.service import record_finding_review


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"
SEED = 1729
SUITE = build_account_support_suite(seed=SEED)
SCENARIO_A = SUITE[0]
SCENARIO_B = SUITE[1]
SCENARIO_C = SUITE[2]


def _property(value: object) -> AgentProperty[object]:
    return AgentProperty(
        value=value,
        source=SourceReference(kind=SourceKind.RUNTIME_INTROSPECTION, locator="test"),
        confidence=1,
        evidence=(SpecEvidence(evidence_id="e", summary="test"),),
    )


def _spec(spec_id: str = "spec") -> AgentSpec:
    return AgentSpec(
        spec_id=spec_id,
        identity=IdentitySpec(
            name=_property("Account Support Agent"),  # type: ignore[arg-type]
            framework=_property("OpenAI Agents SDK"),  # type: ignore[arg-type]
            framework_version=_property("0.20.0"),  # type: ignore[arg-type]
            provider=_property(None),  # type: ignore[arg-type]
            model=_property(None),  # type: ignore[arg-type]
        ),
        interface=InterfaceSpec(
            entrypoint=_property("agent.py:agent"),  # type: ignore[arg-type]
            input_modalities=_property(("text",)),  # type: ignore[arg-type]
            output_modalities=_property(("text",)),  # type: ignore[arg-type]
            input_schema=_property(None),  # type: ignore[arg-type]
            output_schema=_property(None),  # type: ignore[arg-type]
            interactive=_property(True),  # type: ignore[arg-type]
        ),
        instructions=InstructionsSpec(
            system=_property("support"),  # type: ignore[arg-type]
            developer=_property(None),  # type: ignore[arg-type]
        ),
        capabilities=CapabilitiesSpec(),
        tools=ToolsSpec(),
        runtime=RuntimeSpec(
            max_model_turns=_property(None),  # type: ignore[arg-type]
            max_tool_calls=_property(None),  # type: ignore[arg-type]
            timeout_seconds=_property(None),  # type: ignore[arg-type]
            token_budget=_property(None),  # type: ignore[arg-type]
            cost_budget_usd=_property(None),  # type: ignore[arg-type]
        ),
        observability=ObservabilitySpec(
            supported_event_types=_property(("final_output",)),  # type: ignore[arg-type]
            usage_metrics=_property(()),  # type: ignore[arg-type]
            provider_request_ids=_property(False),  # type: ignore[arg-type]
            source_event_links=_property(True),  # type: ignore[arg-type]
        ),
        provenance=InspectionProvenance(
            inspector="test",
            inspector_version="1",
            inspected_at=utc_now(),
            target="test",
            sources=(SourceReference(kind=SourceKind.RUNTIME_INTROSPECTION, locator="test"),),
        ),
    )


def _pass_evaluation(scenario_id: str, run_id: str) -> CaseEvaluation:
    now = utc_now()
    return CaseEvaluation(
        evaluation_id=f"eval-{scenario_id}",
        scenario_id=scenario_id,
        run_id=run_id,
        verdict=Verdict.PASS,
        assertions=(
            AssertionResult(
                assertion_id="a-pass",
                criterion="completed",
                result=Verdict.PASS,
                oracle_ids=("oracle-1",),
                rationale="The stored fixture passed.",
            ),
        ),
        started_at=now,
        completed_at=now,
        summary="passed",
    )


def _fail_evaluation(
    scenario_id: str,
    run_id: str,
    *,
    assertion_id: str = "a-fail",
    oracle_ids: tuple[str, ...] = ("oracle-1",),
) -> CaseEvaluation:
    now = utc_now()
    return CaseEvaluation(
        evaluation_id=f"eval-{scenario_id}",
        scenario_id=scenario_id,
        run_id=run_id,
        verdict=Verdict.FAIL,
        assertions=(
            AssertionResult(
                assertion_id=assertion_id,
                criterion="must not fail the contract",
                result=Verdict.FAIL,
                oracle_ids=oracle_ids,
                supporting_evidence_ids=("ev-1",),
                rationale="A required assertion failed.",
                confidence=0.95,
            ),
        ),
        evidence=(
            Evidence(
                evidence_id="ev-1",
                kind=EvidenceKind.TOOL_ATTEMPT,
                summary="observed failure",
                source_ids=("event-1",),
            ),
        ),
        started_at=now,
        completed_at=now,
        summary="failed",
    )


def _inconclusive_evaluation(scenario_id: str, run_id: str) -> CaseEvaluation:
    now = utc_now()
    return CaseEvaluation(
        evaluation_id=f"eval-{scenario_id}",
        scenario_id=scenario_id,
        run_id=run_id,
        verdict=Verdict.INCONCLUSIVE,
        assertions=(
            AssertionResult(
                assertion_id="a-inc",
                criterion="budget must be measured",
                result=Verdict.INCONCLUSIVE,
                oracle_ids=("oracle-1",),
                missing_evidence=("token usage",),
                rationale="A declared budget could not be measured.",
                confidence=0.4,
            ),
        ),
        started_at=now,
        completed_at=now,
        summary="inconclusive",
    )


def _infra_evaluation(scenario_id: str, run_id: str) -> CaseEvaluation:
    now = utc_now()
    return CaseEvaluation(
        evaluation_id=f"eval-{scenario_id}",
        scenario_id=scenario_id,
        run_id=run_id,
        verdict=Verdict.INFRA_ERROR,
        assertions=(
            AssertionResult(
                assertion_id="a-infra",
                criterion="scenario must execute",
                result=Verdict.INCONCLUSIVE,
                oracle_ids=("oracle-1",),
                missing_evidence=("worker trajectory",),
                rationale="The worker failed before producing a trajectory.",
            ),
        ),
        evidence=(
            Evidence(
                evidence_id="ev-infra",
                kind=EvidenceKind.ERROR,
                summary="worker crashed",
                source_ids=("worker-1",),
            ),
        ),
        started_at=now,
        completed_at=now,
        summary="infrastructure error",
        infrastructure_error=InfrastructureError(
            code="worker_crashed",
            message="Child exited without an artifact.",
            phase="runner",
        ),
    )


def _finding(scenario_id: str) -> Finding:
    return Finding(
        finding_id="finding:duplicate_side_effect",
        failure_signature="duplicate_side_effect",
        title="Duplicate side-effect call",
        description="The same state-changing tool and arguments were executed more than once.",
        severity=Severity.HIGH,
        confidence=0.95,
        affected_scenario_ids=(scenario_id,),
        evidence_ids=("ev-1",),
        root_cause_layer=RootCauseLayer.WORKFLOW_LOGIC,
        likely_cause="The workflow lacks an execution/idempotency guard.",
        suggested_fixes=(
            SuggestedFix(
                fix_id="fix:duplicate_side_effect",
                target=FixTarget.WORKFLOW_LOGIC,
                summary="Record successful side effects.",
                rationale="Supported by one deterministic failing case.",
                proposed_change="Record successful side effects.",
                confidence=0.9,
                evidence_ids=("ev-1",),
            ),
        ),
    )


def _mutate_world(scenario: Scenario, key: str, value: str) -> Scenario:
    data = scenario.model_dump(mode="python")
    world = dict(data.get("initial_world_state") or {})
    world[key] = value
    data["initial_world_state"] = world
    data["fingerprint"] = ""
    return Scenario.model_validate(data)


def _write_run(
    root: Path,
    run_id: str,
    cases: list[tuple[Scenario, CaseEvaluation]],
    *,
    spec: AgentSpec | None = None,
    findings: tuple[Finding, ...] = (),
) -> Path:
    spec = spec or _spec()
    runs = []
    evaluations = []
    scenarios = []
    counts = {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0, "INFRA_ERROR": 0}
    for index, (scenario, evaluation) in enumerate(cases, start=1):
        scenarios.append(scenario)
        evaluations.append(evaluation)
        counts[evaluation.verdict.value] += 1
        runs.append(
            CanonicalRun(
                run_id=evaluation.run_id or f"{run_id}-case-{index:03d}",
                scenario_id=scenario.scenario_id,
                target_id=spec.spec_id,
                started_at=utc_now(),
                ended_at=utc_now(),
                termination=RunTermination.COMPLETED,
            )
        )
    artifacts = ArtifactStore(root, ".agentcheck", run_id)
    artifacts.write_json("agent-spec.json", spec)
    artifacts.write_json(
        "suite.json",
        {
            "schema_version": "agentcheck.suite.v1",
            "run_id": run_id,
            "seed": SEED,
            "scenarios": scenarios,
        },
    )
    artifacts.write_json(
        "invalid-scenarios.json",
        {"schema_version": "agentcheck.invalid_scenarios.v1", "items": []},
    )
    artifacts.write_jsonl("runs.jsonl", runs)
    artifacts.write_jsonl("evaluations.jsonl", evaluations)
    artifacts.write_json("findings.json", findings)
    suite_size = len(scenarios)
    fail = counts["FAIL"]
    artifacts.write_json(
        "summary.json",
        {
            "schema_version": "agentcheck.summary.v1",
            "run_id": run_id,
            "target": str(root),
            "git_revision": None,
            "suite_size": suite_size,
            "invalid_scenarios": 0,
            "observed_suite_pass_rate": (
                None if suite_size == 0 else (suite_size - fail) / suite_size
            ),
            "counts": counts,
            "finding_count": len(findings),
            "seed": SEED,
        },
    )
    artifacts.write_text("report.html", "<html><body>placeholder</body></html>")
    return artifacts.root


def _target(tmp_path: Path) -> Path:
    root = tmp_path / "account_agent"
    shutil.copytree(
        EXAMPLE,
        root,
        ignore=shutil.ignore_patterns(".agentcheck", "__pycache__"),
    )
    return root


def _patch_execution_tripwires(monkeypatch: pytest.MonkeyPatch) -> None:
    import agentcheck.application as application
    import agentcheck.runner.orchestrator as orchestrator
    from agentcheck.runner.tool_gateway import ToolGateway

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("baseline commands must not inspect, run a worker, or invoke tools")

    monkeypatch.setattr(application, "inspect_in_subprocess", explode)
    monkeypatch.setattr(application, "run_scenario_in_subprocess", explode)
    monkeypatch.setattr(orchestrator, "inspect_in_subprocess", explode)
    monkeypatch.setattr(orchestrator, "run_scenario_in_subprocess", explode)
    monkeypatch.setattr(ToolGateway, "__init__", explode)


def _create_and_check(
    root: Path,
    *,
    baseline_run: str,
    current_run: str,
) -> tuple[int, object]:
    create_baseline(root, run_id=baseline_run, out="agentcheck-baseline.json")
    checked = check_baseline(
        root,
        baseline_path="agentcheck-baseline.json",
        run_id=current_run,
    )
    return checked.exit_code, checked.comparison


def test_identical_baseline_and_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-base",
        [
            (SCENARIO_A, _pass_evaluation(SCENARIO_A.scenario_id, "run-base-a")),
            (SCENARIO_B, _fail_evaluation(SCENARIO_B.scenario_id, "run-base-b")),
        ],
    )
    exit_code, comparison = _create_and_check(root, baseline_run="run-base", current_run="run-base")
    assert exit_code == 0
    assert comparison.new_regression_count == 0
    assert comparison.unchanged_failure_count == 1
    assert comparison.resolved_count == 0
    assert comparison.schema_version == COMPARISON_CONTRACT_VERSION


def test_existing_failure_remains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    cases = [
        (SCENARIO_A, _pass_evaluation(SCENARIO_A.scenario_id, "case-a")),
        (SCENARIO_B, _fail_evaluation(SCENARIO_B.scenario_id, "case-b")),
    ]
    _write_run(root, "run-old", cases)
    _write_run(root, "run-new", cases)
    exit_code, comparison = _create_and_check(root, baseline_run="run-old", current_run="run-new")
    assert exit_code == 0
    assert comparison.unchanged_failure_count == 1
    assert comparison.new_regression_count == 0


def test_failure_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [(SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "old-a"))],
    )
    _write_run(
        root,
        "run-new",
        [(SCENARIO_A, _pass_evaluation(SCENARIO_A.scenario_id, "new-a"))],
    )
    exit_code, comparison = _create_and_check(root, baseline_run="run-old", current_run="run-new")
    assert exit_code == 0
    assert comparison.resolved_count == 1
    assert comparison.new_regression_count == 0


def test_removed_failure_does_not_block_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [
            (SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "old-a")),
            (SCENARIO_B, _pass_evaluation(SCENARIO_B.scenario_id, "old-b")),
        ],
    )
    _write_run(
        root,
        "run-new",
        [(SCENARIO_B, _pass_evaluation(SCENARIO_B.scenario_id, "new-b"))],
    )
    exit_code, comparison = _create_and_check(root, baseline_run="run-old", current_run="run-new")
    assert exit_code == 0
    assert any(item.category == "removed_scenario" for item in comparison.items)
    assert comparison.new_regression_count == 0


def test_new_hard_failure_and_pass_to_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [(SCENARIO_A, _pass_evaluation(SCENARIO_A.scenario_id, "old-a"))],
    )
    _write_run(
        root,
        "run-new",
        [
            (SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "new-a")),
            (SCENARIO_B, _fail_evaluation(SCENARIO_B.scenario_id, "new-b")),
        ],
    )
    exit_code, comparison = _create_and_check(root, baseline_run="run-old", current_run="run-new")
    assert exit_code == 1
    assert comparison.new_regression_count == 2
    categories = {item.category for item in comparison.items if item.blocking}
    assert categories == {"new_regression"}


def test_unsigned_schema_and_budget_fails_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    schema_fail = _fail_evaluation(
        SCENARIO_A.scenario_id, "new-schema", assertion_id="schema:attempt-0001"
    )
    budget_fail = _fail_evaluation(
        SCENARIO_B.scenario_id, "new-budget", assertion_id="budget:attempt-0002"
    )
    _write_run(
        root,
        "run-old",
        [
            (SCENARIO_A, _pass_evaluation(SCENARIO_A.scenario_id, "old-a")),
            (SCENARIO_C, _pass_evaluation(SCENARIO_C.scenario_id, "old-c")),
        ],
    )
    _write_run(
        root,
        "run-new",
        [
            (SCENARIO_A, schema_fail),
            (SCENARIO_B, budget_fail),
            (SCENARIO_C, _pass_evaluation(SCENARIO_C.scenario_id, "new-c")),
        ],
    )
    exit_code, comparison = _create_and_check(root, baseline_run="run-old", current_run="run-new")
    assert exit_code == 1
    by_id = {item.scenario_id: item for item in comparison.items}
    assert by_id[SCENARIO_A.scenario_id].category == "new_regression"
    assert by_id[SCENARIO_A.scenario_id].blocking is True
    assert by_id[SCENARIO_A.scenario_id].baseline_verdict == "PASS"
    assert by_id[SCENARIO_A.scenario_id].current_verdict == "FAIL"
    assert by_id[SCENARIO_B.scenario_id].category == "new_regression"
    assert by_id[SCENARIO_B.scenario_id].blocking is True
    assert by_id[SCENARIO_B.scenario_id].baseline_verdict is None
    assert comparison.new_regression_count == 2
    assert all(item.category != "inconclusive_change" for item in comparison.items)
    assert all(
        item.category != "new_scenario"
        for item in comparison.items
        if item.current_verdict == "FAIL"
    )

    _write_run(
        root,
        "run-old-swapped",
        [
            (SCENARIO_A, _pass_evaluation(SCENARIO_A.scenario_id, "old-a2")),
            (SCENARIO_C, _pass_evaluation(SCENARIO_C.scenario_id, "old-c2")),
        ],
    )
    _write_run(
        root,
        "run-new-swapped",
        [
            (
                SCENARIO_A,
                _fail_evaluation(
                    SCENARIO_A.scenario_id,
                    "new-budget-from-pass",
                    assertion_id="budget:attempt-0003",
                ),
            ),
            (
                SCENARIO_B,
                _fail_evaluation(
                    SCENARIO_B.scenario_id,
                    "new-schema-absent",
                    assertion_id="schema:attempt-0004",
                ),
            ),
            (SCENARIO_C, _pass_evaluation(SCENARIO_C.scenario_id, "new-c2")),
        ],
    )
    create_baseline(root, run_id="run-old-swapped", out="swapped.json")
    swapped = check_baseline(root, baseline_path="swapped.json", run_id="run-new-swapped")
    assert swapped.exit_code == 1
    swapped_by_id = {item.scenario_id: item for item in swapped.comparison.items}
    assert swapped_by_id[SCENARIO_A.scenario_id].category == "new_regression"
    assert swapped_by_id[SCENARIO_A.scenario_id].blocking is True
    assert swapped_by_id[SCENARIO_A.scenario_id].baseline_verdict == "PASS"
    assert swapped_by_id[SCENARIO_B.scenario_id].category == "new_regression"
    assert swapped_by_id[SCENARIO_B.scenario_id].baseline_verdict is None
    assert swapped.comparison.new_regression_count == 2


def test_signature_bearing_fail_behavior_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [(SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "old-a"))],
    )
    _write_run(
        root,
        "run-new",
        [(SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "new-a"))],
    )
    exit_code, comparison = _create_and_check(root, baseline_run="run-old", current_run="run-new")
    assert exit_code == 0
    assert comparison.unchanged_failure_count == 1
    assert comparison.items[0].category == "unchanged_failure"
    assert comparison.items[0].blocking is False


def test_incomparable_fail_to_fail_cannot_certify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    signed = _fail_evaluation(SCENARIO_A.scenario_id, "old-a")
    unsigned = _fail_evaluation(
        SCENARIO_A.scenario_id, "new-a", assertion_id="schema:attempt-0001"
    )
    _write_run(root, "run-old", [(SCENARIO_A, signed)])
    _write_run(root, "run-new", [(SCENARIO_A, unsigned)])
    exit_code, comparison = _create_and_check(root, baseline_run="run-old", current_run="run-new")
    assert exit_code == 2
    assert comparison.items[0].category == "uncertifiable_failure"
    assert comparison.items[0].blocking is False
    assert comparison.new_regression_count == 0

    _write_run(root, "run-both-unsigned-old", [(SCENARIO_A, unsigned)])
    _write_run(
        root,
        "run-both-unsigned-new",
        [
            (
                SCENARIO_A,
                _fail_evaluation(
                    SCENARIO_A.scenario_id,
                    "newer-a",
                    assertion_id="budget:attempt-0009",
                ),
            )
        ],
    )
    create_baseline(root, run_id="run-both-unsigned-old", out="unsigned.json", force=True)
    both = check_baseline(root, baseline_path="unsigned.json", run_id="run-both-unsigned-new")
    assert both.exit_code == 2
    assert both.comparison.items[0].category == "uncertifiable_failure"
    summary = format_comparison(both.comparison)
    assert "Uncertifiable failures:" in summary


def test_fail_to_inconclusive_is_visible_and_non_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [(SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "old-a"))],
    )
    _write_run(
        root,
        "run-new",
        [(SCENARIO_A, _inconclusive_evaluation(SCENARIO_A.scenario_id, "new-a"))],
    )
    exit_code, comparison = _create_and_check(root, baseline_run="run-old", current_run="run-new")
    assert exit_code == 0
    assert comparison.items[0].category == "inconclusive_change"
    summary = format_comparison(comparison)
    assert "FAIL -> INCONCLUSIVE:" in summary
    assert "weak evidence is not a PASS" in summary


def test_removed_known_failures_are_visible_in_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [
            (SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "old-a")),
            (SCENARIO_B, _pass_evaluation(SCENARIO_B.scenario_id, "old-b")),
        ],
    )
    _write_run(
        root,
        "run-new",
        [(SCENARIO_B, _pass_evaluation(SCENARIO_B.scenario_id, "new-b"))],
    )
    exit_code, comparison = _create_and_check(root, baseline_run="run-old", current_run="run-new")
    assert exit_code == 0
    summary = format_comparison(comparison)
    assert "Removed known failures:" in summary
    assert "not equivalent to a clean stable suite" in summary


def test_new_low_confidence_failure_does_not_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [(SCENARIO_A, _pass_evaluation(SCENARIO_A.scenario_id, "old-a"))],
    )
    _write_run(
        root,
        "run-new",
        [
            (SCENARIO_A, _pass_evaluation(SCENARIO_A.scenario_id, "new-a")),
            (SCENARIO_B, _inconclusive_evaluation(SCENARIO_B.scenario_id, "new-b")),
        ],
    )
    exit_code, comparison = _create_and_check(root, baseline_run="run-old", current_run="run-new")
    assert exit_code == 0
    assert comparison.new_regression_count == 0
    assert any(item.category == "inconclusive_change" for item in comparison.items)


def test_inconclusive_and_infra_transitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [
            (SCENARIO_A, _pass_evaluation(SCENARIO_A.scenario_id, "old-a")),
            (SCENARIO_B, _inconclusive_evaluation(SCENARIO_B.scenario_id, "old-b")),
            (SCENARIO_C, _pass_evaluation(SCENARIO_C.scenario_id, "old-c")),
        ],
    )
    _write_run(
        root,
        "run-inc",
        [
            (SCENARIO_A, _inconclusive_evaluation(SCENARIO_A.scenario_id, "inc-a")),
            (SCENARIO_B, _fail_evaluation(SCENARIO_B.scenario_id, "inc-b")),
            (SCENARIO_C, _pass_evaluation(SCENARIO_C.scenario_id, "inc-c")),
        ],
    )
    create_baseline(root, run_id="run-old", out="agentcheck-baseline.json")
    inc = check_baseline(root, baseline_path="agentcheck-baseline.json", run_id="run-inc")
    assert inc.exit_code == 1
    assert inc.comparison.new_regression_count == 1
    assert any(item.category == "inconclusive_change" for item in inc.comparison.items)

    _write_run(
        root,
        "run-infra",
        [
            (SCENARIO_A, _infra_evaluation(SCENARIO_A.scenario_id, "infra-a")),
            (SCENARIO_B, _fail_evaluation(SCENARIO_B.scenario_id, "infra-b")),
            (SCENARIO_C, _pass_evaluation(SCENARIO_C.scenario_id, "infra-c")),
        ],
    )
    infra = check_baseline(root, baseline_path="agentcheck-baseline.json", run_id="run-infra")
    assert infra.exit_code == 2
    assert any(item.category == "infra_change" for item in infra.comparison.items)


def test_changed_scenario_fingerprint_same_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    mutated = _mutate_world(SCENARIO_A, "baseline_probe", "changed")
    assert mutated.scenario_id == SCENARIO_A.scenario_id
    assert mutated.fingerprint != SCENARIO_A.fingerprint
    _write_run(
        root,
        "run-old",
        [(SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "old-a"))],
    )
    _write_run(
        root,
        "run-new",
        [(mutated, _fail_evaluation(mutated.scenario_id, "new-a"))],
    )
    exit_code, comparison = _create_and_check(root, baseline_run="run-old", current_run="run-new")
    assert exit_code == 0
    assert any(item.category == "changed_scenario" for item in comparison.items)
    assert comparison.new_regression_count == 0


def test_changed_failure_signature_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [(SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "old-a"))],
    )
    _write_run(
        root,
        "run-new",
        [
            (
                SCENARIO_A,
                _fail_evaluation(
                    SCENARIO_A.scenario_id,
                    "new-a",
                    assertion_id="a-other",
                    oracle_ids=("oracle-other",),
                ),
            )
        ],
    )
    exit_code, comparison = _create_and_check(root, baseline_run="run-old", current_run="run-new")
    assert exit_code == 1
    assert comparison.changed_failure_count == 1
    assert any(item.category == "changed_failure" and item.blocking for item in comparison.items)


def test_tampered_and_unsupported_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [(SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "old-a"))],
    )
    created = create_baseline(root, run_id="run-old", out="agentcheck-baseline.json")
    payload = json.loads(created.path.read_text(encoding="utf-8"))
    payload["cases"][0]["fingerprint"] = "sha256:" + ("ab" * 32)
    created.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="fingerprint does not match"):
        check_baseline(root, baseline_path="agentcheck-baseline.json", run_id="run-old")

    payload["schema_version"] = "agentcheck.baseline.v0"
    created.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unsupported baseline contract"):
        check_baseline(root, baseline_path="agentcheck-baseline.json", run_id="run-old")


def test_wrong_artifact_kinds_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [(SCENARIO_A, _pass_evaluation(SCENARIO_A.scenario_id, "old-a"))],
    )
    html = root / "report.html"
    html.write_text("<html><body>nope</body></html>", encoding="utf-8")
    sqlite = root / "index.sqlite"
    sqlite.write_bytes(b"SQLite format 3\x00")
    with pytest.raises(ConfigurationError, match="HTML"):
        check_baseline(root, baseline_path="report.html", run_id="run-old")
    with pytest.raises(ConfigurationError, match="SQLite"):
        check_baseline(root, baseline_path="index.sqlite", run_id="run-old")


def test_path_containment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [(SCENARIO_A, _pass_evaluation(SCENARIO_A.scenario_id, "old-a"))],
    )
    with pytest.raises(ConfigurationError, match="inside the target"):
        create_baseline(root, run_id="run-old", out="../escape.json")
    assert main(
        [
            "baseline",
            "check",
            str(root),
            "--baseline",
            "../escape.json",
            "--run-id",
            "run-old",
        ]
    ) == 2


def test_no_target_execution_during_compare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [(SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "old-a"))],
    )
    assert main(["baseline", "create", str(root), "--run-id", "run-old"]) == 0
    assert (
        main(
            [
                "baseline",
                "check",
                str(root),
                "--baseline",
                "agentcheck-baseline.json",
                "--run-id",
                "run-old",
            ]
        )
        == 0
    )


def test_deterministic_comparison_ordering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    old = [
        (SCENARIO_A, _pass_evaluation(SCENARIO_A.scenario_id, "old-a")),
    ]
    new = [
        (SCENARIO_C, _fail_evaluation(SCENARIO_C.scenario_id, "new-c")),
        (SCENARIO_B, _fail_evaluation(SCENARIO_B.scenario_id, "new-b")),
        (SCENARIO_A, _pass_evaluation(SCENARIO_A.scenario_id, "new-a")),
    ]
    _write_run(root, "run-old", old)
    _write_run(root, "run-new", new)
    create_baseline(root, run_id="run-old", out="agentcheck-baseline.json")
    first = check_baseline(root, baseline_path="agentcheck-baseline.json", run_id="run-new")
    reversed_new = list(reversed(new))
    _write_run(root, "run-new-2", reversed_new)
    second = check_baseline(root, baseline_path="agentcheck-baseline.json", run_id="run-new-2")
    assert [item.fingerprint for item in first.comparison.items] == [
        item.fingerprint for item in second.comparison.items
    ]
    assert [item.category for item in first.comparison.items] == [
        item.category for item in second.comparison.items
    ]
    assert first.comparison.new_regression_count == second.comparison.new_regression_count
    baseline_obj = baseline_from_loaded(load_stored_run(root, run_id="run-old"))
    current_obj = baseline_from_loaded(load_stored_run(root, run_id="run-new"))
    direct = compare_baselines(baseline_obj, current_obj)
    assert [item.fingerprint for item in direct.items] == [
        item.fingerprint for item in first.comparison.items
    ]


def test_machine_readable_output_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [
            (SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "old-a")),
            (SCENARIO_B, _fail_evaluation(SCENARIO_B.scenario_id, "old-b")),
        ],
    )
    _write_run(
        root,
        "run-new",
        [
            (SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "new-a")),
            (SCENARIO_B, _fail_evaluation(SCENARIO_B.scenario_id, "new-b")),
            (SCENARIO_C, _fail_evaluation(SCENARIO_C.scenario_id, "new-c")),
        ],
    )
    create_baseline(root, run_id="run-old", out="agentcheck-baseline.json")
    assert (
        main(
            [
                "baseline",
                "check",
                str(root),
                "--baseline",
                "agentcheck-baseline.json",
                "--run-id",
                "run-new",
                "--json",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    summary = captured.err
    payload = json.loads(captured.out)
    assert "Baseline:" in summary
    assert "New regressions:" in summary
    assert "UNCHANGED:" in summary
    assert payload["schema_version"] == COMPARISON_CONTRACT_VERSION
    assert payload["new_regression_count"] == 1
    formatted = format_comparison(check_baseline(
        root, baseline_path="agentcheck-baseline.json", run_id="run-new"
    ).comparison)
    assert "2 known failures" in formatted
    assert "3 failures" in formatted


def test_create_refuses_overwrite_and_force_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [(SCENARIO_A, _pass_evaluation(SCENARIO_A.scenario_id, "old-a"))],
    )
    assert main(["baseline", "create", str(root), "--run-id", "run-old"]) == 0
    assert main(["baseline", "create", str(root), "--run-id", "run-old"]) == 2
    assert main(["baseline", "create", str(root), "--run-id", "run-old", "--force"]) == 0


def test_human_review_does_not_change_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    finding = _finding(SCENARIO_A.scenario_id)
    _write_run(
        root,
        "run-old",
        [(SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "old-a"))],
        findings=(finding,),
    )
    create_baseline(root, run_id="run-old", out="agentcheck-baseline.json")
    record_finding_review(
        root,
        run_id="run-old",
        finding_id=finding.finding_id,
        decision="accepted",
        note="Confirmed",
    )
    checked = check_baseline(
        root, baseline_path="agentcheck-baseline.json", run_id="run-old"
    )
    assert checked.exit_code == 0
    assert checked.comparison.unchanged_failure_count == 1
    dumped = json.loads((root / "agentcheck-baseline.json").read_text(encoding="utf-8"))
    assert dumped["schema_version"] == BASELINE_CONTRACT_VERSION
    assert "accepted" not in json.dumps(dumped)


def test_equivalent_runs_share_baseline_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    cases = [(SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "case-a"))]
    _write_run(root, "run-one", cases)
    _write_run(root, "run-two", cases)
    first = create_baseline(root, run_id="run-one", out="one.json")
    second = create_baseline(root, run_id="run-two", out="two.json")
    assert first.baseline.fingerprint == second.baseline.fingerprint
    assert first.baseline.created_from_run_id != second.baseline.created_from_run_id


def test_spec_mismatch_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _patch_execution_tripwires(monkeypatch)
    _write_run(
        root,
        "run-old",
        [(SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "old-a"))],
        spec=_spec("spec-old"),
    )
    _write_run(
        root,
        "run-new",
        [(SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "new-a"))],
        spec=_spec("spec-new"),
    )
    create_baseline(root, run_id="run-old", out="agentcheck-baseline.json")
    with pytest.raises(ConfigurationError, match="spec_id does not match"):
        check_baseline(root, baseline_path="agentcheck-baseline.json", run_id="run-new")


def test_baseline_does_not_invoke_replay_shrink_or_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application

    root = _target(tmp_path)
    _write_run(
        root,
        "run-old",
        [(SCENARIO_A, _fail_evaluation(SCENARIO_A.scenario_id, "old-a"))],
    )

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("baseline must not execute test, replay, shrink, or workers")

    monkeypatch.setattr(application, "replay_suite", explode)
    monkeypatch.setattr(application, "shrink_suite", explode)
    monkeypatch.setattr(application, "execute_suite", explode)
    _patch_execution_tripwires(monkeypatch)
    create_baseline(root, run_id="run-old", out="agentcheck-baseline.json")
    checked = check_baseline(
        root, baseline_path="agentcheck-baseline.json", run_id="run-old"
    )
    assert checked.exit_code == 0


def test_cli_help_documents_no_execution(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["baseline", "--help"])
    assert excinfo.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "never import the target" in help_text
    assert "never spawn a worker" in help_text
    assert "never implicitly accepted" in help_text
