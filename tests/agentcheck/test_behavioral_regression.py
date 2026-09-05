"""Run-to-run behavioral regression comparison.

Every fixture writes ordinary stored artifacts. No test executes a target,
invokes a tool, or calls a provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, TypeVar

import pytest

from agentcheck.artifacts import ArtifactStore
from agentcheck.cli import main
from agentcheck.coverage import (
    BehavioralCoverage,
    BehavioralCoverageReferenceScope,
    analyze_behavioral_coverage,
    behavioral_coverage_spec_digest,
    verify_behavioral_coverage_binding,
)
from agentcheck.coverage.analyzer import _legacy_spec_digest
from agentcheck.domain import (
    AgentProperty,
    AgentSpec,
    AssertionResult,
    CanonicalRun,
    CapabilitiesSpec,
    CaseEvaluation,
    Evidence,
    EvidenceKind,
    IdentitySpec,
    InfrastructureError,
    InspectionProvenance,
    InstructionsSpec,
    InterfaceSpec,
    ObservabilitySpec,
    RiskAuthority,
    RiskAxis,
    RunTermination,
    RuntimeSpec,
    Scenario,
    SourceKind,
    SourceReference,
    SpecEvidence,
    ToolDefinition,
    ToolRiskAssertion,
    ToolRiskSpec,
    ToolsSpec,
    Verdict,
    canonical_hash,
    utc_now,
)
from agentcheck.errors import ConfigurationError
from agentcheck.generate import build_account_support_suite
from agentcheck.generate.selection import SelectionDecision, SelectionPlan
from agentcheck.regression import (
    Comparability,
    ComparabilityCaveat,
    IncomparableReason,
    RunComparison,
    compare_stored_runs,
)
from agentcheck.regression.contract import RunComparisonItem
from agentcheck.report.load import load_stored_run


SEED = 1729


T = TypeVar("T")


def _property(value: T) -> AgentProperty[T]:
    return AgentProperty(
        value=value,
        source=SourceReference(
            kind=SourceKind.RUNTIME_INTROSPECTION, locator="test"
        ),
        confidence=1,
        evidence=(SpecEvidence(evidence_id="e", summary="test"),),
    )


def _spec(name: str = "Account Support Agent", *, spec_id: str = "spec") -> AgentSpec:
    return AgentSpec(
        spec_id=spec_id,
        identity=IdentitySpec(
            name=_property(name),
            framework=_property("OpenAI Agents SDK"),
            framework_version=_property("0.20.0"),
            provider=_property(None),
            model=_property(None),
        ),
        interface=InterfaceSpec(
            entrypoint=_property("agent.py:agent"),
            input_modalities=_property(("text",)),
            output_modalities=_property(("text",)),
            input_schema=_property(None),
            output_schema=_property(None),
            interactive=_property(True),
        ),
        instructions=InstructionsSpec(
            system=_property("system"), developer=_property(None)
        ),
        capabilities=CapabilitiesSpec(),
        tools=ToolsSpec(),
        runtime=RuntimeSpec(
            max_model_turns=_property(None),
            max_tool_calls=_property(None),
            timeout_seconds=_property(None),
            token_budget=_property(None),
            cost_budget_usd=_property(None),
        ),
        observability=ObservabilitySpec(
            supported_event_types=_property(("final_output",)),
            usage_metrics=_property(()),
            provider_request_ids=_property(False),
            source_event_links=_property(True),
        ),
        provenance=InspectionProvenance(
            inspector="test",
            inspector_version="1",
            inspected_at=utc_now(),
            target="test",
            sources=(
                SourceReference(kind=SourceKind.RUNTIME_INTROSPECTION, locator="test"),
            ),
        ),
    )


def _evaluation(
    scenario_id: str,
    run_id: str,
    verdict: Verdict,
    *,
    assertion_id: str = "a1",
) -> CaseEvaluation:
    now = utc_now()
    evidence: tuple[Evidence, ...] = ()
    if verdict is Verdict.FAIL:
        evidence = (
            Evidence(
                evidence_id="evidence-1",
                kind=EvidenceKind.OUTPUT,
                summary="Recorded final output.",
                source_ids=("output-1",),
            ),
        )
        assertions = (
            AssertionResult(
                assertion_id=assertion_id,
                criterion="declared behavior",
                result=Verdict.FAIL,
                oracle_ids=("oracle-1",),
                contradicting_evidence_ids=("evidence-1",),
                rationale="The declared behavior did not hold.",
            ),
        )
    else:
        assertions = (
            AssertionResult(
                assertion_id=assertion_id,
                criterion="declared behavior",
                result=verdict,
                oracle_ids=("oracle-1",),
                missing_evidence=(
                    ("tool_outcome",) if verdict is Verdict.INCONCLUSIVE else ()
                ),
                rationale="Recorded outcome.",
            ),
        )
    infrastructure_error = (
        InfrastructureError(
            code="worker_timeout",
            message="The worker exceeded its wall-clock budget.",
            phase="execute",
        )
        if verdict is Verdict.INFRA_ERROR
        else None
    )
    return CaseEvaluation(
        evaluation_id=f"eval-{scenario_id}",
        scenario_id=scenario_id,
        run_id=run_id,
        verdict=verdict,
        assertions=assertions,
        evidence=evidence,
        started_at=now,
        completed_at=now,
        summary=verdict.value,
        infrastructure_error=infrastructure_error,
    )


def _risk_spec(
    *,
    authorities: tuple[RiskAuthority, RiskAuthority] | None,
    state_changing: bool = True,
    destructive: bool = True,
    schema_authoritative: bool = False,
    tool_name: str = "purge_records",
) -> AgentSpec:
    tool = _property(
        ToolDefinition(
            name=tool_name,
            input_schema={"type": "object"},
            state_changing=state_changing,
            destructive=destructive,
            replaceable=True,
        )
    ).model_copy(update={"authoritative": schema_authoritative})
    assertions = (
        (
            ToolRiskAssertion(
                tool_name=tool_name,
                state_changing=RiskAxis(
                    value=state_changing, authority=authorities[0], confidence=0.9
                ),
                destructive=RiskAxis(
                    value=destructive, authority=authorities[1], confidence=0.9
                ),
                evidence=(SpecEvidence(evidence_id="risk", summary="Declared risk."),),
            ),
        )
        if authorities is not None
        else ()
    )
    return _spec().model_copy(
        update={
            "tools": ToolsSpec(items=(tool,)),
            "tool_risk": ToolRiskSpec(items=assertions),
        }
    )


def _write_run(
    root: Path,
    run_id: str,
    cases: Sequence[tuple[Scenario, Verdict]],
    *,
    spec: AgentSpec | None = None,
    seed: int = SEED,
    git_revision: str | None = "revision-a",
    selection: SelectionPlan | None = None,
    assertion_id: str = "a1",
    coverage: BehavioralCoverage | None = None,
    legacy_coverage: bool = False,
) -> Path:
    spec = spec or _spec()
    scenarios = tuple(scenario for scenario, _ in cases)
    runs = tuple(
        CanonicalRun(
            run_id=f"{run_id}-case-{index:03d}",
            scenario_id=scenario.scenario_id,
            target_id=spec.spec_id,
            started_at=utc_now(),
            ended_at=utc_now(),
            termination=RunTermination.COMPLETED,
        )
        for index, (scenario, _) in enumerate(cases)
    )
    evaluations = tuple(
        _evaluation(
            scenario.scenario_id,
            runs[index].run_id,
            verdict,
            assertion_id=assertion_id,
        )
        for index, (scenario, verdict) in enumerate(cases)
    )
    counts = {verdict.value: 0 for verdict in Verdict}
    for _, verdict in cases:
        counts[verdict.value] += 1
    reference_scope = (
        BehavioralCoverageReferenceScope.AVAILABLE_SCENARIOS_ONLY
        if selection is not None and selection.excluded_ids
        else BehavioralCoverageReferenceScope.COMPLETE
    )
    coverage = coverage or analyze_behavioral_coverage(
        spec, scenarios, reference_scope=reference_scope
    )
    if legacy_coverage:
        coverage = _as_legacy_coverage(coverage, spec)
    artifacts = ArtifactStore(root, ".agentcheck", run_id)
    artifacts.write_json("agent-spec.json", spec)
    artifacts.write_json(
        "suite.json",
        {
            "schema_version": "agentcheck.suite.v1",
            "run_id": run_id,
            "seed": seed,
            "scenarios": list(scenarios),
        },
    )
    artifacts.write_json(
        "invalid-scenarios.json",
        {"schema_version": "agentcheck.invalid_scenarios.v1", "items": []},
    )
    artifacts.write_jsonl("runs.jsonl", runs)
    artifacts.write_jsonl("evaluations.jsonl", evaluations)
    artifacts.write_json("findings.json", ())
    summary: dict[str, object] = {
        "schema_version": "agentcheck.summary.v1",
        "run_id": run_id,
        "target": str(root),
        "git_revision": git_revision,
        "suite_size": len(scenarios),
        "invalid_scenarios": 0,
        "observed_suite_pass_rate": (
            counts["PASS"] / len(cases) if cases else None
        ),
        "counts": counts,
        "finding_count": 0,
        "seed": seed,
        "behavioral_coverage": coverage.model_dump(mode="json"),
    }
    if selection is not None:
        summary["selection"] = selection.model_dump(mode="json")
        summary["excluded_by_selection"] = len(selection.excluded_ids)
    artifacts.write_json("summary.json", summary)
    artifacts.write_text("report.html", "<html><body>placeholder</body></html>")
    return artifacts.root


def _as_legacy_coverage(
    coverage: BehavioralCoverage, spec: AgentSpec
) -> BehavioralCoverage:
    # Preserve the historical payload shape, checksum and semantic rows. The
    # original F1 test below independently pins the pre-fix writer's digest.
    return BehavioralCoverage.model_validate({
        **coverage.model_dump(),
        "spec_digest": _legacy_spec_digest(spec),
        "fingerprint": "",
    })


def _selection_for(selected: Sequence[str], excluded: Sequence[str]) -> SelectionPlan:
    return SelectionPlan(
        selected_ids=tuple(selected),
        excluded_ids=tuple(excluded),
        decisions=(
            *(
                SelectionDecision(
                    scenario_id=scenario_id, selected=True, reason="selected"
                )
                for scenario_id in selected
            ),
            *(
                SelectionDecision(
                    scenario_id=scenario_id, selected=False, reason="excluded"
                )
                for scenario_id in excluded
            ),
        ),
    )


def _compare(root: Path, base: str, head: str) -> RunComparison:
    return compare_stored_runs(root, base_run_id=base, head_run_id=head).comparison


def _item(comparison: RunComparison, scenario_id: str) -> RunComparisonItem:
    return next(
        item for item in comparison.items if item.scenario_id == scenario_id
    )


@pytest.fixture
def suite() -> tuple[Scenario, ...]:
    return build_account_support_suite(seed=SEED)[:3]


def test_identical_runs_report_no_regression(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    cases = tuple((scenario, Verdict.PASS) for scenario in suite)
    _write_run(tmp_path, "base", cases)
    _write_run(tmp_path, "head", cases)

    checked = compare_stored_runs(tmp_path, base_run_id="base", head_run_id="head")

    assert checked.comparison.comparability is Comparability.COMPARABLE
    assert checked.comparison.new_regression_count == 0
    assert checked.comparison.shared_scenario_count == len(suite)
    assert all(item.category == "unchanged" for item in checked.comparison.items)
    assert checked.exit_code == 0
    # Identical scenario digests mean the same scenario set ran, so no
    # denominator caveat is warranted.
    assert ComparabilityCaveat.SCENARIO_SET_CHANGED not in checked.comparison.caveats


def test_pass_to_fail_on_the_same_identity_is_an_attributable_regression(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    _write_run(tmp_path, "base", tuple((s, Verdict.PASS) for s in suite))
    _write_run(
        tmp_path,
        "head",
        ((suite[0], Verdict.FAIL), *((s, Verdict.PASS) for s in suite[1:])),
        git_revision="revision-b",
    )

    checked = compare_stored_runs(tmp_path, base_run_id="base", head_run_id="head")
    item = _item(checked.comparison, suite[0].scenario_id)

    assert item.category == "new_regression"
    assert item.blocking is True
    assert item.base_verdict == "PASS"
    assert item.head_verdict == "FAIL"
    assert checked.comparison.new_regression_count == 1
    assert checked.exit_code == 1


def test_fail_to_pass_is_resolved_and_does_not_block(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    _write_run(
        tmp_path,
        "base",
        ((suite[0], Verdict.FAIL), *((s, Verdict.PASS) for s in suite[1:])),
    )
    _write_run(
        tmp_path,
        "head",
        tuple((s, Verdict.PASS) for s in suite),
        git_revision="revision-b",
    )

    checked = compare_stored_runs(tmp_path, base_run_id="base", head_run_id="head")

    assert _item(checked.comparison, suite[0].scenario_id).category == "resolved_failure"
    assert checked.comparison.resolved_count == 1
    assert checked.exit_code == 0


def test_infra_error_is_never_a_regression_and_never_a_pass(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    _write_run(tmp_path, "base", tuple((s, Verdict.PASS) for s in suite))
    _write_run(
        tmp_path,
        "head",
        ((suite[0], Verdict.INFRA_ERROR), *((s, Verdict.PASS) for s in suite[1:])),
        git_revision="revision-b",
    )

    checked = compare_stored_runs(tmp_path, base_run_id="base", head_run_id="head")
    item = _item(checked.comparison, suite[0].scenario_id)

    assert item.category == "infra_change"
    assert item.blocking is False
    assert checked.comparison.new_regression_count == 0
    assert checked.comparison.resolved_count == 0
    # An outcome AgentCheck could not certify is neither clean nor a regression.
    assert checked.exit_code == 2


def test_fail_to_inconclusive_does_not_resolve_a_failure(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    _write_run(
        tmp_path,
        "base",
        ((suite[0], Verdict.FAIL), *((s, Verdict.PASS) for s in suite[1:])),
    )
    _write_run(
        tmp_path,
        "head",
        ((suite[0], Verdict.INCONCLUSIVE), *((s, Verdict.PASS) for s in suite[1:])),
        git_revision="revision-b",
    )

    checked = compare_stored_runs(tmp_path, base_run_id="base", head_run_id="head")

    assert (
        _item(checked.comparison, suite[0].scenario_id).category
        == "inconclusive_change"
    )
    assert checked.comparison.resolved_count == 0
    assert checked.exit_code == 0


def test_scenarios_match_by_fingerprint_not_array_position(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    _write_run(tmp_path, "base", tuple((s, Verdict.PASS) for s in suite))
    reordered = tuple(reversed(suite))
    _write_run(
        tmp_path,
        "head",
        tuple((s, Verdict.PASS) for s in reordered),
        git_revision="revision-b",
    )

    comparison = _compare(tmp_path, "base", "head")

    assert comparison.comparability is Comparability.COMPARABLE
    assert {item.category for item in comparison.items} == {"unchanged"}
    assert comparison.shared_scenario_count == len(suite)


def test_changed_failure_signature_is_reported_separately(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    _write_run(
        tmp_path,
        "base",
        ((suite[0], Verdict.FAIL), *((s, Verdict.PASS) for s in suite[1:])),
        assertion_id="assertion-one",
    )
    _write_run(
        tmp_path,
        "head",
        ((suite[0], Verdict.FAIL), *((s, Verdict.PASS) for s in suite[1:])),
        git_revision="revision-b",
        assertion_id="assertion-two",
    )

    checked = compare_stored_runs(tmp_path, base_run_id="base", head_run_id="head")
    item = _item(checked.comparison, suite[0].scenario_id)

    assert item.category == "changed_failure"
    assert item.blocking is True
    assert item.base_failure_fingerprint != item.head_failure_fingerprint
    assert checked.exit_code == 1


def test_unchanged_failure_signature_does_not_block(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    cases = ((suite[0], Verdict.FAIL), *((s, Verdict.PASS) for s in suite[1:]))
    _write_run(tmp_path, "base", cases)
    _write_run(tmp_path, "head", cases, git_revision="revision-b")

    checked = compare_stored_runs(tmp_path, base_run_id="base", head_run_id="head")

    assert (
        _item(checked.comparison, suite[0].scenario_id).category
        == "unchanged_failure"
    )
    assert checked.comparison.unchanged_failure_count == 1
    assert checked.exit_code == 0


def test_added_and_removed_scenarios_are_reported_without_blocking(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    _write_run(tmp_path, "base", tuple((s, Verdict.PASS) for s in suite[:2]))
    _write_run(
        tmp_path,
        "head",
        tuple((s, Verdict.PASS) for s in suite[1:]),
        git_revision="revision-b",
    )

    comparison = _compare(tmp_path, "base", "head")

    assert _item(comparison, suite[0].scenario_id).category == "removed_scenario"
    assert _item(comparison, suite[2].scenario_id).category == "new_scenario"
    assert not any(item.blocking for item in comparison.items)
    assert ComparabilityCaveat.SCENARIO_SET_CHANGED in comparison.caveats


def test_a_scenario_excluded_by_selection_is_deselected_not_removed(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    # The head run never evaluated this scenario, so reporting it as removed
    # would claim a suite change that did not happen.
    _write_run(tmp_path, "base", tuple((s, Verdict.PASS) for s in suite[:2]))
    _write_run(
        tmp_path,
        "head",
        ((suite[1], Verdict.PASS),),
        git_revision="revision-b",
        selection=_selection_for(
            (suite[1].scenario_id,), (suite[0].scenario_id,)
        ),
    )

    comparison = _compare(tmp_path, "base", "head")
    item = _item(comparison, suite[0].scenario_id)

    assert item.category == "deselected_scenario"
    assert item.blocking is False
    assert ComparabilityCaveat.SELECTION_ACTIVE in comparison.caveats


def test_two_runs_without_a_shared_identity_are_incomparable(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    _write_run(tmp_path, "base", ((suite[0], Verdict.PASS),))
    _write_run(
        tmp_path,
        "head",
        ((suite[1], Verdict.FAIL),),
        git_revision="revision-b",
    )

    checked = compare_stored_runs(tmp_path, base_run_id="base", head_run_id="head")

    assert checked.comparison.comparability is Comparability.INCOMPARABLE
    assert (
        checked.comparison.incomparable_reason
        is IncomparableReason.NO_SHARED_SCENARIOS
    )
    assert checked.comparison.items == ()
    # A comparison that could not be made is never a clean result.
    assert checked.exit_code == 3
    assert "Incomparable" in checked.summary


def test_comparing_a_run_to_itself_is_incomparable(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    _write_run(tmp_path, "base", tuple((s, Verdict.PASS) for s in suite))

    checked = compare_stored_runs(tmp_path, base_run_id="base", head_run_id="base")

    assert checked.comparison.incomparable_reason is IncomparableReason.SAME_RUN
    assert checked.comparison.items == ()
    assert checked.exit_code == 3


def test_an_unchanged_source_revision_discloses_stochastic_variation(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    # Same source, different verdict: this is run-to-run variation, and the
    # comparison must not let a reader attribute it to a code change.
    _write_run(tmp_path, "base", tuple((s, Verdict.PASS) for s in suite))
    _write_run(
        tmp_path,
        "head",
        ((suite[0], Verdict.FAIL), *((s, Verdict.PASS) for s in suite[1:])),
        git_revision="revision-a",
    )

    checked = compare_stored_runs(tmp_path, base_run_id="base", head_run_id="head")

    assert (
        ComparabilityCaveat.SOURCE_REVISION_UNCHANGED in checked.comparison.caveats
    )
    assert "run-to-run variation" in checked.summary
    # It is still reported: disclosure is not suppression.
    assert checked.comparison.new_regression_count == 1


def test_a_missing_source_revision_is_disclosed(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    _write_run(tmp_path, "base", tuple((s, Verdict.PASS) for s in suite))
    _write_run(
        tmp_path,
        "head",
        tuple((s, Verdict.PASS) for s in suite),
        git_revision=None,
    )

    comparison = _compare(tmp_path, "base", "head")

    assert ComparabilityCaveat.SOURCE_REVISION_UNRECORDED in comparison.caveats


def test_a_changed_spec_is_disclosed_but_shared_identities_still_compare(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    _write_run(tmp_path, "base", tuple((s, Verdict.PASS) for s in suite))
    _write_run(
        tmp_path,
        "head",
        ((suite[0], Verdict.FAIL), *((s, Verdict.PASS) for s in suite[1:])),
        spec=_spec("Renamed Agent", spec_id="other-spec"),
        git_revision="revision-b",
    )

    checked = compare_stored_runs(tmp_path, base_run_id="base", head_run_id="head")

    assert ComparabilityCaveat.SPEC_CHANGED in checked.comparison.caveats
    # A scenario whose fingerprint is unchanged asked the same question of both
    # runs, so its verdict change is still real evidence.
    assert _item(checked.comparison, suite[0].scenario_id).category == "new_regression"
    assert "different agent" in checked.summary


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("state_changing", "destructive", "before", "after"),
    [
        (
            True, True,
            (RiskAuthority.INFERRED, RiskAuthority.INFERRED),
            (RiskAuthority.DEVELOPER_DECLARED, RiskAuthority.DEVELOPER_DECLARED),
        ),
        (
            True, False,
            (RiskAuthority.INFERRED, RiskAuthority.UNKNOWN),
            (RiskAuthority.DEVELOPER_DECLARED, RiskAuthority.UNKNOWN),
        ),
        (
            False, True,
            (RiskAuthority.UNKNOWN, RiskAuthority.INFERRED),
            (RiskAuthority.UNKNOWN, RiskAuthority.FRAMEWORK_AUTHORITATIVE),
        ),
        (
            False, False,
            (RiskAuthority.UNKNOWN, RiskAuthority.UNKNOWN),
            (RiskAuthority.DEVELOPER_DECLARED, RiskAuthority.DEVELOPER_DECLARED),
        ),
    ],
    ids=["both-axes", "state-only", "destructive-only", "declared-false-vs-unknown"],
)
def test_risk_authority_change_is_disclosed_with_identical_v1_digests(
    tmp_path: Path,
    suite: tuple[Scenario, ...],
    reverse: bool,
    state_changing: bool,
    destructive: bool,
    before: tuple[RiskAuthority, RiskAuthority],
    after: tuple[RiskAuthority, RiskAuthority],
) -> None:
    base_spec = _risk_spec(
        authorities=before, state_changing=state_changing, destructive=destructive
    )
    head_spec = _risk_spec(
        authorities=after, state_changing=state_changing, destructive=destructive
    )
    if reverse:
        base_spec, head_spec = head_spec, base_spec
    cases = tuple((scenario, Verdict.PASS) for scenario in suite)
    base_dir = _write_run(
        tmp_path, "base", cases, spec=base_spec, legacy_coverage=True
    )
    head_dir = _write_run(
        tmp_path, "head", cases, spec=head_spec, git_revision="revision-b",
        legacy_coverage=True,
    )
    base = load_stored_run(tmp_path, run_id="base")
    head = load_stored_run(tmp_path, run_id="head")
    assert base.spec.spec_id == head.spec.spec_id
    assert base.behavioral_coverage.spec_digest == head.behavioral_coverage.spec_digest
    if state_changing and destructive:
        # Recorded by the v1 writer at main@43148134, before this correction.
        assert base.behavioral_coverage.spec_digest == (
            "sha256:6d0ee564eb4c52e3c7cc166ac0615f6c817c1fee45a7288ba8d91692e9dba6db"
        )
    assert base.behavioral_coverage.families != head.behavioral_coverage.families
    original = {
        path: path.read_bytes()
        for directory in (base_dir, head_dir)
        for path in directory.iterdir()
        if path.is_file()
    }

    checked = compare_stored_runs(tmp_path, base_run_id="base", head_run_id="head")

    assert checked.comparison.caveats == (ComparabilityCaveat.SPEC_CHANGED,)
    assert checked.comparison.comparability is Comparability.COMPARABLE
    assert all(item.category == "unchanged" for item in checked.comparison.items)
    assert checked.exit_code == 0  # A disclosure is not a behavioral regression.
    assert "specification changed" in checked.summary
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.parametrize("schema_authoritative", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("recorded_coverage", [False, True])
def test_risk_authority_comparison_preserves_absent_assertion_fallback(
    tmp_path: Path,
    suite: tuple[Scenario, ...],
    schema_authoritative: bool,
    reverse: bool,
    recorded_coverage: bool,
) -> None:
    legacy = _risk_spec(authorities=None, schema_authoritative=schema_authoritative)
    matched = (
        RiskAuthority.DEVELOPER_DECLARED
        if schema_authoritative
        else RiskAuthority.INFERRED
    )
    different = (
        RiskAuthority.INFERRED
        if schema_authoritative
        else RiskAuthority.DEVELOPER_DECLARED
    )
    cases = tuple((scenario, Verdict.PASS) for scenario in suite)
    legacy_dir = _write_run(tmp_path, "legacy", cases, spec=legacy)
    # Specs predating risk provenance did not serialize an empty section.
    spec_path = legacy_dir / "agent-spec.json"
    spec_document = json.loads(spec_path.read_text(encoding="utf-8"))
    del spec_document["tool_risk"]
    spec_path.write_text(json.dumps(spec_document), encoding="utf-8")
    if not recorded_coverage:
        summary_path = legacy_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        del summary["behavioral_coverage"]
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    for run_id, authority, expected_change in (
        ("matched", matched, False), ("different", different, True)
    ):
        _write_run(
            tmp_path, run_id, cases, git_revision="revision-b",
            spec=_risk_spec(
                authorities=(authority, authority),
                schema_authoritative=schema_authoritative,
            ),
        )
        base_id, head_id = (run_id, "legacy") if reverse else ("legacy", run_id)
        checked = compare_stored_runs(
            tmp_path, base_run_id=base_id, head_run_id=head_id
        )
        changed = ComparabilityCaveat.SPEC_CHANGED in checked.comparison.caveats
        assert changed is expected_change
        assert checked.exit_code == 0


def test_risk_authority_comparison_ignores_equivalent_provenance_and_order(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    declared = _risk_spec(
        authorities=(RiskAuthority.DEVELOPER_DECLARED, RiskAuthority.DEVELOPER_DECLARED)
    )
    unknown = _risk_spec(
        authorities=(RiskAuthority.UNKNOWN, RiskAuthority.UNKNOWN),
        state_changing=False, destructive=False, tool_name="lookup_records",
    )
    base_spec = declared.model_copy(update={
        "tools": ToolsSpec(items=declared.tools.items + unknown.tools.items),
        "tool_risk": ToolRiskSpec(items=declared.tool_risk.items + unknown.tool_risk.items),
    })
    declared_assertion = declared.tool_risk.items[0]
    unknown_assertion = unknown.tool_risk.items[0]
    changed_assertions = tuple(
        assertion.model_copy(update={
            "state_changing": assertion.state_changing.model_copy(update={
                "authority": authority, "confidence": confidence,
            }),
            "destructive": assertion.destructive.model_copy(update={
                "authority": authority, "confidence": confidence,
            }),
            "evidence": (SpecEvidence(evidence_id="changed", summary="Different provenance."),),
            "conflicts": ("Different recorded conflict.",),
        })
        for assertion, authority, confidence in (
            (unknown_assertion, RiskAuthority.INFERRED, 0.5),
            (declared_assertion, RiskAuthority.FRAMEWORK_AUTHORITATIVE, 1.0),
        )
    )
    head_spec = base_spec.model_copy(update={
        "tools": ToolsSpec(items=tuple(reversed(base_spec.tools.items))),
        "tool_risk": ToolRiskSpec(items=changed_assertions),
    })
    assert analyze_behavioral_coverage(
        base_spec, suite
    ) == analyze_behavioral_coverage(head_spec, suite)
    cases = tuple((scenario, Verdict.PASS) for scenario in suite)
    _write_run(tmp_path, "base", cases, spec=base_spec)
    _write_run(tmp_path, "head", cases, spec=head_spec, git_revision="revision-b")

    checked = compare_stored_runs(tmp_path, base_run_id="base", head_run_id="head")

    assert checked.comparison.caveats == ()
    assert checked.exit_code == 0


def test_a_changed_seed_is_disclosed(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    _write_run(tmp_path, "base", tuple((s, Verdict.PASS) for s in suite))
    _write_run(
        tmp_path,
        "head",
        tuple((s, Verdict.PASS) for s in suite),
        seed=SEED + 1,
        git_revision="revision-b",
    )

    comparison = _compare(tmp_path, "base", "head")

    assert ComparabilityCaveat.SEED_CHANGED in comparison.caveats


def test_every_summary_states_that_a_verdict_pair_is_not_determinism(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    cases = tuple((scenario, Verdict.PASS) for scenario in suite)
    _write_run(tmp_path, "base", cases)
    _write_run(tmp_path, "head", cases)
    _write_run(tmp_path, "unrelated", ((suite[0], Verdict.PASS),))

    clean = compare_stored_runs(tmp_path, base_run_id="base", head_run_id="head")
    incomparable = compare_stored_runs(
        tmp_path, base_run_id="base", head_run_id="base"
    )

    for summary in (clean.summary, incomparable.summary):
        assert "not proof that a behavior is stable" in summary
        assert "not model output" in summary


def test_comparison_fingerprint_detects_altered_content(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    cases = tuple((scenario, Verdict.PASS) for scenario in suite)
    _write_run(tmp_path, "base", cases)
    _write_run(tmp_path, "head", cases)
    comparison = _compare(tmp_path, "base", "head")
    document = comparison.model_dump(mode="json")
    document["new_regression_count"] = 5

    with pytest.raises(ValueError, match="fingerprint does not match"):
        RunComparison.model_validate_json(json.dumps(document))


def test_only_a_regression_category_may_block(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    with pytest.raises(ValueError, match="may be marked blocking"):
        RunComparison(
            base_run_id="base",
            head_run_id="head",
            comparability=Comparability.COMPARABLE,
            base_scenario_count=1,
            head_scenario_count=1,
            shared_scenario_count=1,
            base_failure_count=0,
            head_failure_count=0,
            new_regression_count=0,
            changed_failure_count=0,
            resolved_count=0,
            unchanged_failure_count=0,
            uncertifiable_count=0,
            items=(
                RunComparisonItem(
                    category="unchanged",
                    scenario_id="s",
                    fingerprint="sha256:abc",
                    blocking=True,
                ),
            ),
        )


def test_an_incomparable_result_classifies_nothing() -> None:
    with pytest.raises(ValueError, match="must not classify any scenario"):
        RunComparison(
            base_run_id="base",
            head_run_id="head",
            comparability=Comparability.INCOMPARABLE,
            incomparable_reason=IncomparableReason.SAME_RUN,
            base_scenario_count=1,
            head_scenario_count=1,
            shared_scenario_count=1,
            base_failure_count=0,
            head_failure_count=0,
            new_regression_count=0,
            changed_failure_count=0,
            resolved_count=0,
            unchanged_failure_count=0,
            uncertifiable_count=0,
            items=(
                RunComparisonItem(
                    category="unchanged",
                    scenario_id="s",
                    fingerprint="sha256:abc",
                ),
            ),
        )


def test_credential_shaped_identifiers_are_redacted() -> None:
    item = RunComparisonItem(
        category="unchanged",
        scenario_id="sk-secretvalue123",
        fingerprint="sha256:abc",
    )

    assert "sk-secretvalue123" not in item.scenario_id
    assert "[REDACTED]" in item.scenario_id


def test_comparison_requires_exactly_one_head_selector(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    _write_run(tmp_path, "base", tuple((s, Verdict.PASS) for s in suite))

    with pytest.raises(ConfigurationError, match="one of --head or --latest"):
        compare_stored_runs(tmp_path, base_run_id="base")
    with pytest.raises(ConfigurationError, match="only one of --head and --latest"):
        compare_stored_runs(
            tmp_path, base_run_id="base", head_run_id="base", latest=True
        )


def test_cli_compare_reports_a_regression_and_emits_the_contract(
    tmp_path: Path,
    suite: tuple[Scenario, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_run(tmp_path, "base", tuple((s, Verdict.PASS) for s in suite))
    _write_run(
        tmp_path,
        "head",
        ((suite[0], Verdict.FAIL), *((s, Verdict.PASS) for s in suite[1:])),
        git_revision="revision-b",
    )

    code = main(
        [
            "compare",
            str(tmp_path),
            "--base",
            "base",
            "--head",
            "head",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    # The human summary goes to stderr so stdout stays a clean contract stream.
    document = json.loads(captured.out)
    assert document["schema_version"] == "agentcheck.run_comparison.v1"
    assert document["new_regression_count"] == 1
    assert document["comparability"] == "comparable"
    assert "New regressions:" in captured.err


def test_cli_compare_exits_three_when_runs_are_incomparable(
    tmp_path: Path,
    suite: tuple[Scenario, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_run(tmp_path, "base", ((suite[0], Verdict.PASS),))
    _write_run(tmp_path, "head", ((suite[1], Verdict.PASS),))

    code = main(["compare", str(tmp_path), "--base", "base", "--head", "head"])

    assert code == 3
    assert "Incomparable" in capsys.readouterr().out


def test_cli_compare_accepts_latest_as_the_head_run(
    tmp_path: Path,
    suite: tuple[Scenario, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = tuple((scenario, Verdict.PASS) for scenario in suite)
    _write_run(tmp_path, "base", cases)
    _write_run(tmp_path, "head", cases)

    code = main(["compare", str(tmp_path), "--base", "base", "--latest"])

    assert code == 0
    assert "Base run: base" in capsys.readouterr().out


def test_comparison_never_reads_the_configured_suite_file(
    tmp_path: Path, suite: tuple[Scenario, ...]
) -> None:
    # A frozen suite written after both runs must not become a comparison
    # input; suite identity comes from what each run recorded for itself.
    cases = tuple((scenario, Verdict.PASS) for scenario in suite)
    _write_run(tmp_path, "base", cases)
    _write_run(tmp_path, "head", cases)
    before = _compare(tmp_path, "base", "head")

    loaded = load_stored_run(tmp_path, run_id="base")
    assert loaded.frozen_suite is None

    after = _compare(tmp_path, "base", "head")
    assert after == before


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("axes", [(0,), (1,), (0, 1)])
@pytest.mark.parametrize("risk_value", [False, True])
def test_selected_v2_coverage_binds_each_effective_authority_axis(
    suite: tuple[Scenario, ...], reverse: bool, axes: tuple[int, ...],
    risk_value: bool,
) -> None:
    non_authoritative = RiskAuthority.INFERRED if risk_value else RiskAuthority.UNKNOWN
    before = [non_authoritative, non_authoritative]
    after = before.copy()
    for axis in axes:
        after[axis] = RiskAuthority.DEVELOPER_DECLARED
    specs = [
        _risk_spec(
            authorities=(values[0], values[1]),
            state_changing=risk_value, destructive=risk_value,
        )
        for values in (before, after)
    ]
    if reverse:
        specs.reverse()
    actual, reference = suite[:1], suite[:2]
    coverage = analyze_behavioral_coverage(
        specs[0], actual, reference_scenarios=reference
    )
    original = coverage.canonical_json()
    assert coverage.spec_digest.startswith("coverage-spec.v2:sha256:")
    assert _legacy_spec_digest(specs[0]) == _legacy_spec_digest(specs[1])
    verify_behavioral_coverage_binding(coverage, specs[0], actual)
    with pytest.raises(ValueError, match="spec_digest"):
        verify_behavioral_coverage_binding(coverage, specs[1], actual)
    assert coverage.reference_scenario_count == 2
    assert coverage.canonical_json() == original


def test_v2_digest_pins_algorithm_and_historical_projection() -> None:
    spec = _risk_spec(authorities=(RiskAuthority.DEVELOPER_DECLARED,) * 2)
    expected = canonical_hash({
        "algorithm": "coverage-spec.v2",
        "legacy_spec_digest": "sha256:6d0ee564eb4c52e3c7cc166ac0615f6c817c1fee45a7288ba8d91692e9dba6db",
        "risk_authority": [["purge_records", True, True]],
    })
    assert behavioral_coverage_spec_digest(spec) == f"coverage-spec.v2:{expected}"


def test_v2_authority_binding_ignores_only_equivalent_metadata() -> None:
    legacy_fallback = _risk_spec(authorities=None, schema_authoritative=True)
    declared = _risk_spec(
        authorities=(RiskAuthority.DEVELOPER_DECLARED,) * 2,
        schema_authoritative=True,
    )
    framework = _risk_spec(
        authorities=(RiskAuthority.FRAMEWORK_AUTHORITATIVE,) * 2,
        schema_authoritative=True,
    )
    assert behavioral_coverage_spec_digest(legacy_fallback) == (
        behavioral_coverage_spec_digest(declared)
    ) == behavioral_coverage_spec_digest(framework)
    unknown = _risk_spec(
        authorities=(RiskAuthority.UNKNOWN,) * 2, state_changing=False, destructive=False
    )
    inferred = _risk_spec(
        authorities=(RiskAuthority.INFERRED,) * 2, state_changing=False, destructive=False
    )
    declared_false = _risk_spec(
        authorities=(RiskAuthority.DEVELOPER_DECLARED,) * 2,
        state_changing=False, destructive=False,
    )
    assert behavioral_coverage_spec_digest(unknown) == behavioral_coverage_spec_digest(inferred)
    assert behavioral_coverage_spec_digest(unknown) != behavioral_coverage_spec_digest(declared_false)
    orphan = inferred.tool_risk.items[0].model_copy(update={"tool_name": "not-declared"})
    with_orphan = inferred.model_copy(update={"tool_risk": ToolRiskSpec(
        items=(*inferred.tool_risk.items, orphan)
    )})
    assert behavioral_coverage_spec_digest(inferred) == behavioral_coverage_spec_digest(with_orphan)
    doubled = inferred.model_copy(update={"tools": ToolsSpec(
        items=(*inferred.tools.items, *inferred.tools.items)
    )})
    assert behavioral_coverage_spec_digest(inferred) != behavioral_coverage_spec_digest(doubled)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("changed_authority", [False, True])
def test_mixed_digest_algorithms_compare_validated_semantics(
    tmp_path: Path, suite: tuple[Scenario, ...], reverse: bool,
    changed_authority: bool,
) -> None:
    first = _risk_spec(authorities=(RiskAuthority.INFERRED,) * 2)
    second = _risk_spec(authorities=(
        RiskAuthority.DEVELOPER_DECLARED if changed_authority else RiskAuthority.INFERRED,
    ) * 2)
    cases = tuple((scenario, Verdict.PASS) for scenario in suite)
    _write_run(tmp_path, "legacy", cases, spec=first, legacy_coverage=True)
    _write_run(tmp_path, "current", cases, spec=second, git_revision="revision-b")
    original = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    base, head = ("current", "legacy") if reverse else ("legacy", "current")
    result = compare_stored_runs(tmp_path, base_run_id=base, head_run_id=head)
    assert (ComparabilityCaveat.SPEC_CHANGED in result.comparison.caveats) is changed_authority
    assert result.exit_code == 0
    assert all(item.category == "unchanged" for item in result.comparison.items)
    assert {path: path.read_bytes() for path in original} == original


def test_legacy_reference_rederivation_preserves_checksum_and_denominator(
    suite: tuple[Scenario, ...],
) -> None:
    spec = _risk_spec(authorities=(RiskAuthority.INFERRED,) * 2)
    actual, reference = suite[:1], suite[:2]
    coverage = _as_legacy_coverage(analyze_behavioral_coverage(
        spec, actual, reference_scenarios=reference, suite_fingerprint="frozen-reference"
    ), spec)
    original = coverage.canonical_json()
    with pytest.raises(ValueError, match="legacy_spec_digest_requires_complete_reference"):
        verify_behavioral_coverage_binding(coverage, spec, actual)
    verify_behavioral_coverage_binding(
        coverage, spec, actual, reference_scenarios=tuple(reversed(reference)),
        suite_fingerprint="frozen-reference",
    )
    for wrong_reference in (actual, (reference[0], reference[0]), suite):
        with pytest.raises(ValueError, match="reference"):
            verify_behavioral_coverage_binding(
                coverage, spec, actual, reference_scenarios=wrong_reference
            )
    with pytest.raises(ValueError, match="suite_fingerprint"):
        verify_behavioral_coverage_binding(
            coverage, spec, actual, reference_scenarios=reference,
            suite_fingerprint="wrong-frozen-reference",
        )
    substituted = _risk_spec(authorities=(RiskAuthority.DEVELOPER_DECLARED,) * 2)
    with pytest.raises(ValueError, match="derived_coverage"):
        verify_behavioral_coverage_binding(
            coverage, substituted, actual, reference_scenarios=reference
        )
    assert coverage.reference_scenario_count == 2
    assert coverage.canonical_json() == original


@pytest.mark.parametrize("algorithm", [
    "coverage-spec.v3:sha256:", "coverage-spec.v2:SHA256:",
    "coverage-spec.v2:", "SHA256:", "sha256:", "",
])
def test_unknown_or_malformed_digest_algorithm_never_falls_back(
    tmp_path: Path, suite: tuple[Scenario, ...], algorithm: str,
) -> None:
    spec = _risk_spec(authorities=(RiskAuthority.DEVELOPER_DECLARED,) * 2)
    coverage = analyze_behavioral_coverage(spec, suite)
    suffix = "f" * (63 if algorithm == "sha256:" else 64)
    malformed = BehavioralCoverage.model_validate({
        **coverage.model_dump(), "spec_digest": algorithm + suffix, "fingerprint": "",
    })
    directory = _write_run(
        tmp_path, "unknown-algorithm", tuple((s, Verdict.PASS) for s in suite),
        spec=spec, coverage=malformed,
    )
    original = {path: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    with pytest.raises(ConfigurationError, match="spec_digest algorithm"):
        load_stored_run(tmp_path, run_id="unknown-algorithm")
    assert {path: path.read_bytes() for path in original} == original


def test_v2_cannot_downgrade_or_ignore_a_corrupt_checksum(
    suite: tuple[Scenario, ...],
) -> None:
    spec = _risk_spec(authorities=(RiskAuthority.DEVELOPER_DECLARED,) * 2)
    actual, reference = suite[:1], suite[:2]
    coverage = analyze_behavioral_coverage(spec, actual, reference_scenarios=reference)
    downgrade = _as_legacy_coverage(coverage, spec)
    with pytest.raises(ValueError, match="legacy_spec_digest_requires_complete_reference"):
        verify_behavioral_coverage_binding(downgrade, spec, actual)
    wrong_digest = BehavioralCoverage.model_validate({
        **coverage.model_dump(),
        "spec_digest": _legacy_spec_digest(spec).replace("sha256:", "coverage-spec.v2:sha256:"),
        "fingerprint": "",
    })
    with pytest.raises(ValueError, match="spec_digest"):
        verify_behavioral_coverage_binding(wrong_digest, spec, actual)
    # model_copy deliberately bypasses contract validation: the binding gate
    # must still independently verify an already-instantiated object's checksum.
    corrupt = coverage.model_copy(update={"fingerprint": "sha256:" + "0" * 64})
    with pytest.raises(ValueError, match="fingerprint"):
        verify_behavioral_coverage_binding(corrupt, spec, actual)


@pytest.mark.parametrize("consumer", ["load", "render", "baseline", "baseline-check", "finding", "compare"])
@pytest.mark.parametrize("failure", ["legacy-selected", "authority-substitution"])
def test_unverifiable_selected_coverage_is_rejected_by_all_stored_consumers(
    tmp_path: Path, suite: tuple[Scenario, ...], consumer: str, failure: str,
) -> None:
    from agentcheck.baseline.service import check_baseline, create_baseline
    from agentcheck.report.load import render_stored_run
    from agentcheck.review.service import record_finding_review

    spec = _risk_spec(authorities=(RiskAuthority.INFERRED,) * 2)
    actual, reference = suite[:1], suite[:2]
    coverage = analyze_behavioral_coverage(spec, actual, reference_scenarios=reference)
    directory = _write_run(
        tmp_path, "selected", ((actual[0], Verdict.PASS),), spec=spec,
        coverage=coverage, legacy_coverage=failure == "legacy-selected",
        selection=_selection_for([actual[0].scenario_id], [reference[1].scenario_id]),
    )
    if failure == "authority-substitution":
        assert load_stored_run(tmp_path, run_id="selected").behavioral_coverage == coverage
        substituted = _risk_spec(authorities=(RiskAuthority.DEVELOPER_DECLARED,) * 2)
        (directory / "agent-spec.json").write_text(substituted.canonical_json())
    if consumer == "baseline-check":
        _write_run(tmp_path, "baseline-source", ((actual[0], Verdict.PASS),), spec=spec)
        create_baseline(tmp_path, run_id="baseline-source", out="comparison-baseline.json")
    original = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    reason = "legacy_spec_digest_requires_complete_reference" if failure == "legacy-selected" else "spec_digest"
    with pytest.raises(ConfigurationError, match=reason):
        if consumer == "load":
            load_stored_run(tmp_path, run_id="selected")
        elif consumer == "render":
            render_stored_run(tmp_path, run_id="selected")
        elif consumer == "baseline":
            create_baseline(tmp_path, run_id="selected")
        elif consumer == "baseline-check":
            check_baseline(
                tmp_path, baseline_path="comparison-baseline.json", run_id="selected"
            )
        elif consumer == "finding":
            record_finding_review(
                tmp_path, run_id="selected", finding_id="not-reached", decision="accepted"
            )
        else:
            compare_stored_runs(tmp_path, base_run_id="selected", head_run_id="selected")
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == original


def test_selected_summary_originally_without_coverage_remains_explicitly_limited(
    tmp_path: Path, suite: tuple[Scenario, ...],
) -> None:
    spec = _risk_spec(authorities=None, schema_authoritative=True)
    actual, reference = suite[:1], suite[:2]
    directory = _write_run(
        tmp_path, "pre-coverage-summary", ((actual[0], Verdict.PASS),), spec=spec,
        selection=_selection_for([actual[0].scenario_id], [reference[1].scenario_id]),
    )
    # Construct the historical pre-coverage summary shape, not a recovery path
    # for a recorded larger denominator; the product never erases this field.
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    summary.pop("behavioral_coverage")
    path.write_text(json.dumps(summary))
    original = {path: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    coverage = load_stored_run(tmp_path, run_id="pre-coverage-summary").behavioral_coverage
    assert coverage.reference_scope is BehavioralCoverageReferenceScope.AVAILABLE_SCENARIOS_ONLY
    assert coverage.reference_scenario_count == coverage.scenario_count == 1
    assert coverage.spec_digest.startswith("coverage-spec.v2:sha256:")
    assert {path: path.read_bytes() for path in original} == original
