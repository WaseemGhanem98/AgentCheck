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
    BehavioralCoverageReferenceScope,
    analyze_behavioral_coverage,
)
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
    RunTermination,
    RuntimeSpec,
    Scenario,
    SourceKind,
    SourceReference,
    SpecEvidence,
    ToolsSpec,
    Verdict,
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
    coverage = analyze_behavioral_coverage(
        spec, scenarios, reference_scope=reference_scope
    )
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
