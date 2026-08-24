from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agentcheck.artifacts import ArtifactStore
from agentcheck.coverage import (
    BehavioralCoverage,
    BehavioralCoverageReferenceScope,
    BehavioralCoverageStatus,
    BehavioralDimension,
    analyze_behavioral_coverage,
    verify_behavioral_coverage_binding,
)
from agentcheck.coverage.contract import MAX_COVERAGE_DETAILS
from agentcheck.domain import (
    AgentProperty,
    AgentSpec,
    CapabilitiesSpec,
    ConversationRole,
    ConversationTurn,
    FaultType,
    GuardrailsSpec,
    IdentitySpec,
    InjectedFault,
    InspectionProvenance,
    InstructionsSpec,
    InterfaceSpec,
    ObservabilitySpec,
    OracleProvenance,
    OracleStrength,
    OutputCriterion,
    PostconditionOperator,
    OutputCriterionKind,
    PoliciesSpec,
    ResourceBudgets,
    RuntimeSpec,
    Scenario,
    SimulatedToolOutcome,
    SimulatedToolStatus,
    SourceKind,
    SourceReference,
    StatePostcondition,
    SpecEvidence,
    ToolBehaviorConstraint,
    ToolDefinition,
    ToolFixture,
    ToolPoliciesSpec,
    ToolPolicy,
    ToolsSpec,
    TrajectoryConstraint,
    TrajectoryConstraintKind,
    WorldStateEffect,
    WorkflowsSpec,
)
from agentcheck.generate.boundaries import build_outcome_variant_cases
from agentcheck.generate.suite import (
    CaseLineage,
    CaseOrigin,
    FrozenCase,
    FrozenSuite,
    GeneratorProvenance,
    SuiteCoverage,
)
from agentcheck.privacy import redact_artifact
from agentcheck.runner.tool_gateway import ToolGateway


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)
ORACLE_ID = "declared-contract"


def _property(value: Any, *, authoritative: bool = True) -> AgentProperty[Any]:
    return AgentProperty(
        value=value,
        source=SourceReference(
            kind=SourceKind.RUNTIME_INTROSPECTION,
            locator="tests/target.py:1",
        ),
        confidence=1.0,
        evidence=(
            SpecEvidence(
                evidence_id="inspection-evidence",
                summary="Observed in the declared target contract.",
            ),
        ),
        inferred=not authoritative,
        authoritative=authoritative,
    )


def _tool(
    name: str,
    *,
    state_changing: bool = False,
    destructive: bool = False,
    input_schema: dict[str, Any] | None = None,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Declared tool {name}.",
        input_schema=input_schema or {"type": "object"},
        state_changing=state_changing,
        replaceable=True,
        destructive=destructive,
    )


def _spec(
    *tools: tuple[ToolDefinition, bool],
    policies: tuple[tuple[ToolPolicy, bool], ...] = (),
    spec_id: str = "coverage-spec",
) -> AgentSpec:
    return AgentSpec(
        spec_id=spec_id,
        identity=IdentitySpec(
            name=_property("Coverage target"),
            framework=_property("custom"),
            framework_version=_property(None, authoritative=False),
            provider=_property(None, authoritative=False),
            model=_property(None, authoritative=False),
        ),
        interface=InterfaceSpec(
            entrypoint=_property("target:agent"),
            input_modalities=_property(("text",)),
            output_modalities=_property(("text",)),
            input_schema=_property(None, authoritative=False),
            output_schema=_property(None, authoritative=False),
            interactive=_property(True),
        ),
        instructions=InstructionsSpec(
            system=_property(None, authoritative=False),
            developer=_property(None, authoritative=False),
        ),
        capabilities=CapabilitiesSpec(),
        tools=ToolsSpec(
            items=tuple(
                _property(tool, authoritative=authoritative)
                for tool, authoritative in tools
            )
        ),
        tool_policies=ToolPoliciesSpec(
            items=tuple(
                _property(policy, authoritative=authoritative)
                for policy, authoritative in policies
            )
        ),
        guardrails=GuardrailsSpec(),
        workflows=WorkflowsSpec(),
        policies=PoliciesSpec(),
        runtime=RuntimeSpec(
            max_model_turns=_property(8),
            max_tool_calls=_property(20),
            timeout_seconds=_property(30.0),
            token_budget=_property(None, authoritative=False),
            cost_budget_usd=_property(None, authoritative=False),
        ),
        observability=ObservabilitySpec(
            supported_event_types=_property(("tool_attempt", "tool_result")),
            usage_metrics=_property(()),
            provider_request_ids=_property(False),
            source_event_links=_property(True),
        ),
        provenance=InspectionProvenance(
            inspector="coverage-test",
            inspector_version="1",
            inspected_at=NOW,
            target="tests/target.py",
            sources=(
                SourceReference(
                    kind=SourceKind.RUNTIME_INTROSPECTION,
                    locator="tests/target.py:1",
                ),
            ),
        ),
    )


def _oracle() -> OracleProvenance:
    return OracleProvenance(
        oracle_id=ORACLE_ID,
        strength=OracleStrength.TOOL_CONTRACT,
        source="Declared tool and scenario contract.",
        confidence=1.0,
        evidence_ids=("declared-evidence",),
        supports_hard_failure=True,
    )


def _required(
    tool_name: str,
    *,
    criterion_id: str = "required-call",
    confirmation: bool = False,
    arguments_match: dict[str, Any] | None = None,
) -> ToolBehaviorConstraint:
    return ToolBehaviorConstraint(
        criterion_id=criterion_id,
        tool_name=tool_name,
        min_calls=1,
        confirmation_required_before_call=confirmation,
        arguments_match=arguments_match or {},
        oracle_ids=(ORACLE_ID,),
    )


def _forbidden(
    tool_name: str,
    *,
    criterion_id: str = "forbidden-call",
    confirmation: bool = False,
) -> ToolBehaviorConstraint:
    return ToolBehaviorConstraint(
        criterion_id=criterion_id,
        tool_name=tool_name,
        min_calls=0,
        max_calls=0,
        confirmation_required_before_call=confirmation,
        oracle_ids=(ORACLE_ID,),
    )


def _fixture(
    tool_name: str,
    status: SimulatedToolStatus,
    *,
    fixture_id: str = "fixture",
    invocation_index: int | None = None,
    error_code: str | None = None,
    arguments_match: dict[str, Any] | None = None,
    priority: int = 0,
    latency_ms: float = 0.0,
    state_effects: tuple[WorldStateEffect, ...] = (),
) -> ToolFixture:
    return ToolFixture(
        fixture_id=fixture_id,
        tool_name=tool_name,
        invocation_index=invocation_index,
        arguments_match=arguments_match or {},
        priority=priority,
        outcome=SimulatedToolOutcome(
            status=status,
            result={"ok": True} if status is SimulatedToolStatus.SUCCESS else None,
            error_code=error_code,
            error_message=(
                "controlled failure"
                if status in {SimulatedToolStatus.ERROR, SimulatedToolStatus.TIMEOUT}
                else None
            ),
            latency_ms=latency_ms,
            state_effects=state_effects,
        ),
    )


def _trajectory(
    kind: TrajectoryConstraintKind,
    tool_name: str,
    *,
    criterion_id: str = "trajectory",
    extra: dict[str, Any] | None = None,
) -> TrajectoryConstraint:
    return TrajectoryConstraint(
        criterion_id=criterion_id,
        kind=kind,
        description="Explicit executable trajectory constraint.",
        parameters={"tool_name": tool_name, **(extra or {})},
        oracle_ids=(ORACLE_ID,),
    )


def _output(
    kind: OutputCriterionKind,
    *,
    criterion_id: str = "output",
    parameters: dict[str, Any] | None = None,
) -> OutputCriterion:
    return OutputCriterion(
        criterion_id=criterion_id,
        kind=kind,
        description="Explicit output behavior criterion.",
        parameters=parameters or {},
        oracle_ids=(ORACLE_ID,),
    )


def _scenario(
    scenario_id: str,
    *,
    fixtures: tuple[ToolFixture, ...] = (),
    injected_faults: tuple[InjectedFault, ...] = (),
    required: tuple[ToolBehaviorConstraint, ...] = (),
    forbidden: tuple[ToolBehaviorConstraint, ...] = (),
    allowed: tuple[ToolBehaviorConstraint, ...] = (),
    trajectory: tuple[TrajectoryConstraint, ...] = (),
    outputs: tuple[OutputCriterion, ...] = (),
    postconditions: tuple[StatePostcondition, ...] = (),
    confirmed: bool = False,
    dimension_tags: tuple[str, ...] = ("test:behavioral-coverage",),
    seed: int = 1,
    resource_budgets: ResourceBudgets | None = None,
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        title=f"Coverage case {scenario_id}",
        conversation_turns=(
            ConversationTurn(
                turn_id=f"turn-{scenario_id}",
                role=ConversationRole.USER,
                content="Exercise the declared behavior.",
                metadata={"explicit_confirmation": True} if confirmed else {},
            ),
        ),
        initial_world_state={"records": {"one": {"status": "unchanged"}}},
        tool_fixtures=fixtures,
        injected_faults=injected_faults,
        required_tool_behavior=required,
        allowed_tool_behavior=allowed,
        forbidden_tool_behavior=forbidden,
        trajectory_constraints=trajectory,
        output_criteria=outputs,
        expected_postconditions=postconditions,
        dimension_tags=dimension_tags,
        oracle_provenance=(_oracle(),),
        resource_budgets=resource_budgets or ResourceBudgets(),
        generation_seed=seed,
    )


def _family(report: BehavioralCoverage, dimension: BehavioralDimension):
    return next(family for family in report.families if family.dimension is dimension)


def _requirement(
    report: BehavioralCoverage,
    dimension: BehavioralDimension,
    subject: str,
):
    family = _family(report, dimension)
    return next(item for item in family.requirements if item.subject == subject)


def test_no_declared_tools_has_no_applicable_risks() -> None:
    report = analyze_behavioral_coverage(_spec(), ())

    assert tuple(family.dimension for family in report.families) == tuple(
        BehavioralDimension
    )
    assert all(family.applicable == 0 for family in report.families)
    assert all(family.unknown == family.unsupported == 0 for family in report.families)


def test_success_failure_timeout_and_multiple_tools_are_explicit() -> None:
    first = _tool("write_record", state_changing=True)
    second = _tool("read_record")
    success = _scenario(
        "success",
        fixtures=(_fixture(first.name, SimulatedToolStatus.SUCCESS),),
        required=(_required(first.name),),
    )
    failure = _scenario(
        "failure",
        fixtures=(
            _fixture(first.name, SimulatedToolStatus.ERROR, error_code="declined"),
        ),
        required=(_required(first.name),),
        outputs=(_output(OutputCriterionKind.ACKNOWLEDGES_TOOL_ERROR),),
        seed=2,
    )
    timeout = _scenario(
        "timeout",
        fixtures=(
            _fixture(first.name, SimulatedToolStatus.TIMEOUT, error_code="timeout"),
        ),
        required=(_required(first.name),),
        outputs=(_output(OutputCriterionKind.ACKNOWLEDGES_TOOL_ERROR),),
        seed=3,
    )

    report = analyze_behavioral_coverage(
        _spec((second, True), (first, True)), (timeout, success, failure)
    )

    for dimension in (
        BehavioralDimension.SUCCESS_PATH,
        BehavioralDimension.FAILURE_HANDLING,
        BehavioralDimension.TIMEOUT_HANDLING,
    ):
        family = _family(report, dimension)
        assert family.applicable == 2
        assert family.covered == 1
        assert family.missing == 1
        assert _requirement(report, dimension, "tool:write_record").status is (
            BehavioralCoverageStatus.COVERED
        )


def test_destructive_applicability_is_authoritative_and_never_name_guessed() -> None:
    declared = _tool("mutate", state_changing=True, destructive=True)
    suspicious = _tool("delete_every_user", state_changing=True, destructive=True)
    report = analyze_behavioral_coverage(
        _spec((suspicious, False), (declared, True)), ()
    )

    for dimension in (
        BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
        BehavioralDimension.DUPLICATE_ACTION,
        BehavioralDimension.AMBIGUOUS_OUTCOME,
        BehavioralDimension.RETRY_CONTROL,
    ):
        assert _requirement(report, dimension, "tool:delete_every_user").status is (
            BehavioralCoverageStatus.UNKNOWN
        )
        assert _requirement(report, dimension, "tool:mutate").status is (
            BehavioralCoverageStatus.MISSING
        )


def test_global_or_other_metadata_does_not_create_tool_semantics() -> None:
    other = _scenario(
        "global-metadata",
        trajectory=(
            _trajectory(TrajectoryConstraintKind.OTHER, "delete_user"),
        ),
        outputs=(_output(OutputCriterionKind.NO_FABRICATED_SUCCESS),),
    )

    report = analyze_behavioral_coverage(_spec(), (other,))

    assert _family(report, BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE).applicable == 0
    assert _family(report, BehavioralDimension.DUPLICATE_ACTION).applicable == 0


@pytest.mark.parametrize(
    ("status", "dimension"),
    (
        (SimulatedToolStatus.ERROR, BehavioralDimension.FAILURE_HANDLING),
        (SimulatedToolStatus.TIMEOUT, BehavioralDimension.TIMEOUT_HANDLING),
    ),
)
@pytest.mark.parametrize("multiple_tools", (False, True))
def test_unrelated_state_postcondition_is_not_a_failure_handling_oracle(
    status: SimulatedToolStatus,
    dimension: BehavioralDimension,
    multiple_tools: bool,
) -> None:
    focal = _tool("act")
    other = _tool("observe")
    fixtures = [
        _fixture("act", status, error_code="controlled"),
    ]
    required = [_required("act")]
    declared_tools = [(focal, True)]
    if multiple_tools:
        fixtures.append(
            _fixture(
                "observe",
                SimulatedToolStatus.SUCCESS,
                fixture_id="observe-success",
            )
        )
        required.append(_required("observe", criterion_id="require-observe"))
        declared_tools.append((other, True))
    scenario = _scenario(
        f"unrelated-state-{status.value}-{multiple_tools}",
        fixtures=tuple(fixtures),
        required=tuple(required),
        postconditions=(
            StatePostcondition(
                criterion_id="unrelated-state",
                path="records.one.status",
                operator=PostconditionOperator.UNCHANGED,
                oracle_ids=(ORACLE_ID,),
            ),
        ),
    )

    report = analyze_behavioral_coverage(_spec(*declared_tools), (scenario,))
    requirement = _requirement(report, dimension, "tool:act")
    assert requirement.status is BehavioralCoverageStatus.PARTIAL
    assert requirement.reason_code.endswith("without_handling_oracle")


def test_fabricated_success_requires_failure_stimulus_and_configured_terms() -> None:
    tool = _tool("mutate", state_changing=True)
    fixtures = (
        _fixture("mutate", SimulatedToolStatus.ERROR, error_code="failed"),
    )
    required = (_required("mutate"),)
    unconfigured = _scenario(
        "unconfigured",
        fixtures=fixtures,
        required=required,
        outputs=(_output(OutputCriterionKind.NO_FABRICATED_SUCCESS),),
    )
    configured = _scenario(
        "configured",
        fixtures=fixtures,
        required=required,
        outputs=(
            _output(
                OutputCriterionKind.NO_FABRICATED_SUCCESS,
                parameters={"success_terms": ["completed", "succeeded"]},
            ),
        ),
        seed=2,
    )

    partial = analyze_behavioral_coverage(_spec((tool, True)), (unconfigured,))
    covered = analyze_behavioral_coverage(_spec((tool, True)), (configured,))

    partial_requirement = _requirement(
        partial,
        BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
        "tool:mutate",
    )
    assert partial_requirement.status is BehavioralCoverageStatus.PARTIAL
    assert partial_requirement.reason_code == "fabrication_terms_unconfigured"
    assert _requirement(
        covered,
        BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
        "tool:mutate",
    ).status is BehavioralCoverageStatus.COVERED


def test_duplicate_ambiguous_and_retry_need_prohibited_attempt_fixtures() -> None:
    tool = _tool("destroy", state_changing=True, destructive=True)
    spec = _spec((tool, True))
    one = (_fixture("destroy", SimulatedToolStatus.SUCCESS, invocation_index=1),)
    two = (
        *one,
        _fixture(
            "destroy",
            SimulatedToolStatus.SUCCESS,
            fixture_id="fixture-2",
            invocation_index=2,
        ),
    )
    duplicate_partial = _scenario(
        "duplicate-partial",
        fixtures=one,
        required=(_required("destroy"),),
        trajectory=(_trajectory(TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT, "destroy"),),
    )
    duplicate_covered = _scenario(
        "duplicate-covered",
        fixtures=two,
        required=(_required("destroy"),),
        trajectory=(_trajectory(TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT, "destroy"),),
        seed=2,
    )
    assert _requirement(
        analyze_behavioral_coverage(spec, (duplicate_partial,)),
        BehavioralDimension.DUPLICATE_ACTION,
        "tool:destroy",
    ).status is BehavioralCoverageStatus.PARTIAL
    assert _requirement(
        analyze_behavioral_coverage(spec, (duplicate_covered,)),
        BehavioralDimension.DUPLICATE_ACTION,
        "tool:destroy",
    ).status is BehavioralCoverageStatus.COVERED

    ambiguous_one = _scenario(
        "ambiguous-one",
        fixtures=(
            _fixture(
                "destroy",
                SimulatedToolStatus.TIMEOUT,
                invocation_index=1,
                error_code="ambiguous_timeout",
            ),
        ),
        required=(_required("destroy"),),
        trajectory=(
            _trajectory(
                TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT,
                "destroy",
            ),
        ),
        seed=3,
    )
    ambiguous_two = ambiguous_one.model_copy(
        update={
            "scenario_id": "ambiguous-two",
            "tool_fixtures": (
                *ambiguous_one.tool_fixtures,
                _fixture(
                    "destroy",
                    SimulatedToolStatus.SUCCESS,
                    fixture_id="after-ambiguous",
                    invocation_index=2,
                ),
            ),
            "generation_seed": 4,
            "fingerprint": "",
        }
    )
    ambiguous_two = Scenario.model_validate(ambiguous_two.model_dump())
    assert _requirement(
        analyze_behavioral_coverage(spec, (ambiguous_one,)),
        BehavioralDimension.AMBIGUOUS_OUTCOME,
        "tool:destroy",
    ).status is BehavioralCoverageStatus.PARTIAL
    assert _requirement(
        analyze_behavioral_coverage(spec, (ambiguous_two,)),
        BehavioralDimension.AMBIGUOUS_OUTCOME,
        "tool:destroy",
    ).status is BehavioralCoverageStatus.COVERED

    retry_two = _scenario(
        "retry-two",
        fixtures=two,
        required=(_required("destroy"),),
        trajectory=(
            _trajectory(
                TrajectoryConstraintKind.MAX_RETRIES,
                "destroy",
                extra={"max_retries": 1},
            ),
        ),
        seed=5,
    )
    retry_three = _scenario(
        "retry-three",
        fixtures=(
            *two,
            _fixture(
                "destroy",
                SimulatedToolStatus.SUCCESS,
                fixture_id="fixture-3",
                invocation_index=3,
            ),
        ),
        required=(_required("destroy"),),
        trajectory=retry_two.trajectory_constraints,
        seed=6,
    )
    assert _requirement(
        analyze_behavioral_coverage(spec, (retry_two,)),
        BehavioralDimension.RETRY_CONTROL,
        "tool:destroy",
    ).status is BehavioralCoverageStatus.PARTIAL
    assert _requirement(
        analyze_behavioral_coverage(spec, (retry_three,)),
        BehavioralDimension.RETRY_CONTROL,
        "tool:destroy",
    ).status is BehavioralCoverageStatus.COVERED


def test_confirmation_paths_are_distinct() -> None:
    tool = _tool("change", state_changing=True)
    policy = ToolPolicy(
        policy_id="confirmation",
        tool_name="change",
        description="Require explicit confirmation.",
        confirmation_required=True,
    )
    spec = _spec((tool, True), policies=((policy, True),))
    without = _scenario(
        "without-consent",
        fixtures=(_fixture("change", SimulatedToolStatus.SUCCESS),),
        forbidden=(_forbidden("change", confirmation=True),),
    )
    with_consent = _scenario(
        "with-consent",
        fixtures=(_fixture("change", SimulatedToolStatus.SUCCESS),),
        required=(_required("change", confirmation=True),),
        confirmed=True,
        seed=2,
    )
    report = analyze_behavioral_coverage(spec, (with_consent, without))

    assert _requirement(
        report,
        BehavioralDimension.CONFIRMATION_WITHOUT_CONSENT,
        "tool:change",
    ).status is BehavioralCoverageStatus.COVERED
    assert _requirement(
        report,
        BehavioralDimension.CONFIRMATION_WITH_CONSENT,
        "tool:change",
    ).status is BehavioralCoverageStatus.COVERED


def test_prerequisite_coverage_stays_conservative_until_order_is_observable() -> None:
    focal = _tool("act", state_changing=True)
    prerequisite = _tool("lookup")
    spec = _spec((focal, True), (prerequisite, True))
    success = _scenario(
        "prerequisite-success",
        fixtures=(
            _fixture(
                "lookup",
                SimulatedToolStatus.SUCCESS,
                fixture_id="prerequisite-lookup",
            ),
            _fixture("act", SimulatedToolStatus.SUCCESS, fixture_id="focal"),
        ),
        required=(
            _required("lookup", criterion_id="require-lookup"),
            _required("act", criterion_id="require-act"),
        ),
        dimension_tags=("source:positive_path", "tool:act"),
    )
    failure_vacuous = _scenario(
        "prerequisite-failure-vacuous",
        fixtures=(
            _fixture(
                "lookup",
                SimulatedToolStatus.ERROR,
                fixture_id="prerequisite-lookup-error",
                error_code="missing",
            ),
            _fixture("act", SimulatedToolStatus.SUCCESS, fixture_id="focal"),
        ),
        forbidden=(_forbidden("act"),),
        dimension_tags=("source:behavioral_outcome", "tool:act"),
        seed=2,
    )
    failure_asserted = _scenario(
        "prerequisite-failure-asserted",
        fixtures=failure_vacuous.tool_fixtures,
        required=(_required("lookup", criterion_id="require-failing-lookup"),),
        forbidden=(_forbidden("act"),),
        dimension_tags=("source:behavioral_outcome", "tool:act"),
        seed=3,
    )
    subject = "prerequisite:lookup->tool:act"

    success_report = analyze_behavioral_coverage(spec, (success,))
    success_requirement = _requirement(
        success_report, BehavioralDimension.PREREQUISITE_SUCCESS, subject
    )
    assert success_requirement.status is BehavioralCoverageStatus.PARTIAL
    assert success_requirement.reason_code == "calls_asserted_but_order_unobservable"
    ordering = _requirement(success_report, BehavioralDimension.ORDERING, subject)
    assert ordering.status is BehavioralCoverageStatus.UNSUPPORTED
    assert "scenario:prerequisite-success" in ordering.evidence
    assert "fixture:prerequisite-lookup" in ordering.evidence

    vacuous_report = analyze_behavioral_coverage(spec, (failure_vacuous,))
    assert _requirement(
        vacuous_report, BehavioralDimension.PREREQUISITE_FAILURE, subject
    ).status is BehavioralCoverageStatus.PARTIAL
    asserted_report = analyze_behavioral_coverage(spec, (failure_asserted,))
    assert _requirement(
        asserted_report, BehavioralDimension.PREREQUISITE_FAILURE, subject
    ).status is BehavioralCoverageStatus.COVERED


def test_arbitrary_ordering_is_explicitly_unsupported() -> None:
    tool = _tool("act")
    scenario = _scenario(
        "ordering",
        trajectory=(_trajectory(TrajectoryConstraintKind.ORDERING, "act"),),
    )
    report = analyze_behavioral_coverage(_spec((tool, True)), (scenario,))

    requirement = _requirement(report, BehavioralDimension.ORDERING, "tool:act")
    assert requirement.status is BehavioralCoverageStatus.UNSUPPORTED
    assert _family(report, BehavioralDimension.ORDERING).applicable == 0


def test_reference_pool_preserves_denominator_and_must_contain_actual_sources() -> None:
    tool = _tool("act")
    spec = _spec((tool, True))
    selected = _scenario(
        "selected-success",
        fixtures=(_fixture("act", SimulatedToolStatus.SUCCESS),),
        required=(_required("act"),),
    )
    excluded = _scenario(
        "excluded-duplicate",
        fixtures=(
            _fixture("act", SimulatedToolStatus.SUCCESS, invocation_index=1),
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                fixture_id="second",
                invocation_index=2,
            ),
        ),
        required=(_required("act"),),
        trajectory=(
            _trajectory(TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT, "act"),
        ),
        seed=2,
    )

    selected_report = analyze_behavioral_coverage(
        spec,
        (selected,),
        reference_scenarios=(excluded, selected),
    )
    duplicate = _requirement(
        selected_report, BehavioralDimension.DUPLICATE_ACTION, "tool:act"
    )
    assert duplicate.status is BehavioralCoverageStatus.MISSING
    assert selected_report.scenario_count == 1
    assert not _family(
        analyze_behavioral_coverage(spec, (selected,)),
        BehavioralDimension.DUPLICATE_ACTION,
    ).requirements

    with pytest.raises(ValueError, match="must be contained"):
        analyze_behavioral_coverage(
            spec,
            (selected,),
            reference_scenarios=(excluded,),
        )
    with pytest.raises(ValueError, match="must be contained"):
        analyze_behavioral_coverage(
            spec,
            (selected, selected),
            reference_scenarios=(selected,),
        )


def test_source_binding_and_content_integrity_fail_closed() -> None:
    tool = _tool("act")
    spec = _spec((tool, True))
    scenario = _scenario(
        "bound",
        fixtures=(_fixture("act", SimulatedToolStatus.SUCCESS),),
        required=(_required("act"),),
    )
    report = analyze_behavioral_coverage(
        spec, (scenario,), suite_fingerprint="sha256:suite"
    )

    verify_behavioral_coverage_binding(report, spec, (scenario,))
    verify_behavioral_coverage_binding(
        report, spec, (scenario,), suite_fingerprint="sha256:suite"
    )
    with pytest.raises(ValueError, match="suite_fingerprint"):
        verify_behavioral_coverage_binding(
            report, spec, (scenario,), suite_fingerprint=None
        )
    with pytest.raises(ValueError, match="suite_fingerprint"):
        verify_behavioral_coverage_binding(
            report, spec, (scenario,), suite_fingerprint="sha256:wrong"
        )
    with pytest.raises(ValueError, match="spec_id"):
        verify_behavioral_coverage_binding(
            report, spec.model_copy(update={"spec_id": "other"}), (scenario,)
        )
    with pytest.raises(ValueError, match="scenario_count"):
        verify_behavioral_coverage_binding(report, spec, ())

    relabelled = scenario.model_copy(
        update={"scenario_id": "renamed", "title": "Renamed display identity"}
    )
    assert relabelled.fingerprint == scenario.fingerprint
    with pytest.raises(ValueError, match="scenario_digest"):
        verify_behavioral_coverage_binding(report, spec, (relabelled,))

    corrupt_memory = report.model_copy(update={"scenario_count": 99})
    with pytest.raises(ValueError, match="fingerprint"):
        verify_behavioral_coverage_binding(corrupt_memory, spec, (scenario,))

    payload = report.model_dump(mode="json")
    success_family = next(
        family
        for family in payload["families"]
        if family["dimension"] == BehavioralDimension.SUCCESS_PATH.value
    )
    success_family["requirements"][0]["status"] = BehavioralCoverageStatus.PARTIAL.value
    success_family["covered"] = 0
    success_family["partial"] = 1
    with pytest.raises(ValidationError, match="fingerprint does not match"):
        BehavioralCoverage.model_validate_json(json.dumps(payload))


def test_contract_round_trip_ordering_and_strictness_are_deterministic() -> None:
    alpha = _tool("alpha")
    beta = _tool("beta")
    first = _scenario(
        "first",
        fixtures=(_fixture("alpha", SimulatedToolStatus.SUCCESS),),
        required=(_required("alpha"),),
        seed=1,
    )
    second = _scenario(
        "second",
        fixtures=(_fixture("beta", SimulatedToolStatus.SUCCESS),),
        required=(_required("beta"),),
        seed=2,
    )
    forward = analyze_behavioral_coverage(
        _spec((alpha, True), (beta, True)), (first, second)
    )
    reverse = analyze_behavioral_coverage(
        _spec((beta, True), (alpha, True)), (second, first)
    )

    assert forward == reverse
    assert forward.fingerprint == forward.expected_fingerprint()
    assert BehavioralCoverage.from_json(forward.canonical_json()) == forward
    payload = forward.model_dump(mode="json")
    payload["extra"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BehavioralCoverage.model_validate(payload)


def test_analysis_never_invokes_gateway_or_mutates_scenario_or_frozen_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _tool("mutate", state_changing=True)
    spec = _spec((tool, True))
    scenario = _scenario(
        "integrity",
        fixtures=(
            _fixture(
                "mutate",
                SimulatedToolStatus.SUCCESS,
                state_effects=(
                    WorldStateEffect(
                        path="records.one.status",
                        before="unchanged",
                        after="changed",
                    ),
                ),
            ),
        ),
        required=(_required("mutate"),),
    )
    suite = FrozenSuite(
        spec_id=spec.spec_id,
        seed=17,
        provenance=GeneratorProvenance(
            generator="coverage-test",
            generator_version="1",
            sources=("tests/target.py",),
        ),
        coverage=SuiteCoverage(tools=("mutate",)),
        cases=(
            FrozenCase(
                scenario=scenario,
                lineage=CaseLineage(origin=CaseOrigin.BUILT_IN),
            ),
        ),
    )
    scenario_json = scenario.model_dump_json()
    scenario_fingerprint = scenario.fingerprint
    suite_json = suite.model_dump_json()
    suite_fingerprint = suite.fingerprint
    world = deepcopy(scenario.initial_world_state)

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("coverage analysis must not invoke the gateway")

    monkeypatch.setattr(ToolGateway, "invoke", explode)
    analyze_behavioral_coverage(
        spec,
        suite.scenarios,
        suite_fingerprint=suite.fingerprint,
    )

    assert scenario.model_dump_json() == scenario_json
    assert scenario.fingerprint == scenario_fingerprint
    assert scenario.initial_world_state == world
    assert suite.model_dump_json() == suite_json
    assert suite.fingerprint == suite_fingerprint
    assert suite.expected_fingerprint() == suite_fingerprint


def test_high_cardinality_details_are_globally_bounded_and_redaction_safe() -> None:
    many_tools = tuple((_tool(f"tool-{index:02d}"), True) for index in range(30))
    missing_report = analyze_behavioral_coverage(_spec(*many_tools), ())
    success = _family(missing_report, BehavioralDimension.SUCCESS_PATH)
    assert success.missing == 30
    assert len(success.requirements) == MAX_COVERAGE_DETAILS
    assert success.omitted == 30 - MAX_COVERAGE_DETAILS

    tool = _tool("observed")
    scenarios = tuple(
        _scenario(
            f"evidence-{index:02d}",
            fixtures=(
                _fixture(
                    "observed",
                    SimulatedToolStatus.SUCCESS,
                    fixture_id=f"fixture-{index:02d}",
                ),
            ),
            required=(
                _required("observed", criterion_id=f"required-{index:02d}"),
            ),
            seed=index,
        )
        for index in range(30)
    )
    evidence_report = analyze_behavioral_coverage(_spec((tool, True)), scenarios)
    requirement = _requirement(
        evidence_report, BehavioralDimension.SUCCESS_PATH, "tool:observed"
    )
    assert len(requirement.evidence) == MAX_COVERAGE_DETAILS
    assert requirement.omitted_evidence > 0
    assert all(
        len(family.requirements) <= MAX_COVERAGE_DETAILS
        for family in evidence_report.families
    )
    assert all(
        len(item.evidence) <= MAX_COVERAGE_DETAILS
        for family in evidence_report.families
        for item in family.requirements
    )

    redacted = redact_artifact(evidence_report)
    encoded = json.dumps(redacted, sort_keys=True)
    assert "[TRUNCATED" not in encoded
    assert len(encoded.encode("utf-8")) < 8 * 1024 * 1024
    assert BehavioralCoverage.model_validate_json(encoded) == evidence_report


def _stored_coverage_round_trip(
    tmp_path: Path,
    report: BehavioralCoverage,
    *,
    run_id: str,
) -> tuple[BehavioralCoverage, str]:
    path = ArtifactStore(tmp_path, ".agentcheck", run_id).write_json(
        "coverage.json", report
    )
    encoded = path.read_text(encoding="utf-8")
    return BehavioralCoverage.model_validate_json(encoded), encoded


def test_secret_spec_id_remains_verifiable_after_safe_normalization() -> None:
    spec = _spec((_tool("read"), True), spec_id="sk-secretvalue123")
    report = analyze_behavioral_coverage(spec, ())

    verify_behavioral_coverage_binding(report, spec, ())
    assert "secretvalue123" not in report.canonical_json()


def test_redacted_subject_collisions_are_omitted_and_round_trip(
    tmp_path: Path,
) -> None:
    first = _tool("sk-secretvalue123")
    second = _tool("sk-differentsecret456")

    report = analyze_behavioral_coverage(
        _spec((first, True), (second, True)),
        (),
    )
    success = _family(report, BehavioralDimension.SUCCESS_PATH)

    assert success.unknown == 2
    assert success.requirements == ()
    assert success.omitted == 2
    restored, encoded = _stored_coverage_round_trip(
        tmp_path, report, run_id="subject-collision"
    )
    assert restored == report
    assert "secretvalue123" not in encoded
    assert "differentsecret456" not in encoded


def test_redacted_evidence_collisions_are_counted_and_round_trip(
    tmp_path: Path,
) -> None:
    tool = _tool("read")
    scenarios = (
        _scenario(
            "sk-secretvalue123",
            fixtures=(
                _fixture(
                    "read",
                    SimulatedToolStatus.SUCCESS,
                    fixture_id="sk-secretvalue123",
                ),
            ),
            required=(_required("read", criterion_id="required-first"),),
        ),
        _scenario(
            "sk-differentsecret456",
            fixtures=(
                _fixture(
                    "read",
                    SimulatedToolStatus.SUCCESS,
                    fixture_id="sk-differentsecret456",
                ),
            ),
            required=(_required("read", criterion_id="required-second"),),
            seed=2,
        ),
    )

    report = analyze_behavioral_coverage(_spec((tool, True)), scenarios)
    requirement = _requirement(
        report, BehavioralDimension.SUCCESS_PATH, "tool:read"
    )
    assert requirement.omitted_evidence > 0
    restored, encoded = _stored_coverage_round_trip(
        tmp_path, report, run_id="evidence-collision"
    )
    assert restored == report
    assert "secretvalue123" not in encoded
    assert "differentsecret456" not in encoded


def test_spec_digest_rejects_changed_authoritativeness_and_schema() -> None:
    original_tool = _tool(
        "act",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    )
    spec = _spec((original_tool, True))
    report = analyze_behavioral_coverage(spec, ())

    with pytest.raises(ValueError, match="spec_digest"):
        verify_behavioral_coverage_binding(
            report,
            _spec((original_tool, False)),
            (),
        )

    changed_schema = original_tool.model_copy(
        update={
            "input_schema": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            }
        }
    )
    with pytest.raises(ValueError, match="spec_digest"):
        verify_behavioral_coverage_binding(
            report,
            _spec((changed_schema, True)),
            (),
        )


def test_user_truncation_marker_cannot_collide_in_spec_binding() -> None:
    original_tool = _tool(
        "act",
        input_schema={
            "type": "object",
            "properties": {
                "[TRUNCATED_ITEMS]": {"type": "string"},
                "x": {"type": "string"},
            },
            "required": ["x"],
            "additionalProperties": False,
        },
    )
    changed_tool = original_tool.model_copy(
        update={
            "input_schema": {
                **original_tool.input_schema,
                "properties": {
                    "[TRUNCATED_ITEMS]": {"type": "string"},
                    "x": {"type": "integer"},
                },
            }
        }
    )
    scenario = _scenario(
        "marker-property",
        fixtures=(
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                arguments_match={"x": "ok"},
            ),
        ),
        required=(_required("act", arguments_match={"x": "ok"}),),
    )

    report = analyze_behavioral_coverage(
        _spec((original_tool, True)), (scenario,)
    )
    assert _family(report, BehavioralDimension.SUCCESS_PATH).covered == 1
    with pytest.raises(ValueError, match="spec_digest|derived_coverage"):
        verify_behavioral_coverage_binding(
            report,
            _spec((changed_tool, True)),
            (scenario,),
        )


@pytest.mark.parametrize(
    ("tool_name", "input_schema", "arguments"),
    (
        ("sk-secretvalue123", {"type": "object"}, {}),
        (
            "act",
            {
                "type": "object",
                "properties": {"sk-secretvalue123": {"type": "string"}},
                "required": ["sk-secretvalue123"],
            },
            {"sk-secretvalue123": "value"},
        ),
    ),
)
def test_redaction_lossy_tool_bindings_are_unknown(
    tool_name: str,
    input_schema: dict[str, Any],
    arguments: dict[str, Any],
) -> None:
    tool = _tool(tool_name, input_schema=input_schema)
    scenario = _scenario(
        "lossy-binding",
        fixtures=(
            _fixture(
                tool_name,
                SimulatedToolStatus.SUCCESS,
                arguments_match=arguments,
            ),
        ),
        required=(_required(tool_name, arguments_match=arguments),),
    )

    report = analyze_behavioral_coverage(_spec((tool, True)), (scenario,))
    success = _family(report, BehavioralDimension.SUCCESS_PATH)
    assert success.covered == 0
    assert success.unknown == 1
    verify_behavioral_coverage_binding(report, _spec((tool, True)), (scenario,))


@pytest.mark.parametrize(
    ("tool_name", "input_schema"),
    (
        ("sk-secretvalue123", {"type": "object"}),
        (
            "act",
            {
                "type": "object",
                "properties": {"sk-secretvalue123": {"type": "string"}},
                "required": ["sk-secretvalue123"],
            },
        ),
    ),
)
def test_redaction_lossy_tool_binding_survives_spec_and_coverage_artifact_round_trip(
    tmp_path: Path,
    tool_name: str,
    input_schema: dict[str, Any],
) -> None:
    spec = _spec((_tool(tool_name, input_schema=input_schema), True))
    report = analyze_behavioral_coverage(spec, ())
    path = ArtifactStore(tmp_path, ".agentcheck", "lossy-binding").write_json(
        "binding.json",
        {"spec": spec, "coverage": report},
    )
    encoded = path.read_text(encoding="utf-8")
    assert "secretvalue123" not in encoded
    payload = json.loads(encoded)
    stored_spec = AgentSpec.model_validate_json(json.dumps(payload["spec"]))
    stored_report = BehavioralCoverage.model_validate_json(
        json.dumps(payload["coverage"])
    )

    verify_behavioral_coverage_binding(stored_report, stored_spec, ())
    success = _family(stored_report, BehavioralDimension.SUCCESS_PATH)
    assert success.covered == 0
    assert success.unknown == 1


@pytest.mark.parametrize(
    "schema_kind",
    ("long-leaf", "deep-schema"),
)
def test_truncated_tool_schema_survives_spec_and_coverage_artifact_round_trip(
    tmp_path: Path,
    schema_kind: str,
) -> None:
    if schema_kind == "long-leaf":
        input_schema: dict[str, Any] = {
            "type": "object",
            "description": "x" * 9_000,
        }
        expected_marker = "...[TRUNCATED]"
    else:
        nested: dict[str, Any] = {"type": "string"}
        for _ in range(24):
            nested = {"nested": nested}
        input_schema = {"type": "object", "properties": nested}
        expected_marker = "[MAX_DEPTH]"

    spec = _spec((_tool("act", input_schema=input_schema), True))
    report = analyze_behavioral_coverage(spec, ())
    success = _family(report, BehavioralDimension.SUCCESS_PATH)
    assert success.covered == 0
    assert success.unknown == 1
    path = ArtifactStore(
        tmp_path,
        ".agentcheck",
        f"lossy-{schema_kind}",
    ).write_json(
        "binding.json",
        {"spec": spec, "coverage": report},
    )
    encoded = path.read_text(encoding="utf-8")
    assert expected_marker in encoded
    payload = json.loads(encoded)
    stored_spec = AgentSpec.model_validate_json(json.dumps(payload["spec"]))
    stored_report = BehavioralCoverage.model_validate_json(
        json.dumps(payload["coverage"])
    )

    assert stored_report == report
    verify_behavioral_coverage_binding(stored_report, stored_spec, ())
    restored_success = _family(stored_report, BehavioralDimension.SUCCESS_PATH)
    assert restored_success.covered == 0
    assert restored_success.unknown == 1


def test_wide_schema_spec_and_coverage_survive_artifact_round_trip(
    tmp_path: Path,
) -> None:
    properties = {
        f"field_{index:03d}": {"type": "string"} for index in range(120)
    }
    spec = _spec(
        (
            _tool(
                "wide",
                input_schema={"type": "object", "properties": properties},
            ),
            True,
        )
    )
    report = analyze_behavioral_coverage(spec, ())
    path = ArtifactStore(tmp_path, ".agentcheck", "wide-schema").write_json(
        "binding.json",
        {"spec": spec, "coverage": report},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_spec = AgentSpec.model_validate_json(
        json.dumps(payload["spec"])
    )
    stored_report = BehavioralCoverage.model_validate_json(
        json.dumps(payload["coverage"])
    )

    verify_behavioral_coverage_binding(stored_report, stored_spec, ())


def test_reference_universe_is_bound_and_verifiable() -> None:
    tool = _tool("act")
    spec = _spec((tool, True))
    selected = _scenario(
        "selected",
        fixtures=(_fixture("act", SimulatedToolStatus.SUCCESS),),
        required=(_required("act"),),
    )
    excluded = _scenario(
        "excluded",
        fixtures=(
            _fixture("act", SimulatedToolStatus.SUCCESS, invocation_index=1),
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                fixture_id="second",
                invocation_index=2,
            ),
        ),
        required=(_required("act"),),
        trajectory=(
            _trajectory(TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT, "act"),
        ),
        seed=2,
    )
    other_reference = _scenario(
        "other-reference",
        fixtures=(_fixture("act", SimulatedToolStatus.ERROR, error_code="failed"),),
        required=(_required("act"),),
        seed=3,
    )

    report = analyze_behavioral_coverage(
        spec,
        (selected,),
        reference_scenarios=(selected, excluded),
    )
    assert report.reference_scope is BehavioralCoverageReferenceScope.COMPLETE
    assert report.scenario_count == 1
    assert report.reference_scenario_count == 2
    assert report.reference_scenario_digest != report.scenario_digest
    verify_behavioral_coverage_binding(
        report,
        spec,
        (selected,),
        reference_scenarios=(excluded, selected),
    )
    with pytest.raises(ValueError, match="reference"):
        verify_behavioral_coverage_binding(
            report,
            spec,
            (selected,),
            reference_scenarios=(selected, other_reference),
        )
    with pytest.raises(ValueError, match="reference"):
        verify_behavioral_coverage_binding(
            report,
            spec,
            (selected,),
            reference_scenarios=(excluded,),
        )


def test_reference_scope_contract_rejects_impossible_bindings() -> None:
    scenario = _scenario("one")
    report = analyze_behavioral_coverage(_spec(), (scenario,))
    payload = report.model_dump(mode="json")
    payload["fingerprint"] = ""
    payload["reference_scenario_digest"] = "sha256:different"
    with pytest.raises(ValidationError, match="reference.*digest"):
        BehavioralCoverage.model_validate_json(json.dumps(payload))

    payload = report.model_dump(mode="json")
    payload["fingerprint"] = ""
    payload["reference_scope"] = (
        BehavioralCoverageReferenceScope.AVAILABLE_SCENARIOS_ONLY.value
    )
    payload["reference_scenario_count"] = 2
    with pytest.raises(ValidationError, match="reference.*count"):
        BehavioralCoverage.model_validate_json(json.dumps(payload))


def test_reference_subset_validation_preserves_multiplicity() -> None:
    scenario = _scenario("repeated")
    report = analyze_behavioral_coverage(
        _spec(),
        (scenario, scenario),
        reference_scenarios=(scenario, scenario, scenario),
    )
    verify_behavioral_coverage_binding(
        report,
        _spec(),
        (scenario, scenario),
        reference_scenarios=(scenario, scenario, scenario),
    )
    with pytest.raises(ValueError, match="reference"):
        verify_behavioral_coverage_binding(
            report,
            _spec(),
            (scenario, scenario),
            reference_scenarios=(scenario,),
        )


def test_generated_cases_do_not_launder_nonauthoritative_risk_metadata() -> None:
    tool = _tool(
        "mutate",
        state_changing=True,
        destructive=True,
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    )
    authoritative_spec = _spec((tool, True))
    scenarios = build_outcome_variant_cases(
        authoritative_spec,
        seed=7,
        representative_inputs={"mutate": {"id": "one"}},
    )
    spec = _spec((tool, False))

    assert scenarios
    report = analyze_behavioral_coverage(spec, scenarios)
    for dimension in (
        BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
        BehavioralDimension.DUPLICATE_ACTION,
        BehavioralDimension.AMBIGUOUS_OUTCOME,
        BehavioralDimension.RETRY_CONTROL,
    ):
        assert _requirement(report, dimension, "tool:mutate").status is (
            BehavioralCoverageStatus.UNKNOWN
        )


def test_confirmation_wrong_polarity_never_counts_as_covered() -> None:
    tool = _tool("change", state_changing=True)
    policy = ToolPolicy(
        policy_id="confirmation",
        tool_name="change",
        description="Require explicit confirmation.",
        confirmation_required=True,
    )
    spec = _spec((tool, True), policies=((policy, True),))
    confirmed_but_forbidden = _scenario(
        "confirmed-but-forbidden",
        fixtures=(_fixture("change", SimulatedToolStatus.SUCCESS),),
        forbidden=(_forbidden("change", confirmation=True),),
        confirmed=True,
    )
    unconfirmed_but_required = _scenario(
        "unconfirmed-but-required",
        fixtures=(_fixture("change", SimulatedToolStatus.SUCCESS),),
        required=(_required("change", confirmation=True),),
        seed=2,
    )

    report = analyze_behavioral_coverage(
        spec, (confirmed_but_forbidden, unconfirmed_but_required)
    )
    assert _requirement(
        report,
        BehavioralDimension.CONFIRMATION_WITH_CONSENT,
        "tool:change",
    ).status is not BehavioralCoverageStatus.COVERED
    assert _requirement(
        report,
        BehavioralDimension.CONFIRMATION_WITHOUT_CONSENT,
        "tool:change",
    ).status is not BehavioralCoverageStatus.COVERED


@pytest.mark.parametrize(
    ("scenario_id", "action_arguments", "fixture_arguments"),
    (
        ("fixture-mismatch", {"id": "expected"}, {"id": "other"}),
        ("schema-invalid", {"id": 1}, {"id": 1}),
    ),
)
def test_success_requires_schema_valid_matching_selected_fixture(
    scenario_id: str,
    action_arguments: dict[str, Any],
    fixture_arguments: dict[str, Any],
) -> None:
    tool = _tool(
        "act",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    )
    scenario = _scenario(
        scenario_id,
        fixtures=(
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                arguments_match=fixture_arguments,
            ),
        ),
        required=(_required("act", arguments_match=action_arguments),),
    )

    report = analyze_behavioral_coverage(_spec((tool, True)), (scenario,))
    assert _requirement(
        report, BehavioralDimension.SUCCESS_PATH, "tool:act"
    ).status is not BehavioralCoverageStatus.COVERED


def test_fixture_priority_ties_and_exact_ambiguity_slot_are_conservative() -> None:
    tool = _tool("act", state_changing=True, destructive=True)
    spec = _spec((tool, True))
    priority = _scenario(
        "priority",
        fixtures=(
            _fixture("act", SimulatedToolStatus.SUCCESS, priority=0),
            _fixture(
                "act",
                SimulatedToolStatus.ERROR,
                fixture_id="higher",
                error_code="selected",
                priority=10,
            ),
        ),
        required=(_required("act"),),
    )
    tie = _scenario(
        "tie",
        fixtures=(
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                fixture_id="tie-one",
                invocation_index=1,
            ),
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                fixture_id="tie-two",
                invocation_index=1,
            ),
        ),
        required=(_required("act"),),
        trajectory=(
            _trajectory(TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT, "act"),
        ),
        seed=2,
    )
    unreachable_ambiguous = _scenario(
        "unreachable-ambiguous-next",
        fixtures=(
            _fixture("act", SimulatedToolStatus.SUCCESS, invocation_index=1),
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                fixture_id="second",
                invocation_index=2,
            ),
            _fixture(
                "act",
                SimulatedToolStatus.TIMEOUT,
                fixture_id="third-ambiguous",
                invocation_index=3,
                error_code="ambiguous_timeout",
            ),
        ),
        required=(_required("act"),),
        trajectory=(
            _trajectory(
                TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT,
                "act",
            ),
        ),
        seed=3,
    )
    reachable_ambiguous = _scenario(
        "reachable-ambiguous-next",
        fixtures=(
            *unreachable_ambiguous.tool_fixtures,
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                fixture_id="fourth",
                invocation_index=4,
            ),
        ),
        required=unreachable_ambiguous.required_tool_behavior,
        trajectory=unreachable_ambiguous.trajectory_constraints,
        seed=4,
    )

    assert _requirement(
        analyze_behavioral_coverage(spec, (priority,)),
        BehavioralDimension.SUCCESS_PATH,
        "tool:act",
    ).status is not BehavioralCoverageStatus.COVERED
    assert _requirement(
        analyze_behavioral_coverage(spec, (tie,)),
        BehavioralDimension.DUPLICATE_ACTION,
        "tool:act",
    ).status is BehavioralCoverageStatus.PARTIAL
    assert _requirement(
        analyze_behavioral_coverage(spec, (unreachable_ambiguous,)),
        BehavioralDimension.AMBIGUOUS_OUTCOME,
        "tool:act",
    ).status is BehavioralCoverageStatus.PARTIAL
    assert _requirement(
        analyze_behavioral_coverage(spec, (reachable_ambiguous,)),
        BehavioralDimension.AMBIGUOUS_OUTCOME,
        "tool:act",
    ).status is BehavioralCoverageStatus.COVERED


def test_state_changing_timeout_normalization_matches_gateway_for_fixture_and_fault() -> None:
    tool = _tool("act", state_changing=True, destructive=True)
    spec = _spec((tool, True))
    constraint = _trajectory(
        TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT,
        "act",
    )
    ordinary = _scenario(
        "ordinary-timeout",
        fixtures=(
            _fixture(
                "act",
                SimulatedToolStatus.TIMEOUT,
                invocation_index=1,
                error_code="timeout",
            ),
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                fixture_id="ordinary-next",
                invocation_index=2,
            ),
        ),
        required=(_required("act"),),
        trajectory=(constraint,),
    )
    injected = _scenario(
        "injected-timeout",
        fixtures=(
            _fixture("act", SimulatedToolStatus.SUCCESS, invocation_index=1),
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                fixture_id="fault-next",
                invocation_index=2,
            ),
        ),
        injected_faults=(
            InjectedFault(
                fault_id="timeout-fault",
                tool_name="act",
                fault_type=FaultType.TIMEOUT,
                invocation_index=1,
            ),
        ),
        required=(_required("act"),),
        trajectory=(constraint,),
        seed=2,
    )

    for scenario in (ordinary, injected):
        report = analyze_behavioral_coverage(spec, (scenario,))
        assert _requirement(
            report,
            BehavioralDimension.AMBIGUOUS_OUTCOME,
            "tool:act",
        ).status is BehavioralCoverageStatus.COVERED
        assert _requirement(
            report,
            BehavioralDimension.RETRY_CONTROL,
            "tool:act",
        ).status is BehavioralCoverageStatus.COVERED


def test_duplicate_reachability_obeys_tool_budget_latency_and_state_effects() -> None:
    tool = _tool("act", state_changing=True)
    spec = _spec((tool, True))
    trajectory = (
        _trajectory(TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT, "act"),
    )
    two = (
        _fixture("act", SimulatedToolStatus.SUCCESS, invocation_index=1),
        _fixture(
            "act",
            SimulatedToolStatus.SUCCESS,
            fixture_id="second",
            invocation_index=2,
        ),
    )
    budget_limited = _scenario(
        "budget-limited",
        fixtures=two,
        required=(_required("act"),),
        trajectory=trajectory,
        resource_budgets=ResourceBudgets(max_tool_calls=1),
    )
    latency_limited = _scenario(
        "latency-limited",
        fixtures=(
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                invocation_index=1,
                latency_ms=600,
            ),
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                fixture_id="second",
                invocation_index=2,
                latency_ms=600,
            ),
        ),
        required=(_required("act"),),
        trajectory=trajectory,
        resource_budgets=ResourceBudgets(wall_clock_seconds=1),
        seed=2,
    )
    state_effect_limited = _scenario(
        "state-effect-limited",
        fixtures=(
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                invocation_index=1,
                state_effects=(
                    WorldStateEffect(
                        path="records.one.status",
                        before="unchanged",
                        after="changed",
                    ),
                ),
            ),
            two[1],
        ),
        required=(_required("act"),),
        trajectory=trajectory,
        seed=3,
    )

    for scenario in (budget_limited, latency_limited, state_effect_limited):
        assert _requirement(
            analyze_behavioral_coverage(spec, (scenario,)),
            BehavioralDimension.DUPLICATE_ACTION,
            "tool:act",
        ).status is BehavioralCoverageStatus.PARTIAL


@pytest.mark.parametrize("maximum", (True, "1", -1, None))
def test_malformed_max_retries_never_counts_as_coverage(maximum: Any) -> None:
    tool = _tool("act", state_changing=True, destructive=True)
    scenario = _scenario(
        f"invalid-retries-{type(maximum).__name__}",
        fixtures=(
            _fixture(
                "act",
                SimulatedToolStatus.TIMEOUT,
                invocation_index=1,
                error_code="timeout",
            ),
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                fixture_id="second",
                invocation_index=2,
            ),
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                fixture_id="third",
                invocation_index=3,
            ),
        ),
        required=(_required("act"),),
        trajectory=(
            _trajectory(
                TrajectoryConstraintKind.MAX_RETRIES,
                "act",
                extra={"max_retries": maximum},
            ),
        ),
    )
    report = analyze_behavioral_coverage(_spec((tool, True)), (scenario,))

    for dimension in (
        BehavioralDimension.RETRY_CONTROL,
        BehavioralDimension.TIMEOUT_HANDLING,
    ):
        assert _requirement(
            report, dimension, "tool:act"
        ).status is not BehavioralCoverageStatus.COVERED


@pytest.mark.parametrize(
    "success_terms",
    ("succeeded", [""], [1], {"term": "succeeded"}),
)
def test_malformed_success_terms_never_count_as_supported_oracle(
    success_terms: Any,
) -> None:
    tool = _tool("act", state_changing=True)
    scenario = _scenario(
        f"invalid-success-terms-{type(success_terms).__name__}",
        fixtures=(
            _fixture(
                "act",
                SimulatedToolStatus.ERROR,
                error_code="failed",
            ),
        ),
        required=(_required("act"),),
        outputs=(
            _output(
                OutputCriterionKind.NO_FABRICATED_SUCCESS,
                parameters={"success_terms": success_terms},
            ),
        ),
    )
    report = analyze_behavioral_coverage(_spec((tool, True)), (scenario,))

    for dimension in (
        BehavioralDimension.FAILURE_HANDLING,
        BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
    ):
        assert _requirement(
            report, dimension, "tool:act"
        ).status is not BehavioralCoverageStatus.COVERED


def test_global_oracles_do_not_cover_multiple_focal_tools() -> None:
    alpha = _tool("alpha", state_changing=True)
    beta = _tool("beta", state_changing=True)
    required = (
        _required("alpha", criterion_id="require-alpha"),
        _required("beta", criterion_id="require-beta"),
    )
    failures = _scenario(
        "multi-focal-failures",
        fixtures=(
            _fixture(
                "alpha",
                SimulatedToolStatus.ERROR,
                fixture_id="alpha-error",
                error_code="failed",
            ),
            _fixture(
                "beta",
                SimulatedToolStatus.ERROR,
                fixture_id="beta-error",
                error_code="failed",
            ),
        ),
        required=required,
        outputs=(
            _output(
                OutputCriterionKind.ACKNOWLEDGES_TOOL_ERROR,
                criterion_id="global-ack",
            ),
            _output(
                OutputCriterionKind.NO_FABRICATED_SUCCESS,
                criterion_id="global-no-fabrication",
                parameters={"success_terms": ["succeeded"]},
            ),
        ),
    )
    timeouts = _scenario(
        "multi-focal-timeouts",
        fixtures=(
            _fixture(
                "alpha",
                SimulatedToolStatus.TIMEOUT,
                fixture_id="alpha-timeout",
                error_code="timeout",
            ),
            _fixture(
                "beta",
                SimulatedToolStatus.TIMEOUT,
                fixture_id="beta-timeout",
                error_code="timeout",
            ),
        ),
        required=required,
        outputs=(
            _output(
                OutputCriterionKind.ACKNOWLEDGES_TOOL_ERROR,
                criterion_id="global-timeout-ack",
            ),
        ),
        seed=2,
    )
    report = analyze_behavioral_coverage(
        _spec((alpha, True), (beta, True)),
        (failures, timeouts),
    )

    for tool_name in ("alpha", "beta"):
        for dimension in (
            BehavioralDimension.FAILURE_HANDLING,
            BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
            BehavioralDimension.TIMEOUT_HANDLING,
        ):
            assert _requirement(
                report, dimension, f"tool:{tool_name}"
            ).status is not BehavioralCoverageStatus.COVERED


def test_prerequisite_fixture_prefix_requires_unambiguous_generator_lineage() -> None:
    lookup = _tool("lookup")
    act = _tool("act", state_changing=True)
    other = _tool("other", state_changing=True)
    prerequisite_fixture = _fixture(
        "lookup",
        SimulatedToolStatus.SUCCESS,
        fixture_id="prerequisite-lookup",
    )
    no_source = _scenario(
        "no-generator-source",
        fixtures=(prerequisite_fixture,),
        required=(
            _required("lookup", criterion_id="lookup"),
            _required("act", criterion_id="act"),
        ),
    )
    multi_focal = _scenario(
        "multi-focal-generator-shape",
        fixtures=(
            prerequisite_fixture,
            _fixture("act", SimulatedToolStatus.SUCCESS, fixture_id="act"),
            _fixture("other", SimulatedToolStatus.SUCCESS, fixture_id="other"),
        ),
        required=(
            _required("lookup", criterion_id="lookup"),
            _required("act", criterion_id="act"),
            _required("other", criterion_id="other"),
        ),
        dimension_tags=(
            "source:positive_path",
            "tool:act",
            "tool:other",
        ),
        seed=2,
    )
    report = analyze_behavioral_coverage(
        _spec((lookup, True), (act, True), (other, True)),
        (no_source, multi_focal),
    )

    for dimension in (
        BehavioralDimension.PREREQUISITE_SUCCESS,
        BehavioralDimension.PREREQUISITE_FAILURE,
    ):
        assert _family(report, dimension).requirements == ()
        assert _family(report, dimension).applicable == 0


def test_duplicate_declared_tool_names_are_unknown_not_covered() -> None:
    # ToolGateway refuses to execute an ambiguous declared name. Derived
    # coverage is report metadata, so it must not abort generation either;
    # every subject bound to the ambiguous name stays unknown.
    duplicate = _tool("duplicate", state_changing=True, destructive=True)
    unique = _tool("unique")
    spec = _spec((duplicate, True), (duplicate, True), (unique, True))

    report = analyze_behavioral_coverage(spec, ())

    for dimension in BehavioralDimension:
        family = _family(report, dimension)
        ambiguous = [
            item for item in family.requirements if item.subject == "tool:duplicate"
        ]
        for item in ambiguous:
            assert item.status is BehavioralCoverageStatus.UNKNOWN
            assert item.reason_code == "declared_tool_name_is_ambiguous"
        assert len(ambiguous) <= 1

    success = _requirement(
        report, BehavioralDimension.SUCCESS_PATH, "tool:duplicate"
    )
    assert success.status is BehavioralCoverageStatus.UNKNOWN
    # An unambiguous sibling keeps its ordinary applicable requirement.
    assert (
        _requirement(report, BehavioralDimension.SUCCESS_PATH, "tool:unique").status
        is BehavioralCoverageStatus.MISSING
    )
    assert _family(report, BehavioralDimension.SUCCESS_PATH).applicable == 1


def test_a_simulated_mutation_still_covers_its_own_controlled_outcome() -> None:
    # A state-changing success fixture is the ordinary shape of a mutating
    # action. Its own outcome is controlled and observable, so excluding it
    # would report the most important tools as having no success path at all.
    tool = _tool("act", state_changing=True)
    spec = _spec((tool, True))
    mutation = (
        WorldStateEffect(path="records.one.status", before="unchanged", after="changed"),
    )
    scenario = _scenario(
        "mutating-success",
        fixtures=(
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                invocation_index=1,
                state_effects=mutation,
            ),
        ),
        required=(_required("act"),),
    )

    requirement = _requirement(
        analyze_behavioral_coverage(spec, (scenario,)),
        BehavioralDimension.SUCCESS_PATH,
        "tool:act",
    )

    assert requirement.status is BehavioralCoverageStatus.COVERED
    assert requirement.reason_code == "required_success_with_controlled_fixture"


def test_a_simulated_mutation_still_bounds_the_next_invocation() -> None:
    # The mutation makes the following invocation depend on world state this
    # analysis does not evaluate, so a duplicate attempt remains unproven even
    # though a second fixture exists.
    tool = _tool("act", state_changing=True)
    spec = _spec((tool, True))
    scenario = _scenario(
        "mutating-duplicate",
        fixtures=(
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                invocation_index=1,
                state_effects=(
                    WorldStateEffect(
                        path="records.one.status", before="unchanged", after="changed"
                    ),
                ),
            ),
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                fixture_id="second",
                invocation_index=2,
            ),
        ),
        required=(_required("act"),),
        trajectory=(_trajectory(TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT, "act"),),
    )

    requirement = _requirement(
        analyze_behavioral_coverage(spec, (scenario,)),
        BehavioralDimension.DUPLICATE_ACTION,
        "tool:act",
    )

    assert requirement.status is BehavioralCoverageStatus.PARTIAL
    assert requirement.reason_code == "duplicate_oracle_lacks_second_fixture"


def _untooled_duplicate_constraint() -> TrajectoryConstraint:
    return TrajectoryConstraint(
        criterion_id="untooled-duplicate",
        kind=TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT,
        description="A duplicate constraint that declares no tool signature.",
        parameters={},
        oracle_ids=(ORACLE_ID,),
    )


def test_scenario_scoped_requirement_never_borrows_another_scenario() -> None:
    # A constraint without a tool signature yields a scenario-scoped subject.
    # It must not draw evidence from a different scenario, and a tool-scoped
    # constraint elsewhere is not evidence that this signature is undeclared.
    tool = _tool("act", state_changing=True)
    spec = _spec((tool, True))
    untooled = _scenario(
        "untooled",
        trajectory=(_untooled_duplicate_constraint(),),
    )
    tooled = _scenario(
        "tooled",
        fixtures=(
            _fixture("act", SimulatedToolStatus.SUCCESS, invocation_index=1),
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                fixture_id="second",
                invocation_index=2,
            ),
        ),
        required=(_required("act"),),
        trajectory=(_trajectory(TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT, "act"),),
        seed=2,
    )

    report = analyze_behavioral_coverage(spec, (untooled, tooled))
    scoped = _requirement(
        report, BehavioralDimension.DUPLICATE_ACTION, "scenario:untooled"
    )

    assert scoped.status is BehavioralCoverageStatus.PARTIAL
    assert scoped.reason_code == "duplicate_oracle_without_declared_tool_signature"
    assert "scenario:tooled" not in scoped.evidence
    assert "scenario:untooled" in scoped.evidence
    # The tool-scoped constraint still belongs to its own tool requirement.
    assert (
        _requirement(
            report, BehavioralDimension.DUPLICATE_ACTION, "tool:act"
        ).status
        is BehavioralCoverageStatus.COVERED
    )


def test_scenario_scoped_requirement_excluded_by_selection_is_missing() -> None:
    # Selection dropped the only scenario that declared this subject, so its
    # behavior is missing rather than partially covered by a survivor.
    tool = _tool("act", state_changing=True)
    spec = _spec((tool, True))
    untooled = _scenario(
        "untooled",
        trajectory=(_untooled_duplicate_constraint(),),
    )
    survivor = _scenario(
        "survivor",
        trajectory=(
            TrajectoryConstraint(
                criterion_id="other-untooled",
                kind=TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT,
                description="Another constraint without a tool signature.",
                parameters={},
                oracle_ids=(ORACLE_ID,),
            ),
        ),
        seed=2,
    )

    report = analyze_behavioral_coverage(
        spec, (survivor,), reference_scenarios=(survivor, untooled)
    )
    scoped = _requirement(
        report, BehavioralDimension.DUPLICATE_ACTION, "scenario:untooled"
    )

    assert scoped.status is BehavioralCoverageStatus.MISSING
    assert scoped.reason_code == "duplicate_action_case_missing"
    assert scoped.evidence == ()


def test_an_unevaluable_input_schema_never_claims_controlled_reach() -> None:
    # An unresolvable local reference only fails while validating arguments,
    # not while building the validator. Coverage is derived report metadata,
    # so it fails closed instead of aborting the run.
    tool = ToolDefinition(
        name="act",
        input_schema={
            "type": "object",
            "properties": {"value": {"$ref": "#/$defs/absent"}},
        },
        state_changing=True,
        replaceable=True,
    )
    spec = _spec((tool, True))
    scenario = _scenario(
        "unevaluable-schema",
        fixtures=(
            _fixture(
                "act",
                SimulatedToolStatus.SUCCESS,
                invocation_index=1,
                arguments_match={"value": "x"},
            ),
        ),
        required=(_required("act", arguments_match={"value": "x"}),),
    )

    requirement = _requirement(
        analyze_behavioral_coverage(spec, (scenario,)),
        BehavioralDimension.SUCCESS_PATH,
        "tool:act",
    )

    assert requirement.status is BehavioralCoverageStatus.MISSING
    assert requirement.reason_code == "success_path_missing"
