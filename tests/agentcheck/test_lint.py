from __future__ import annotations

import pytest

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.domain import (
    FaultType,
    InjectedFault,
    OutputCriterion,
    OutputCriterionKind,
    PostconditionOperator,
    Scenario,
    StatePostcondition,
    ToolBehaviorConstraint,
    ToolFixture,
    TrajectoryConstraint,
    TrajectoryConstraintKind,
    WorldStateEffect,
)
from agentcheck.generate import build_account_support_suite, lint_scenario, lint_suite
from agentcheck.inspect import load_target


def _spec():  # type annotation would expose the optional SDK dependency here
    target, source = load_target("examples/evaluation/account_agent")
    return OpenAIAgentsAdapter().inspect(target, source=source)


def _replace(scenario: Scenario, **updates: object) -> Scenario:
    data = scenario.model_dump(mode="python")
    data.update(updates)
    data["fingerprint"] = ""
    return Scenario.model_validate(data)


def _codes(scenario: Scenario) -> set[str]:
    return {issue.code for issue in lint_scenario(scenario, _spec())}


def test_phase_one_templates_all_pass_lint() -> None:
    assert all(not issues for _, issues in lint_suite(build_account_support_suite(), _spec()))


def test_required_arguments_need_a_compatible_fixture() -> None:
    scenario = build_account_support_suite()[0]
    original = scenario.tool_fixtures[0]
    fixture = original.model_copy(update={"arguments_match": {"account_id": "acct_999"}})
    mutated = _replace(scenario, tool_fixtures=(fixture,))

    assert "missing_fixture" in _codes(mutated)


def test_disjoint_required_and_forbidden_arguments_are_not_contradictory() -> None:
    scenario = build_account_support_suite()[0]
    forbidden = ToolBehaviorConstraint(
        criterion_id="forbid-other-account",
        tool_name="lookup_account",
        arguments_match={"account_id": "acct_999"},
        min_calls=0,
        max_calls=0,
        oracle_ids=scenario.required_tool_behavior[0].oracle_ids,
    )
    mutated = _replace(scenario, forbidden_tool_behavior=(forbidden,))

    assert "contradictory_tool_expectation" not in _codes(mutated)


def test_unsupported_and_unsafe_output_oracles_are_rejected() -> None:
    scenario = build_account_support_suite()[0]
    unsupported = OutputCriterion(
        criterion_id="semantic-only",
        kind=OutputCriterionKind.OTHER,
        description="Unsupported semantic expectation.",
        oracle_ids=scenario.output_criteria[0].oracle_ids,
    )
    remote_schema = OutputCriterion(
        criterion_id="remote-schema",
        kind=OutputCriterionKind.JSON_SCHEMA,
        description="Unsafe remote schema.",
        parameters={"schema": {"$ref": "https://example.invalid/schema.json"}},
        oracle_ids=scenario.output_criteria[0].oracle_ids,
    )
    mutated = _replace(scenario, output_criteria=(unsupported, remote_schema))

    assert {"unsupported_output_criterion", "invalid_output_schema"} <= _codes(mutated)


def test_duplicate_fault_slots_are_rejected_before_execution() -> None:
    scenario = build_account_support_suite()[0]
    faults = (
        InjectedFault(
            fault_id="timeout-1",
            tool_name="lookup_account",
            fault_type=FaultType.TIMEOUT,
            invocation_index=1,
            message="timeout",
        ),
        InjectedFault(
            fault_id="error-1",
            tool_name="lookup_account",
            fault_type=FaultType.ERROR,
            invocation_index=1,
            message="error",
        ),
    )
    mutated = _replace(scenario, injected_faults=faults)

    assert "ambiguous_injected_fault" in _codes(mutated)


def test_impossible_world_effect_precondition_is_rejected() -> None:
    scenario = build_account_support_suite()[1]
    fixture = scenario.tool_fixtures[0]
    bad_effect = WorldStateEffect(
        path="accounts.acct_123.email",
        before="never-a-possible-value@example.com",
        after="new@example.com",
    )
    bad_fixture = ToolFixture(
        **{
            **fixture.model_dump(mode="python"),
            "outcome": fixture.outcome.model_copy(
                update={"state_effects": (bad_effect,)}
            ),
        }
    )
    mutated = _replace(scenario, tool_fixtures=(bad_fixture,))

    assert "invalid_state_effect" in _codes(mutated)


def test_world_effect_cannot_traverse_through_a_scalar() -> None:
    scenario = build_account_support_suite()[1]
    fixture = scenario.tool_fixtures[0]
    bad_effect = WorldStateEffect(
        path="accounts.acct_123.exists.child",
        after=True,
    )
    bad_fixture = fixture.model_copy(
        update={
            "outcome": fixture.outcome.model_copy(
                update={"state_effects": (bad_effect,)}
            )
        }
    )
    mutated = _replace(scenario, tool_fixtures=(bad_fixture,))

    assert "invalid_state_effect" in _codes(mutated)


def test_malformed_world_effect_path_is_a_lint_issue() -> None:
    scenario = build_account_support_suite()[1]
    fixture = scenario.tool_fixtures[0]
    bad_effect = WorldStateEffect(
        path=".",
        before="alex@example.com",
        after="new@example.com",
    )
    bad_fixture = fixture.model_copy(
        update={
            "outcome": fixture.outcome.model_copy(
                update={"state_effects": (bad_effect,)}
            )
        }
    )
    mutated = _replace(scenario, tool_fixtures=(bad_fixture,))

    assert "invalid_state_effect" in _codes(mutated)


def test_contains_postcondition_on_scalar_is_rejected() -> None:
    scenario = build_account_support_suite()[0]
    condition = StatePostcondition(
        criterion_id="contains-bool",
        path="accounts.acct_123.exists",
        operator=PostconditionOperator.CONTAINS,
        expected=True,
        oracle_ids=scenario.required_tool_behavior[0].oracle_ids,
    )
    mutated = _replace(scenario, expected_postconditions=(condition,))

    assert "invalid_postcondition" in _codes(mutated)


@pytest.mark.parametrize("operator", tuple(PostconditionOperator))
def test_malformed_postcondition_path_is_rejected_for_every_operator(
    operator: PostconditionOperator,
) -> None:
    scenario = build_account_support_suite()[0]
    condition = StatePostcondition(
        criterion_id=f"invalid-path-{operator.value}",
        path=".",
        operator=operator,
        expected=True,
        oracle_ids=scenario.required_tool_behavior[0].oracle_ids,
    )
    mutated = _replace(scenario, expected_postconditions=(condition,))

    assert "invalid_postcondition" in _codes(mutated)


def test_json_pointer_postcondition_path_uses_world_simulator_semantics() -> None:
    scenario = build_account_support_suite()[0]
    condition = StatePostcondition(
        criterion_id="pointer-path",
        path="/accounts/acct_123/exists",
        operator=PostconditionOperator.EQUALS,
        expected=True,
        oracle_ids=scenario.required_tool_behavior[0].oracle_ids,
    )
    mutated = _replace(scenario, expected_postconditions=(condition,))

    assert "invalid_postcondition" not in _codes(mutated)


def test_suite_rejects_duplicate_ids_and_structural_fingerprints() -> None:
    scenario = build_account_support_suite()[0]
    same_id = _replace(scenario, title="Same ID")
    same_structure = _replace(scenario, scenario_id="different-id", title="Different ID")

    id_results = lint_suite((scenario, same_id), _spec())
    fingerprint_results = lint_suite((scenario, same_structure), _spec())

    assert all("duplicate_scenario_id" in {item.code for item in issues} for _, issues in id_results)
    assert all(
        "duplicate_scenario_fingerprint" in {item.code for item in issues}
        for _, issues in fingerprint_results
    )


def test_trajectory_typo_and_allowed_only_case_cannot_vacuously_pass() -> None:
    scenario = build_account_support_suite()[0]
    typo = TrajectoryConstraint(
        criterion_id="confirmation-typo",
        kind=TrajectoryConstraintKind.CONFIRMATION_BEFORE_TOOL,
        description="Confirmation is required.",
        parameters={"tool_name": "delete_acount"},
        oracle_ids=scenario.required_tool_behavior[0].oracle_ids,
    )
    typo_scenario = _replace(scenario, trajectory_constraints=(typo,))
    allowed_only = _replace(
        scenario,
        required_tool_behavior=(),
        output_criteria=(),
        allowed_tool_behavior=scenario.required_tool_behavior,
    )

    assert "nonexistent_tool" in _codes(typo_scenario)
    assert "no_oracle" in _codes(allowed_only)
