from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from jsonschema import SchemaError  # type: ignore[import-untyped]

from agentcheck.domain import (
    AgentSpec,
    ConversationRole,
    OutputCriterionKind,
    PostconditionOperator,
    Scenario,
    ToolBehaviorConstraint,
    ToolFixture,
    TrajectoryConstraint,
    TrajectoryConstraintKind,
)
from agentcheck.schema_safety import UnsafeSchemaReference, offline_validator
from agentcheck.runner.world import WorldSimulator, WorldStateError


@dataclass(frozen=True, slots=True)
class ScenarioLintIssue:
    code: str
    message: str
    severity: str = "error"


_SUPPORTED_TRAJECTORY_KINDS = {
    TrajectoryConstraintKind.CONFIRMATION_BEFORE_TOOL,
    TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT,
    TrajectoryConstraintKind.NO_SAME_STAGE_DUPLICATE_ACTION,
    # The evaluator has judged ordering since observed-before landed, but this
    # set never learned about it, so every case carrying an ordering constraint
    # was linted out as an unsupported kind. The relation was evaluable and
    # unreachable at the same time. OTHER stays absent on purpose: it is a
    # catch-all with no evaluator behind it.
    TrajectoryConstraintKind.ORDERING,
    TrajectoryConstraintKind.MAX_RETRIES,
    TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT,
    TrajectoryConstraintKind.MAX_MODEL_TURNS,
    TrajectoryConstraintKind.MAX_TOOL_CALLS,
    TrajectoryConstraintKind.REQUIRED_HANDOFF,
    TrajectoryConstraintKind.FORBIDDEN_HANDOFF,
    TrajectoryConstraintKind.MAX_HANDOFFS,
    TrajectoryConstraintKind.NO_HANDOFF_LOOP,
    TrajectoryConstraintKind.HANDOFF_BEFORE_TOOL,
}
_SUPPORTED_OUTPUT_KINDS = {
    OutputCriterionKind.CONTAINS,
    OutputCriterionKind.NOT_CONTAINS,
    OutputCriterionKind.REGEX,
    OutputCriterionKind.JSON_SCHEMA,
    OutputCriterionKind.NO_FABRICATED_SUCCESS,
    OutputCriterionKind.ACKNOWLEDGES_TOOL_ERROR,
}


def _compatible_arguments(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether two subset matchers can both match one invocation."""

    return all(left[key] == right[key] for key in left.keys() & right.keys())


def _argument_specificity(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(1 + _argument_specificity(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return len(value) + sum(_argument_specificity(item) for item in value)
    return 1


def _fixture_ambiguities(fixtures: tuple[ToolFixture, ...]) -> list[tuple[str, str]]:
    ambiguous: list[tuple[str, str]] = []
    for index, left in enumerate(fixtures):
        for right in fixtures[index + 1 :]:
            if left.tool_name != right.tool_name or left.priority != right.priority:
                continue
            # An invocation-specific fixture outranks a generic fixture. Two
            # different explicit invocation indices can never match together.
            if (left.invocation_index is None) != (right.invocation_index is None):
                continue
            if (
                left.invocation_index is not None
                and left.invocation_index != right.invocation_index
            ):
                continue
            if _argument_specificity(left.arguments_match) != _argument_specificity(
                right.arguments_match
            ):
                continue
            if _compatible_arguments(left.arguments_match, right.arguments_match):
                ambiguous.append((left.fixture_id, right.fixture_id))
    return ambiguous


def _path_value(state: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    world = WorldSimulator(state)
    if not world.exists(path):
        return False, None
    return True, world.get(path)


def _lint_state_effects(scenario: Scenario) -> list[ScenarioLintIssue]:
    """Dry-run each fixture's ordered effects against isolated initial state."""

    issues: list[ScenarioLintIssue] = []
    for fixture in scenario.tool_fixtures:
        world = WorldSimulator(scenario.initial_world_state)
        for effect in fixture.outcome.state_effects:
            try:
                if "before" in effect.model_fields_set:
                    if not world.exists(effect.path):
                        raise WorldStateError(
                            f"expected an existing value at path {effect.path!r}"
                        )
                    if world.get(effect.path) != effect.before:
                        raise WorldStateError(
                            f"precondition did not match at path {effect.path!r}"
                        )
                world.set(effect.path, effect.after)
            except WorldStateError as exc:
                issues.append(
                    ScenarioLintIssue(
                        code="invalid_state_effect",
                        message=(
                            f"fixture {fixture.fixture_id!r} has invalid state effect "
                            f"path {effect.path!r}: {exc}"
                        ),
                    )
                )
                break
    return issues


def _supports_contains(value: Any, expected: Any) -> bool:
    if isinstance(value, str):
        return isinstance(expected, str)
    if isinstance(value, Mapping):
        # JSON object membership tests keys, and JSON object keys are strings.
        return isinstance(expected, str)
    return isinstance(value, list)


def _lint_postconditions(scenario: Scenario) -> list[ScenarioLintIssue]:
    """Reject invalid paths and incompatible containment contracts."""

    issues: list[ScenarioLintIssue] = []
    for condition in scenario.expected_postconditions:
        try:
            exists, initial = _path_value(
                scenario.initial_world_state, condition.path
            )
        except WorldStateError as exc:
            issues.append(
                ScenarioLintIssue(
                    code="invalid_postcondition",
                    message=(
                        f"postcondition {condition.criterion_id!r} has invalid "
                        f"path {condition.path!r}: {exc}"
                    ),
                )
            )
            continue
        if condition.operator != PostconditionOperator.CONTAINS:
            continue

        candidates: list[Any] = []
        if exists:
            candidates.append(initial)
        candidates.extend(
            effect.after
            for fixture in scenario.tool_fixtures
            for effect in fixture.outcome.state_effects
            if effect.path == condition.path
        )
        if candidates and not any(
            _supports_contains(candidate, condition.expected)
            for candidate in candidates
        ):
            issues.append(
                ScenarioLintIssue(
                    code="invalid_postcondition",
                    message=(
                        f"postcondition {condition.criterion_id!r} uses contains "
                        f"on a value that is not a compatible string, list, or object"
                    ),
                )
            )
    return issues


def _lint_trajectory_parameters(scenario: Scenario) -> list[ScenarioLintIssue]:
    issues: list[ScenarioLintIssue] = []
    for constraint in scenario.trajectory_constraints:
        if constraint.kind not in _SUPPORTED_TRAJECTORY_KINDS:
            issues.append(
                ScenarioLintIssue(
                    code="unsupported_trajectory_constraint",
                    message=(
                        f"trajectory criterion {constraint.criterion_id!r} uses "
                        f"unsupported Phase 1 kind {constraint.kind.value!r}"
                    ),
                )
            )
            continue
        tool_name = constraint.parameters.get("tool_name")
        if tool_name is not None and (not isinstance(tool_name, str) or not tool_name):
            issues.append(
                ScenarioLintIssue(
                    code="invalid_trajectory_parameters",
                    message=f"trajectory criterion {constraint.criterion_id!r} has an invalid tool_name",
                )
            )
        if constraint.kind in {
            TrajectoryConstraintKind.CONFIRMATION_BEFORE_TOOL,
            TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT,
        } and not isinstance(tool_name, str):
            issues.append(
                ScenarioLintIssue(
                    code="invalid_trajectory_parameters",
                    message=f"trajectory criterion {constraint.criterion_id!r} requires tool_name",
                )
            )
        numeric_key = {
            TrajectoryConstraintKind.MAX_RETRIES: "max_retries",
            TrajectoryConstraintKind.MAX_MODEL_TURNS: "maximum",
            TrajectoryConstraintKind.MAX_TOOL_CALLS: "maximum",
            TrajectoryConstraintKind.MAX_HANDOFFS: "maximum",
        }.get(constraint.kind)
        if numeric_key is not None:
            value = constraint.parameters.get(numeric_key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                issues.append(
                    ScenarioLintIssue(
                        code="invalid_trajectory_parameters",
                        message=(
                            f"trajectory criterion {constraint.criterion_id!r} requires "
                            f"a non-negative integer {numeric_key!r}"
                        ),
                    )
                )
        issues.extend(_lint_handoff_parameters(constraint))
    return issues


def _lint_handoff_parameters(constraint: TrajectoryConstraint) -> list[ScenarioLintIssue]:
    issues: list[ScenarioLintIssue] = []
    if constraint.kind not in {
        TrajectoryConstraintKind.REQUIRED_HANDOFF,
        TrajectoryConstraintKind.FORBIDDEN_HANDOFF,
        TrajectoryConstraintKind.NO_HANDOFF_LOOP,
        TrajectoryConstraintKind.HANDOFF_BEFORE_TOOL,
    }:
        return issues

    def _invalid(message: str) -> None:
        issues.append(
            ScenarioLintIssue(
                code="invalid_trajectory_parameters",
                message=f"trajectory criterion {constraint.criterion_id!r} {message}",
            )
        )

    from_agent = constraint.parameters.get("from_agent")
    to_agent = constraint.parameters.get("to_agent")
    for label, value in (("from_agent", from_agent), ("to_agent", to_agent)):
        if value is not None and (not isinstance(value, str) or not value):
            _invalid(f"has an invalid {label}")
    if constraint.kind == TrajectoryConstraintKind.REQUIRED_HANDOFF:
        if not isinstance(to_agent, str) or not to_agent:
            _invalid("requires to_agent")
        minimum = constraint.parameters.get("minimum", 1)
        maximum = constraint.parameters.get("maximum")
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            _invalid("requires a positive integer minimum")
        if maximum is not None and (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or (isinstance(minimum, int) and maximum < minimum)
        ):
            _invalid("requires maximum >= minimum")
    elif constraint.kind == TrajectoryConstraintKind.FORBIDDEN_HANDOFF:
        if not (isinstance(from_agent, str) and from_agent) and not (
            isinstance(to_agent, str) and to_agent
        ):
            _invalid("requires from_agent or to_agent")
    elif constraint.kind == TrajectoryConstraintKind.NO_HANDOFF_LOOP:
        repeats = constraint.parameters.get("max_edge_repeats", 1)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            _invalid("requires a positive integer max_edge_repeats")
    elif constraint.kind == TrajectoryConstraintKind.HANDOFF_BEFORE_TOOL:
        tool_name = constraint.parameters.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            _invalid("requires tool_name")
    return issues


def _lint_output_parameters(scenario: Scenario) -> list[ScenarioLintIssue]:
    issues: list[ScenarioLintIssue] = []
    for criterion in scenario.output_criteria:
        if criterion.kind not in _SUPPORTED_OUTPUT_KINDS:
            issues.append(
                ScenarioLintIssue(
                    code="unsupported_output_criterion",
                    message=(
                        f"output criterion {criterion.criterion_id!r} uses "
                        f"unsupported Phase 1 kind {criterion.kind.value!r}"
                    ),
                )
            )
            continue
        if criterion.kind in {
            OutputCriterionKind.CONTAINS,
            OutputCriterionKind.NOT_CONTAINS,
        }:
            value = criterion.parameters.get("text")
            if not isinstance(value, str) or not value:
                issues.append(
                    ScenarioLintIssue(
                        code="invalid_output_parameters",
                        message=f"output criterion {criterion.criterion_id!r} requires non-empty text",
                    )
                )
        elif criterion.kind == OutputCriterionKind.REGEX:
            pattern = criterion.parameters.get("pattern")
            if not isinstance(pattern, str) or not pattern or len(pattern) > 1_000:
                issues.append(
                    ScenarioLintIssue(
                        code="invalid_output_parameters",
                        message=f"output criterion {criterion.criterion_id!r} requires a regex up to 1,000 characters",
                    )
                )
            else:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    issues.append(
                        ScenarioLintIssue(
                            code="invalid_output_parameters",
                            message=f"output criterion {criterion.criterion_id!r} has invalid regex: {exc}",
                        )
                    )
        elif criterion.kind == OutputCriterionKind.JSON_SCHEMA:
            schema = criterion.parameters.get("schema")
            if not isinstance(schema, Mapping):
                issues.append(
                    ScenarioLintIssue(
                        code="invalid_output_schema",
                        message=f"output criterion {criterion.criterion_id!r} requires an object schema",
                    )
                )
            else:
                try:
                    offline_validator(schema)
                except (SchemaError, UnsafeSchemaReference) as exc:
                    issues.append(
                        ScenarioLintIssue(
                            code="invalid_output_schema",
                            message=f"output criterion {criterion.criterion_id!r} has an invalid or unsafe schema: {exc}",
                        )
                    )
        elif criterion.kind == OutputCriterionKind.NO_FABRICATED_SUCCESS:
            terms = criterion.parameters.get("success_terms")
            if terms is not None and (
                not isinstance(terms, list)
                or not terms
                or any(not isinstance(term, str) or not term for term in terms)
            ):
                issues.append(
                    ScenarioLintIssue(
                        code="invalid_output_parameters",
                        message=f"output criterion {criterion.criterion_id!r} success_terms must be a non-empty string list",
                    )
                )
    return issues


def _constraints_overlap(
    left: ToolBehaviorConstraint, right: ToolBehaviorConstraint
) -> bool:
    return left.tool_name == right.tool_name and _compatible_arguments(
        left.arguments_match, right.arguments_match
    )


def lint_scenario(scenario: Scenario, spec: AgentSpec) -> tuple[ScenarioLintIssue, ...]:
    """Reject invalid Phase 1 cases before they can affect an agent score."""

    tool_names = {item.value.name for item in spec.tools.items}
    issues: list[ScenarioLintIssue] = []
    unsupported_roles = sorted(
        {
            turn.role.value
            for turn in scenario.conversation_turns
            if turn.role not in {ConversationRole.USER, ConversationRole.ASSISTANT}
        }
    )
    if unsupported_roles:
        issues.append(
            ScenarioLintIssue(
                code="unsupported_conversation_role",
                message=(
                    "Phase 1 scenario input supports only user and assistant turns; "
                    f"found {', '.join(unsupported_roles)}"
                ),
            )
        )
    unsupported_followups = sorted(
        {
            turn.role.value
            for turn in scenario.followup_turns
            if turn.role != ConversationRole.USER
        }
    )
    if unsupported_followups:
        issues.append(
            ScenarioLintIssue(
                code="unsupported_followup_role",
                message=(
                    "A scripted follow-up is delivered after the agent has "
                    "answered, so it may only be a user turn; found "
                    f"{', '.join(unsupported_followups)}"
                ),
            )
        )
    referenced: list[tuple[str, str]] = []
    referenced.extend(("fixture", fixture.tool_name) for fixture in scenario.tool_fixtures)
    referenced.extend(("fault", fault.tool_name) for fault in scenario.injected_faults)
    for group_name, group in (
        ("required", scenario.required_tool_behavior),
        ("allowed", scenario.allowed_tool_behavior),
        ("forbidden", scenario.forbidden_tool_behavior),
    ):
        referenced.extend((group_name, behavior.tool_name) for behavior in group)
    for trajectory_constraint in scenario.trajectory_constraints:
        trajectory_tool = trajectory_constraint.parameters.get("tool_name")
        if isinstance(trajectory_tool, str) and trajectory_tool:
            referenced.append(("trajectory", trajectory_tool))
    for origin, name in referenced:
        if name not in tool_names:
            issues.append(
                ScenarioLintIssue(
                    code="nonexistent_tool",
                    message=f"{origin} references unavailable tool {name!r}",
                )
            )

    controlled_matchers: dict[str, list[Mapping[str, Any]]] = {}
    for fixture in scenario.tool_fixtures:
        controlled_matchers.setdefault(fixture.tool_name, []).append(
            fixture.arguments_match
        )
    for fault in scenario.injected_faults:
        controlled_matchers.setdefault(fault.tool_name, []).append({})
    for required_constraint in scenario.required_tool_behavior:
        matchers = controlled_matchers.get(required_constraint.tool_name, [])
        if not any(
            _compatible_arguments(required_constraint.arguments_match, matcher)
            for matcher in matchers
        ):
            issues.append(
                ScenarioLintIssue(
                    code="missing_fixture",
                    message=(
                        f"required tool {required_constraint.tool_name!r} has no controlled "
                        "fixture compatible with its argument contract"
                    ),
                )
            )

    for required in scenario.required_tool_behavior:
        for forbidden in scenario.forbidden_tool_behavior:
            if _constraints_overlap(required, forbidden):
                issues.append(
                    ScenarioLintIssue(
                        code="contradictory_tool_expectation",
                        message=(
                            f"tool {required.tool_name!r} has overlapping required "
                            "and forbidden argument contracts"
                        ),
                    )
                )

    all_criterion_groups = (
        scenario.expected_postconditions,
        scenario.required_tool_behavior,
        scenario.allowed_tool_behavior,
        scenario.forbidden_tool_behavior,
        scenario.trajectory_constraints,
        scenario.output_criteria,
    )
    criteria = [criterion for group in all_criterion_groups for criterion in group]
    evaluable_criteria = [
        criterion
        for group in (
            scenario.expected_postconditions,
            scenario.required_tool_behavior,
            scenario.forbidden_tool_behavior,
            scenario.trajectory_constraints,
            scenario.output_criteria,
        )
        for criterion in group
    ]
    if not evaluable_criteria:
        issues.append(ScenarioLintIssue(code="no_oracle", message="scenario has no evaluable criterion"))
    criterion_ids = [criterion.criterion_id for criterion in criteria]
    if len(criterion_ids) != len(set(criterion_ids)):
        issues.append(
            ScenarioLintIssue(code="duplicate_criterion_id", message="criterion IDs must be unique")
        )

    fixture_ids = [fixture.fixture_id for fixture in scenario.tool_fixtures]
    if len(fixture_ids) != len(set(fixture_ids)):
        issues.append(
            ScenarioLintIssue(code="duplicate_fixture_id", message="fixture IDs must be unique")
        )
    fault_ids = [fault.fault_id for fault in scenario.injected_faults]
    if len(fault_ids) != len(set(fault_ids)):
        issues.append(
            ScenarioLintIssue(code="duplicate_fault_id", message="injected fault IDs must be unique")
        )
    fault_slots = [
        (fault.tool_name, fault.invocation_index) for fault in scenario.injected_faults
    ]
    if len(fault_slots) != len(set(fault_slots)):
        issues.append(
            ScenarioLintIssue(
                code="ambiguous_injected_fault",
                message="only one injected fault may target a tool invocation",
            )
        )
    for left_id, right_id in _fixture_ambiguities(scenario.tool_fixtures):
        issues.append(
            ScenarioLintIssue(
                code="ambiguous_fixture",
                message=f"fixtures {left_id!r} and {right_id!r} can tie for one invocation",
            )
        )

    issues.extend(_lint_state_effects(scenario))
    issues.extend(_lint_postconditions(scenario))
    issues.extend(_lint_trajectory_parameters(scenario))
    issues.extend(_lint_output_parameters(scenario))
    return tuple(issues)


def lint_suite(
    scenarios: Sequence[Scenario], spec: AgentSpec
) -> tuple[tuple[Scenario, tuple[ScenarioLintIssue, ...]], ...]:
    """Lint a suite while preserving positional identity for duplicate IDs."""

    scenario_list = tuple(scenarios)
    issue_lists = [list(lint_scenario(scenario, spec)) for scenario in scenario_list]
    ids: dict[str, list[int]] = {}
    fingerprints: dict[str, list[int]] = {}
    for index, scenario in enumerate(scenario_list):
        ids.setdefault(scenario.scenario_id, []).append(index)
        fingerprints.setdefault(scenario.fingerprint, []).append(index)
    for indexes in ids.values():
        if len(indexes) > 1:
            for index in indexes:
                issue_lists[index].append(
                    ScenarioLintIssue(
                        code="duplicate_scenario_id",
                        message="scenario IDs must be unique within a suite",
                    )
                )
    for indexes in fingerprints.values():
        if len(indexes) > 1:
            for index in indexes:
                issue_lists[index].append(
                    ScenarioLintIssue(
                        code="duplicate_scenario_fingerprint",
                        message="structurally duplicate scenarios must be deduplicated",
                    )
                )
    return tuple(
        (scenario, tuple(issues))
        for scenario, issues in zip(scenario_list, issue_lists, strict=True)
    )


__all__ = ["ScenarioLintIssue", "lint_scenario", "lint_suite"]
