"""Pure behavioral-coverage analysis over validated contract data.

The analyzer never imports a target, adapter, gateway, or world simulator.  It
only asks whether a suite contains a controlled opportunity and an executable
oracle for requirements that the target or scenarios establish explicitly.
Names and descriptions are identifiers/evidence only; they are never risk
classification inputs.
"""

from __future__ import annotations

import hmac
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Mapping, Sequence, overload

from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
from referencing.exceptions import Unresolvable

from agentcheck.domain import (
    AgentSpec,
    FaultType,
    OracleStrength,
    OutputCriterion,
    OutputCriterionKind,
    Scenario,
    SimulatedToolStatus,
    ToolBehaviorConstraint,
    ToolDefinition,
    TrajectoryConstraint,
    TrajectoryConstraintKind,
    canonical_hash,
)
from agentcheck.privacy import redact_log_text
from agentcheck.redaction import DEFAULT_REDACTED_KEYS, sanitize_value
from agentcheck.schema_safety import UnsafeSchemaReference, offline_validator

from .contract import (
    MAX_COVERAGE_DETAILS,
    BehavioralCoverage,
    BehavioralCoverageFamily,
    BehavioralCoverageReferenceScope,
    BehavioralCoverageRequirement,
    BehavioralCoverageStatus,
    BehavioralDimension,
)


_PREREQUISITE_PREFIXES = ("prerequisite-", "prerequisite:")
_GENERATOR_ACTION_SOURCES = {
    "source:positive_path",
    "source:behavioral_outcome",
    "source:confirmed_action",
}
_TOOL_SCOPED_TRAJECTORY_KINDS = {
    TrajectoryConstraintKind.CONFIRMATION_BEFORE_TOOL,
    TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT,
    TrajectoryConstraintKind.NO_SAME_STAGE_DUPLICATE_ACTION,
    TrajectoryConstraintKind.MAX_RETRIES,
    TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT,
    TrajectoryConstraintKind.ORDERING,
}


# What a call can hand back that will not support a definite claim: it failed,
# it never established whether it ran, or it returned something unusable. The
# fabrication question is the same for all of them -- did the answer report a
# result the call never produced -- so the degraded outcomes belong in this
# dimension rather than in one of their own.
UNUSABLE_SIMULATED_STATUSES = frozenset(
    {
        SimulatedToolStatus.ERROR,
        SimulatedToolStatus.TIMEOUT,
        SimulatedToolStatus.EMPTY,
        SimulatedToolStatus.MALFORMED,
        SimulatedToolStatus.PARTIAL,
        SimulatedToolStatus.STALE,
    }
)


class _Applicability(str, Enum):
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    APPLICABLE = "applicable"


_APPLICABILITY_RANK = {
    _Applicability.UNKNOWN: 0,
    _Applicability.UNSUPPORTED: 1,
    _Applicability.APPLICABLE: 2,
}


@dataclass(frozen=True, slots=True)
class _Seed:
    dimension: BehavioralDimension
    subject: str
    applicability: _Applicability
    tool_name: str | None = None
    prerequisite_tool_name: str | None = None
    unknown_reason: str = "risk_metadata_not_authoritative"


@dataclass(frozen=True, slots=True)
class _Slot:
    tool_name: str
    invocation_index: int | None
    arguments_match: Mapping[str, object]
    status: SimulatedToolStatus
    error_code: str | None
    evidence_id: str
    priority: int
    latency_ms: float
    has_state_effects: bool
    position: int


@dataclass(frozen=True, slots=True)
class _Outcome:
    status: BehavioralCoverageStatus
    reason_code: str
    evidence: tuple[str, ...]


_OUTCOME_RANK = {
    BehavioralCoverageStatus.PARTIAL: 1,
    BehavioralCoverageStatus.COVERED: 2,
}


def _tool_subject(tool_name: str) -> str:
    return f"tool:{tool_name}"


def _prerequisite_subject(focal_tool: str, prerequisite_tool: str) -> str:
    return f"prerequisite:{prerequisite_tool}->tool:{focal_tool}"


def _artifact_normalize(value: object) -> object:
    """Apply artifact string/key redaction without a shared truncation budget."""

    if isinstance(value, str):
        return sanitize_value(value, redacted_keys=DEFAULT_REDACTED_KEYS)
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for raw_key, item in value.items():
            safe_pair = sanitize_value({raw_key: None}, redacted_keys=DEFAULT_REDACTED_KEYS)
            safe_key, marker = next(iter(safe_pair.items()))
            normalized[str(safe_key)] = (
                marker if marker == "[REDACTED]" else _artifact_normalize(item)
            )
        return normalized
    if isinstance(value, list | tuple):
        return [_artifact_normalize(item) for item in value]
    return sanitize_value(value, redacted_keys=DEFAULT_REDACTED_KEYS)


# These conservative schema-local bounds leave room for the surrounding
# AgentSpec/ArtifactStore envelope beneath its 20-level/100,000-node limits.
# Crossing a bound makes coverage UNKNOWN; the discarded schema content is
# never hashed or reconstructed from a stored prefix.
_STABLE_PROJECTION_ITEMS = 100
_STABLE_PROJECTION_DEPTH = 14
_STABLE_PROJECTION_NODES = 10_000
_SCHEMA_PROJECTION_CONTRACT = "agentcheck.behavioral_coverage.schema_projection.v1"
_ARTIFACT_LOSS_SENTINELS = frozenset(
    {
        "[CYCLE]",
        "[MAX_DEPTH]",
        "[NUMBER_OUT_OF_RANGE]",
        "[REDACTED]",
        "[TRUNCATED]",
        "[TRUNCATED_NODES]",
        "[UNAVAILABLE_ITEMS]",
    }
)


def _contains_artifact_loss_sentinel(value: object) -> bool:
    """Recognize exact normalization sentinels retained in a stored spec.

    Collection truncation is handled structurally by the item bound; a
    user-authored ``[TRUNCATED_ITEMS]`` key is not itself evidence of loss.
    """

    if isinstance(value, str):
        return (
            value in _ARTIFACT_LOSS_SENTINELS
            or "[REDACTED]" in value
            or value.endswith("...[TRUNCATED]")
        )
    if isinstance(value, Mapping):
        return any(
            _contains_artifact_loss_sentinel(key)
            or _contains_artifact_loss_sentinel(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(_contains_artifact_loss_sentinel(item) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class _StableProjection:
    value: object | None
    lossless: bool


def _stable_projection(
    value: object,
    *,
    depth: int,
    nodes_remaining: list[int],
) -> _StableProjection:
    if depth > _STABLE_PROJECTION_DEPTH or nodes_remaining[0] <= 0:
        return _StableProjection(value=None, lossless=False)
    nodes_remaining[0] -= 1

    if isinstance(value, Mapping):
        if len(value) > _STABLE_PROJECTION_ITEMS:
            return _StableProjection(value=None, lossless=False)
        projected: dict[str, object] = {}
        for raw_key, item in value.items():
            safe_pair = sanitize_value(
                {raw_key: None},
                redacted_keys=DEFAULT_REDACTED_KEYS,
            )
            safe_key, marker = next(iter(safe_pair.items()))
            if safe_key != raw_key or marker == "[REDACTED]":
                return _StableProjection(value=None, lossless=False)
            child = _stable_projection(
                item,
                depth=depth + 1,
                nodes_remaining=nodes_remaining,
            )
            if not child.lossless:
                return _StableProjection(value=None, lossless=False)
            projected[str(raw_key)] = child.value
        return _StableProjection(value=projected, lossless=True)

    if isinstance(value, list | tuple):
        if len(value) > _STABLE_PROJECTION_ITEMS:
            return _StableProjection(value=None, lossless=False)
        projected_items: list[object] = []
        for item in value:
            child = _stable_projection(
                item,
                depth=depth + 1,
                nodes_remaining=nodes_remaining,
            )
            if not child.lossless:
                return _StableProjection(value=None, lossless=False)
            projected_items.append(child.value)
        return _StableProjection(value=projected_items, lossless=True)

    normalized = _artifact_normalize(value)
    return _StableProjection(
        value=normalized if normalized == value else None,
        lossless=normalized == value,
    )


def _stable_bounded_projection(value: object) -> _StableProjection:
    """Return an ArtifactStore-idempotent semantic projection.

    Lossy schemas collapse outside the user document, so a literal marker-like
    schema cannot collide with the internal loss state.
    """

    if _contains_artifact_loss_sentinel(value):
        return _StableProjection(value=None, lossless=False)
    return _stable_projection(
        value,
        depth=0,
        nodes_remaining=[_STABLE_PROJECTION_NODES],
    )


def _tool_binding_is_lossless(tool: ToolDefinition) -> bool:
    """Return whether coverage semantics survive the artifact boundary exactly."""
    projection = _stable_bounded_projection(tool.input_schema)
    return (
        not _contains_artifact_loss_sentinel(tool.name)
        and redact_log_text(tool.name) == tool.name
        and projection.lossless
    )


def _input_schema_digest(input_schema: object) -> str:
    projection = _stable_bounded_projection(input_schema)
    return canonical_hash(
        {
            "contract": _SCHEMA_PROJECTION_CONTRACT,
            "lossless": projection.lossless,
            "value": projection.value,
        }
    )


def _scenario_digest(scenarios: Sequence[Scenario]) -> str:
    # Display IDs occur in evidence, so binding only behavioral fingerprints
    # would let a relabelled source validate while its evidence referred to an
    # absent scenario. Sorting retains duplicates while removing input order.
    identities = sorted(
        (scenario.scenario_id, scenario.fingerprint) for scenario in scenarios
    )
    return canonical_hash(_artifact_normalize(identities))


def _spec_digest(spec: AgentSpec) -> str:
    """Bind exactly the declared spec fields that determine coverage meaning."""

    projection = {
        "tools": [
            {
                "authoritative": item.authoritative,
                "name": item.value.name,
                "input_schema_digest": _input_schema_digest(
                    item.value.input_schema
                ),
                "state_changing": item.value.state_changing,
                "destructive": item.value.destructive,
                "replaceable": item.value.replaceable,
            }
            for item in sorted(spec.tools.items, key=lambda item: item.value.name)
        ],
        "tool_policies": [
            {
                "authoritative": item.authoritative,
                "tool_name": item.value.tool_name,
                "confirmation_required": item.value.confirmation_required,
                "idempotent": item.value.idempotent,
                "max_retries": item.value.max_retries,
            }
            for item in sorted(
                spec.tool_policies.items,
                key=lambda item: (item.value.tool_name, item.value.policy_id),
            )
        ],
    }
    return canonical_hash(_artifact_normalize(projection))


def _add_seed(
    seeds: dict[tuple[BehavioralDimension, str], _Seed],
    dimension: BehavioralDimension,
    subject: str,
    applicability: _Applicability,
    *,
    tool_name: str | None = None,
    prerequisite_tool_name: str | None = None,
) -> None:
    key = (dimension, subject)
    candidate = _Seed(
        dimension=dimension,
        subject=subject,
        applicability=applicability,
        tool_name=tool_name,
        prerequisite_tool_name=prerequisite_tool_name,
    )
    existing = seeds.get(key)
    if existing is None or _APPLICABILITY_RANK[applicability] > _APPLICABILITY_RANK[
        existing.applicability
    ]:
        seeds[key] = candidate


def _focal_tools(scenario: Scenario) -> tuple[str, ...]:
    names: set[str] = set()
    for behavior in (
        *scenario.required_tool_behavior,
        *scenario.allowed_tool_behavior,
        *scenario.forbidden_tool_behavior,
    ):
        names.add(behavior.tool_name)
    for trajectory in scenario.trajectory_constraints:
        if trajectory.kind not in _TOOL_SCOPED_TRAJECTORY_KINDS:
            continue
        tool_name = trajectory.parameters.get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            names.add(tool_name)
    return tuple(sorted(names))


def _confirmation_tools(scenario: Scenario) -> tuple[str, ...]:
    names = {
        constraint.tool_name
        for constraint in (
            *scenario.required_tool_behavior,
            *scenario.allowed_tool_behavior,
            *scenario.forbidden_tool_behavior,
        )
        if constraint.confirmation_required_before_call
    }
    for constraint in scenario.trajectory_constraints:
        if constraint.kind is not TrajectoryConstraintKind.CONFIRMATION_BEFORE_TOOL:
            continue
        tool_name = constraint.parameters.get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            names.add(tool_name)
    return tuple(sorted(names))


def _prerequisite_pairs(scenario: Scenario) -> tuple[tuple[str, str], ...]:
    declared_focal = set(_focal_tools(scenario))
    tags = set(scenario.dimension_tags)
    # Fixture IDs are a generator implementation detail, not a general
    # prerequisite declaration. Recognize them only in an unambiguous generated
    # action case whose source and focal-tool tags state that convention.
    if not tags.intersection(_GENERATOR_ACTION_SOURCES):
        return ()
    focal = tuple(
        sorted(
            tag[5:]
            for tag in tags
            if tag.startswith("tool:") and tag[5:] in declared_focal
        )
    )
    if len(focal) != 1:
        return ()
    prerequisites = {
        fixture.tool_name
        for fixture in scenario.tool_fixtures
        if fixture.fixture_id.startswith(_PREREQUISITE_PREFIXES)
    }
    return tuple(
        sorted(
            (focal_tool, prerequisite)
            for focal_tool in focal
            for prerequisite in prerequisites
            if prerequisite != focal_tool
        )
    )


def _seed_from_spec(
    spec: AgentSpec, seeds: dict[tuple[BehavioralDimension, str], _Seed]
) -> None:
    risk_dimensions = (
        BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
        BehavioralDimension.DUPLICATE_ACTION,
        BehavioralDimension.AMBIGUOUS_OUTCOME,
        BehavioralDimension.RETRY_CONTROL,
    )
    for tool_item in sorted(spec.tools.items, key=lambda value: value.value.name):
        tool = tool_item.value
        subject = _tool_subject(tool.name)
        for dimension in (
            BehavioralDimension.SUCCESS_PATH,
            BehavioralDimension.FAILURE_HANDLING,
            BehavioralDimension.TIMEOUT_HANDLING,
        ):
            _add_seed(
                seeds,
                dimension,
                subject,
                _Applicability.APPLICABLE,
                tool_name=tool.name,
            )

        if not tool_item.authoritative:
            for dimension in risk_dimensions:
                _add_seed(
                    seeds,
                    dimension,
                    subject,
                    _Applicability.UNKNOWN,
                    tool_name=tool.name,
                )
            continue

        state_changing = tool.state_changing or tool.destructive
        if state_changing:
            for dimension in (
                BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
                BehavioralDimension.DUPLICATE_ACTION,
            ):
                _add_seed(
                    seeds,
                    dimension,
                    subject,
                    _Applicability.APPLICABLE,
                    tool_name=tool.name,
                )
        if tool.destructive:
            for dimension in (
                BehavioralDimension.AMBIGUOUS_OUTCOME,
                BehavioralDimension.RETRY_CONTROL,
            ):
                _add_seed(
                    seeds,
                    dimension,
                    subject,
                    _Applicability.APPLICABLE,
                    tool_name=tool.name,
                )

    for policy_item in sorted(
        spec.tool_policies.items,
        key=lambda value: (value.value.tool_name, value.value.policy_id),
    ):
        policy = policy_item.value
        subject = _tool_subject(policy.tool_name)
        applicability = (
            _Applicability.APPLICABLE
            if policy_item.authoritative
            else _Applicability.UNKNOWN
        )
        if policy.confirmation_required is True:
            for dimension in (
                BehavioralDimension.CONFIRMATION_WITHOUT_CONSENT,
                BehavioralDimension.CONFIRMATION_WITH_CONSENT,
            ):
                _add_seed(
                    seeds,
                    dimension,
                    subject,
                    applicability,
                    tool_name=policy.tool_name,
                )
        if policy.idempotent is False:
            for dimension in (
                BehavioralDimension.DUPLICATE_ACTION,
                BehavioralDimension.AMBIGUOUS_OUTCOME,
                BehavioralDimension.RETRY_CONTROL,
            ):
                _add_seed(
                    seeds,
                    dimension,
                    subject,
                    applicability,
                    tool_name=policy.tool_name,
                )
        if policy.max_retries is not None:
            _add_seed(
                seeds,
                BehavioralDimension.RETRY_CONTROL,
                subject,
                applicability,
                tool_name=policy.tool_name,
            )


def _scenario_risk_applicability(
    scenario: Scenario,
    tool_name: str | None,
    non_authoritative_risk_tools: set[str],
) -> _Applicability:
    if (
        tool_name in non_authoritative_risk_tools
        and "source:behavioral_outcome" in scenario.dimension_tags
    ):
        # These generator variants are selected from the same inferred tool
        # flags; their controlled-world oracle is not an independent risk
        # declaration and must not launder UNKNOWN into applicability.
        return _Applicability.UNKNOWN
    return _Applicability.APPLICABLE


def _seed_from_scenarios(
    scenarios: Sequence[Scenario],
    seeds: dict[tuple[BehavioralDimension, str], _Seed],
    non_authoritative_risk_tools: set[str],
) -> None:
    for scenario in scenarios:
        focal_tools = _focal_tools(scenario)
        for criterion in scenario.output_criteria:
            if criterion.kind is OutputCriterionKind.NO_FABRICATED_SUCCESS:
                for tool_name in focal_tools:
                    _add_seed(
                        seeds,
                        BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
                        _tool_subject(tool_name),
                        _scenario_risk_applicability(
                            scenario,
                            tool_name,
                            non_authoritative_risk_tools,
                        ),
                        tool_name=tool_name,
                    )

        for constraint in scenario.trajectory_constraints:
            tool_name_value = constraint.parameters.get("tool_name")
            trajectory_tool = (
                tool_name_value
                if isinstance(tool_name_value, str) and tool_name_value
                else None
            )
            subject = (
                _tool_subject(trajectory_tool)
                if trajectory_tool is not None
                else f"scenario:{scenario.scenario_id}"
            )
            if constraint.kind is TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT:
                _add_seed(
                    seeds,
                    BehavioralDimension.DUPLICATE_ACTION,
                    subject,
                    _scenario_risk_applicability(
                        scenario,
                        trajectory_tool,
                        non_authoritative_risk_tools,
                    ),
                    tool_name=trajectory_tool,
                )
            elif constraint.kind is TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT:
                for dimension in (
                    BehavioralDimension.AMBIGUOUS_OUTCOME,
                    BehavioralDimension.RETRY_CONTROL,
                ):
                    _add_seed(
                        seeds,
                        dimension,
                        subject,
                        _scenario_risk_applicability(
                            scenario,
                            trajectory_tool,
                            non_authoritative_risk_tools,
                        ),
                        tool_name=trajectory_tool,
                    )
            elif constraint.kind is TrajectoryConstraintKind.MAX_RETRIES:
                _add_seed(
                    seeds,
                    BehavioralDimension.RETRY_CONTROL,
                    subject,
                    _scenario_risk_applicability(
                        scenario,
                        trajectory_tool,
                        non_authoritative_risk_tools,
                    ),
                    tool_name=trajectory_tool,
                )
            elif constraint.kind is TrajectoryConstraintKind.ORDERING:
                _add_seed(
                    seeds,
                    BehavioralDimension.ORDERING,
                    subject,
                    _Applicability.APPLICABLE,
                    tool_name=trajectory_tool,
                )

        for tool_name in _confirmation_tools(scenario):
            for dimension in (
                BehavioralDimension.CONFIRMATION_WITHOUT_CONSENT,
                BehavioralDimension.CONFIRMATION_WITH_CONSENT,
            ):
                _add_seed(
                    seeds,
                    dimension,
                    _tool_subject(tool_name),
                    _Applicability.APPLICABLE,
                    tool_name=tool_name,
                )

        for focal_tool, prerequisite_tool in _prerequisite_pairs(scenario):
            subject = _prerequisite_subject(focal_tool, prerequisite_tool)
            for dimension in (
                BehavioralDimension.PREREQUISITE_SUCCESS,
                BehavioralDimension.PREREQUISITE_FAILURE,
            ):
                _add_seed(
                    seeds,
                    dimension,
                    subject,
                    _Applicability.APPLICABLE,
                    tool_name=focal_tool,
                    prerequisite_tool_name=prerequisite_tool,
                )
            _add_seed(
                seeds,
                BehavioralDimension.ORDERING,
                subject,
                _Applicability.APPLICABLE,
                tool_name=focal_tool,
                prerequisite_tool_name=prerequisite_tool,
            )


def _controlled_slots(scenario: Scenario) -> tuple[_Slot, ...]:
    fault_slots = {
        (fault.tool_name, fault.invocation_index) for fault in scenario.injected_faults
    }
    baseline = [
        fixture
        for fixture in scenario.tool_fixtures
        if (fixture.tool_name, fixture.invocation_index) not in fault_slots
    ]
    slots = [
        _Slot(
            tool_name=fixture.tool_name,
            invocation_index=fixture.invocation_index,
            arguments_match=fixture.arguments_match,
            status=fixture.outcome.status,
            error_code=fixture.outcome.error_code,
            evidence_id=f"fixture:{fixture.fixture_id}",
            priority=fixture.priority,
            position=position,
            latency_ms=fixture.outcome.latency_ms,
            has_state_effects=bool(fixture.outcome.state_effects),
        )
        for position, fixture in enumerate(baseline)
    ]
    status_by_fault = {
        FaultType.ERROR: SimulatedToolStatus.ERROR,
        FaultType.TIMEOUT: SimulatedToolStatus.TIMEOUT,
        FaultType.MALFORMED_RESPONSE: SimulatedToolStatus.MALFORMED,
        FaultType.EMPTY_RESPONSE: SimulatedToolStatus.EMPTY,
        FaultType.PARTIAL_RESPONSE: SimulatedToolStatus.PARTIAL,
        FaultType.STALE_RESPONSE: SimulatedToolStatus.STALE,
    }
    slots.extend(
        _Slot(
            tool_name=fault.tool_name,
            invocation_index=fault.invocation_index,
            arguments_match={},
            status=status_by_fault[fault.fault_type],
            # The runtime maps injected timeout to "timeout", not
            # "ambiguous_timeout".  Do not upgrade it to ambiguity here.
            error_code=fault.fault_type.value
            if fault.fault_type in {FaultType.ERROR, FaultType.TIMEOUT}
            else None,
            evidence_id=f"fault:{fault.fault_id}",
            priority=100,
            position=len(baseline) + position,
            latency_ms=fault.latency_ms or 0.0,
            has_state_effects=False,
        )
        for position, fault in enumerate(scenario.injected_faults)
    )
    return tuple(slots)


def _slots_for(scenario: Scenario, tool_name: str) -> tuple[_Slot, ...]:
    return tuple(
        slot for slot in _controlled_slots(scenario) if slot.tool_name == tool_name
    )


def _recursive_subset(expected: object, actual: object) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _recursive_subset(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and expected == actual
    return expected == actual


def _argument_specificity(value: object) -> int:
    if isinstance(value, Mapping):
        return sum(1 + _argument_specificity(item) for item in value.values())
    if isinstance(value, list):
        return len(value) + sum(_argument_specificity(item) for item in value)
    return 1


def _schema_allows(tool: ToolDefinition | None, arguments: Mapping[str, object]) -> bool:
    # A schema this analyzer cannot evaluate offline yields no controlled
    # reach. An unresolvable local reference only surfaces while validating,
    # not while building, so both steps fail closed rather than escaping and
    # aborting a run over derived report metadata.
    if tool is None or not tool.replaceable:
        return False
    try:
        validator = offline_validator(tool.input_schema, check_formats=True)
        return not any(validator.iter_errors(dict(arguments)))
    except (SchemaError, UnsafeSchemaReference, Unresolvable):
        return False


def _selected_trace(
    scenario: Scenario,
    tool: ToolDefinition | None,
    tool_name: str,
    arguments: Mapping[str, object],
    invocation_count: int,
) -> tuple[_Slot, ...]:
    """Simulate gateway selection for one declared action signature.

    Slots are consumed, priority/invocation specificity wins, and an equal best
    score stops the trace because the gateway rejects ambiguous fixtures. This
    remains pure contract analysis: no gateway, target, or handler is invoked.
    """

    if invocation_count < 1 or tool is None or tool.name != tool_name:
        return ()
    if not _schema_allows(tool, arguments):
        return ()
    slots = _slots_for(scenario, tool_name)
    used: set[int] = set()
    selected: list[_Slot] = []
    elapsed_ms = 0.0
    reachable_count = min(invocation_count, scenario.resource_budgets.max_tool_calls)
    for invocation in range(1, reachable_count + 1):
        candidates: list[tuple[tuple[int, int, int], _Slot]] = []
        for slot in slots:
            if slot.position in used:
                continue
            if slot.invocation_index not in {None, invocation}:
                continue
            if not _recursive_subset(slot.arguments_match, arguments):
                continue
            score = (
                slot.priority,
                int(slot.invocation_index is not None),
                _argument_specificity(slot.arguments_match),
            )
            candidates.append((score, slot))
        if not candidates:
            break
        best_score = max(score for score, _ in candidates)
        best = [slot for score, slot in candidates if score == best_score]
        if len(best) != 1:
            break
        chosen = best[0]
        if (
            elapsed_ms + chosen.latency_ms
            > scenario.resource_budgets.wall_clock_seconds * 1000
        ):
            break
        elapsed_ms += chosen.latency_ms
        used.add(chosen.position)
        selected.append(chosen)
        if chosen.has_state_effects:
            # This outcome is itself controlled and observable, so it stays in
            # the trace. It also mutates the simulated world, and any later
            # invocation then depends on state this pure contract analysis does
            # not evaluate, so stop instead of claiming the next slot is
            # reachable.
            break
    return tuple(selected)


def _trace_for_constraint(
    scenario: Scenario,
    tool: ToolDefinition | None,
    constraint: ToolBehaviorConstraint,
    invocation_count: int,
) -> tuple[_Slot, ...]:
    return _selected_trace(
        scenario,
        tool,
        constraint.tool_name,
        constraint.arguments_match,
        invocation_count,
    )


def _authoritative_oracle_ids(scenario: Scenario) -> set[str]:
    return {
        oracle.oracle_id
        for oracle in scenario.oracle_provenance
        if oracle.supports_hard_failure
        and oracle.confidence >= 0.8
        and oracle.strength is not OracleStrength.LLM_INFERENCE
    }


def _criterion_is_authoritative(
    scenario: Scenario,
    criterion: object,
) -> bool:
    oracle_ids = getattr(criterion, "oracle_ids", ())
    return bool(
        set(oracle_ids).intersection(_authoritative_oracle_ids(scenario))
    )


def _action_contracts(
    scenario: Scenario, tool_name: str
) -> tuple[
    tuple[ToolBehaviorConstraint, ...],
    tuple[ToolBehaviorConstraint, ...],
    tuple[ToolBehaviorConstraint, ...],
]:
    required = tuple(
        item
        for item in scenario.required_tool_behavior
        if item.tool_name == tool_name and item.min_calls >= 1
    )
    allowed = tuple(
        item for item in scenario.allowed_tool_behavior if item.tool_name == tool_name
    )
    forbidden = tuple(
        item for item in scenario.forbidden_tool_behavior if item.tool_name == tool_name
    )
    return required, allowed, forbidden


def _traces_for_constraints(
    scenario: Scenario,
    tools: Mapping[str, ToolDefinition],
    constraints: Sequence[ToolBehaviorConstraint],
    invocation_count: int,
) -> tuple[tuple[ToolBehaviorConstraint, tuple[_Slot, ...]], ...]:
    return tuple(
        (
            constraint,
            _trace_for_constraint(
                scenario,
                tools.get(constraint.tool_name),
                constraint,
                invocation_count,
            ),
        )
        for constraint in constraints
    )


def _trace_slots(traces: Iterable[tuple[object, tuple[_Slot, ...]]]) -> tuple[_Slot, ...]:
    return tuple(slot for _, trace in traces for slot in trace)


def _scenario_evidence(
    scenario: Scenario,
    *,
    criteria: Iterable[object] = (),
    slots: Iterable[_Slot] = (),
) -> tuple[str, ...]:
    values = {
        f"scenario:{scenario.scenario_id}",
        f"scenario-fingerprint:{scenario.fingerprint}",
    }
    values.update(slot.evidence_id for slot in slots)
    for criterion in criteria:
        criterion_id = getattr(criterion, "criterion_id", None)
        if isinstance(criterion_id, str) and criterion_id:
            values.add(f"criterion:{criterion_id}")
    return tuple(sorted(values))


def _best(candidates: Sequence[_Outcome], missing_reason: str) -> _Outcome:
    if not candidates:
        return _Outcome(BehavioralCoverageStatus.MISSING, missing_reason, ())
    highest = max(_OUTCOME_RANK[item.status] for item in candidates)
    best = [item for item in candidates if _OUTCOME_RANK[item.status] == highest]
    reason = min(item.reason_code for item in best)
    evidence = tuple(sorted({value for item in best for value in item.evidence}))
    return _Outcome(best[0].status, reason, evidence)


def _success_outcome(
    scenarios: Sequence[Scenario], tools: Mapping[str, ToolDefinition], tool_name: str
) -> _Outcome:
    candidates: list[_Outcome] = []
    for scenario in scenarios:
        required, allowed, _ = _action_contracts(scenario, tool_name)
        if not required and not allowed:
            continue
        traces = _traces_for_constraints(scenario, tools, (*required, *allowed), 1)
        successful = tuple(
            (constraint, trace)
            for constraint, trace in traces
            if trace and trace[0].status is SimulatedToolStatus.SUCCESS
        )
        if not successful:
            continue
        hard_required = any(
            constraint in required
            and _criterion_is_authoritative(scenario, constraint)
            for constraint, _ in successful
        )
        criteria = (*required, *allowed)
        if hard_required:
            candidates.append(
                _Outcome(
                    BehavioralCoverageStatus.COVERED,
                    "required_success_with_controlled_fixture",
                    _scenario_evidence(
                        scenario, criteria=criteria, slots=_trace_slots(successful)
                    ),
                )
            )
        else:
            candidates.append(
                _Outcome(
                    BehavioralCoverageStatus.PARTIAL,
                    "optional_success_path",
                    _scenario_evidence(
                        scenario, criteria=criteria, slots=_trace_slots(successful)
                    ),
                )
            )
    return _best(candidates, "success_path_missing")


def _handling_criteria(scenario: Scenario) -> tuple[OutputCriterion, ...]:
    output = tuple(
        criterion
        for criterion in scenario.output_criteria
        if criterion.kind
        in {
            OutputCriterionKind.NO_FABRICATED_SUCCESS,
            OutputCriterionKind.ACKNOWLEDGES_TOOL_ERROR,
        }
    )
    return output


def _handling_is_authoritative(scenario: Scenario, criterion: object) -> bool:
    required = bool(getattr(criterion, "required", True))
    return required and _criterion_is_authoritative(scenario, criterion)


def _has_configured_success_terms(criterion: OutputCriterion) -> bool:
    terms = criterion.parameters.get("success_terms")
    return (
        isinstance(terms, list)
        and bool(terms)
        and all(isinstance(term, str) and bool(term) for term in terms)
    )


def _valid_retry_limit(criterion: TrajectoryConstraint) -> int | None:
    maximum = criterion.parameters.get("max_retries")
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum < 0
    ):
        return None
    return maximum


def _handling_criterion_is_executable(
    scenario: Scenario,
    criterion: object,
    *,
    ambiguous_timeout: bool = False,
) -> bool:
    if not _handling_is_authoritative(scenario, criterion):
        return False
    if isinstance(criterion, OutputCriterion):
        if criterion.kind is OutputCriterionKind.NO_FABRICATED_SUCCESS:
            return _has_configured_success_terms(criterion)
        return criterion.kind is OutputCriterionKind.ACKNOWLEDGES_TOOL_ERROR
    if isinstance(criterion, TrajectoryConstraint):
        if criterion.kind is TrajectoryConstraintKind.MAX_RETRIES:
            return _valid_retry_limit(criterion) is not None
        if criterion.kind is TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT:
            return ambiguous_timeout
        return False
    return True


def _only_unconfigured_fabrication(criteria: Sequence[object]) -> bool:
    output = [item for item in criteria if isinstance(item, OutputCriterion)]
    return bool(output) and len(output) == len(criteria) and all(
        item.kind is OutputCriterionKind.NO_FABRICATED_SUCCESS
        and not _has_configured_success_terms(item)
        for item in output
    )


def _failure_outcome(
    scenarios: Sequence[Scenario], tools: Mapping[str, ToolDefinition], tool_name: str
) -> _Outcome:
    candidates: list[_Outcome] = []
    for scenario in scenarios:
        required, allowed, _ = _action_contracts(scenario, tool_name)
        if not required and not allowed:
            continue
        traces = _traces_for_constraints(scenario, tools, (*required, *allowed), 1)
        failures = tuple(
            (constraint, trace)
            for constraint, trace in traces
            if trace and trace[0].status is SimulatedToolStatus.ERROR
        )
        if not failures:
            continue
        criteria = _handling_criteria(scenario)
        evidence = _scenario_evidence(
            scenario, criteria=criteria, slots=_trace_slots(failures)
        )
        hard = _focal_tools(scenario) == (tool_name,) and any(
            _handling_criterion_is_executable(scenario, item)
            for item in criteria
        )
        hard_required = any(
            constraint in required
            and _criterion_is_authoritative(scenario, constraint)
            for constraint, _ in failures
        )
        if not criteria:
            candidates.append(
                _Outcome(
                    BehavioralCoverageStatus.PARTIAL,
                    "failure_stimulus_without_handling_oracle",
                    evidence,
                )
            )
        elif _only_unconfigured_fabrication(criteria):
            candidates.append(
                _Outcome(
                    BehavioralCoverageStatus.PARTIAL,
                    "fabrication_terms_unconfigured",
                    evidence,
                )
            )
        elif hard and hard_required:
            candidates.append(
                _Outcome(
                    BehavioralCoverageStatus.COVERED,
                    "required_failure_with_supported_oracle",
                    evidence,
                )
            )
        else:
            candidates.append(
                _Outcome(
                    BehavioralCoverageStatus.PARTIAL,
                    "optional_or_nonauthoritative_failure_path",
                    evidence,
                )
            )
    return _best(candidates, "failure_path_missing")


def _timeout_outcome(
    scenarios: Sequence[Scenario], tools: Mapping[str, ToolDefinition], tool_name: str
) -> _Outcome:
    candidates: list[_Outcome] = []
    for scenario in scenarios:
        required, allowed, _ = _action_contracts(scenario, tool_name)
        if not required and not allowed:
            continue
        action_constraints = (*required, *allowed)
        traces = _traces_for_constraints(scenario, tools, action_constraints, 1)
        timeouts = tuple(
            (constraint, trace)
            for constraint, trace in traces
            if trace and trace[0].status is SimulatedToolStatus.TIMEOUT
        )
        if not timeouts:
            continue
        tool = tools.get(tool_name)
        trajectory = tuple(
            constraint
            for constraint in scenario.trajectory_constraints
            if constraint.parameters.get("tool_name") == tool_name
            and constraint.kind
            in {
                TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT,
                TrajectoryConstraintKind.MAX_RETRIES,
            }
        )
        handling = _handling_criteria(scenario)
        criteria = (*handling, *trajectory)
        extended: list[tuple[ToolBehaviorConstraint, tuple[_Slot, ...]]] = []
        hard_path = False
        for action, first_trace in timeouts:
            hard_action = (
                action in required
                and _criterion_is_authoritative(scenario, action)
            )
            if not hard_action:
                continue
            for criterion in criteria:
                if isinstance(criterion, TrajectoryConstraint):
                    if not (
                        criterion.required
                        and _criterion_is_authoritative(scenario, criterion)
                    ):
                        continue
                    if criterion.kind is TrajectoryConstraintKind.MAX_RETRIES:
                        maximum = _valid_retry_limit(criterion)
                        if maximum is None:
                            continue
                        needed = maximum + 2
                        trace = _trace_for_constraint(
                            scenario, tool, action, needed
                        )
                        extended.append((action, trace))
                        hard_path = hard_path or (
                            len(trace) >= needed
                            and trace[0].status is SimulatedToolStatus.TIMEOUT
                        )
                    elif (
                        criterion.kind
                        is TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT
                    ):
                        trace = _trace_for_constraint(
                            scenario,
                            tool,
                            action,
                            _maximum_trace_length(scenario, tool_name),
                        )
                        extended.append((action, trace))
                        hard_path = hard_path or _ambiguous_with_controlled_next(
                            trace, tool
                        )
                elif (
                    _focal_tools(scenario) == (tool_name,)
                    and _handling_criterion_is_executable(
                        scenario,
                        criterion,
                        ambiguous_timeout=_is_ambiguous_timeout(
                            first_trace[0], tool
                        ),
                    )
                ):
                    hard_path = True
        evidence = _scenario_evidence(
            scenario,
            criteria=criteria,
            slots=(*_trace_slots(timeouts), *_trace_slots(extended)),
        )
        if not criteria:
            status = BehavioralCoverageStatus.PARTIAL
            reason = "timeout_stimulus_without_handling_oracle"
        elif _only_unconfigured_fabrication(criteria):
            status = BehavioralCoverageStatus.PARTIAL
            reason = "fabrication_terms_unconfigured"
        elif hard_path:
            status = BehavioralCoverageStatus.COVERED
            reason = "required_timeout_with_supported_oracle"
        else:
            status = BehavioralCoverageStatus.PARTIAL
            reason = "optional_or_nonauthoritative_timeout_path"
        candidates.append(_Outcome(status, reason, evidence))
    return _best(candidates, "timeout_path_missing")


def _fabrication_outcome(
    scenarios: Sequence[Scenario], tools: Mapping[str, ToolDefinition], tool_name: str
) -> _Outcome:
    candidates: list[_Outcome] = []
    for scenario in scenarios:
        if tool_name not in _focal_tools(scenario):
            continue
        criteria = tuple(
            criterion
            for criterion in scenario.output_criteria
            if criterion.kind is OutputCriterionKind.NO_FABRICATED_SUCCESS
        )
        if not criteria:
            continue
        required, allowed, _ = _action_contracts(scenario, tool_name)
        traces = _traces_for_constraints(scenario, tools, (*required, *allowed), 1)
        failed = tuple(
            (constraint, trace)
            for constraint, trace in traces
            if trace and trace[0].status in UNUSABLE_SIMULATED_STATUSES
        )
        hard_configured = any(
            criterion.required and _criterion_is_authoritative(scenario, criterion)
            and _has_configured_success_terms(criterion)
            for criterion in criteria
        ) and _focal_tools(scenario) == (tool_name,)
        configured = any(
            _has_configured_success_terms(criterion) for criterion in criteria
        )
        hard_required = any(
            constraint in required
            and _criterion_is_authoritative(scenario, constraint)
            for constraint, _ in failed
        )
        evidence = _scenario_evidence(
            scenario, criteria=criteria, slots=_trace_slots(failed)
        )
        if not failed:
            status = BehavioralCoverageStatus.PARTIAL
            reason = "fabrication_oracle_without_failure_stimulus"
        elif not configured:
            status = BehavioralCoverageStatus.PARTIAL
            reason = "fabrication_terms_unconfigured"
        elif hard_configured and hard_required:
            status = BehavioralCoverageStatus.COVERED
            reason = "configured_fabrication_oracle_on_required_failure"
        else:
            status = BehavioralCoverageStatus.PARTIAL
            reason = "optional_or_nonauthoritative_fabrication_path"
        candidates.append(_Outcome(status, reason, evidence))
    return _best(candidates, "fabricated_success_case_missing")


def _constraint_tool_name(constraint: TrajectoryConstraint) -> str | None:
    value = constraint.parameters.get("tool_name")
    return value if isinstance(value, str) and value else None


def _matching_trajectory(
    scenario: Scenario,
    tool_name: str | None,
    kinds: set[TrajectoryConstraintKind],
) -> tuple[TrajectoryConstraint, ...]:
    """Return constraints whose declared tool signature is exactly ``tool_name``.

    ``None`` selects only constraints that declare no tool. A constraint that
    names a tool belongs to that tool's requirement, never to the scenario-
    scoped requirement created for an undeclared signature.
    """

    return tuple(
        constraint
        for constraint in scenario.trajectory_constraints
        if constraint.kind in kinds
        and _constraint_tool_name(constraint) == tool_name
    )


def _is_ambiguous_timeout(slot: _Slot, tool: ToolDefinition | None) -> bool:
    return slot.status is SimulatedToolStatus.TIMEOUT and (
        slot.error_code == "ambiguous_timeout"
        or bool(tool is not None and tool.state_changing)
    )


def _ambiguous_with_controlled_next(
    trace: Sequence[_Slot], tool: ToolDefinition | None
) -> bool:
    return any(
        _is_ambiguous_timeout(slot, tool) and index + 1 < len(trace)
        for index, slot in enumerate(trace)
    )


def _maximum_trace_length(scenario: Scenario, tool_name: str) -> int:
    return max(1, len(_slots_for(scenario, tool_name)))


def _ambiguous_outcome(
    scenarios: Sequence[Scenario],
    tools: Mapping[str, ToolDefinition],
    tool_name: str | None,
) -> _Outcome:
    candidates: list[_Outcome] = []
    for scenario in scenarios:
        constraints = _matching_trajectory(
            scenario,
            tool_name,
            {TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT},
        )
        if tool_name is None:
            if constraints:
                candidates.append(
                    _Outcome(
                        BehavioralCoverageStatus.PARTIAL,
                        "ambiguous_oracle_without_declared_tool_signature",
                        _scenario_evidence(scenario, criteria=constraints),
                    )
                )
            continue
        if tool_name not in _focal_tools(scenario):
            continue
        required, allowed, _ = _action_contracts(scenario, tool_name)
        if not required and not allowed:
            continue
        traces = _traces_for_constraints(
            scenario,
            tools,
            (*required, *allowed),
            _maximum_trace_length(scenario, tool_name),
        )
        tool = tools.get(tool_name)
        ambiguous = tuple(
            (constraint, trace)
            for constraint, trace in traces
            if any(_is_ambiguous_timeout(slot, tool) for slot in trace)
        )
        observable = tuple(
            (constraint, trace)
            for constraint, trace in ambiguous
            if _ambiguous_with_controlled_next(trace, tool)
        )
        if not constraints and not ambiguous:
            continue
        hard = any(
            constraint.required and _criterion_is_authoritative(scenario, constraint)
            for constraint in constraints
        )
        hard_observable = any(
            constraint in required
            and _criterion_is_authoritative(scenario, constraint)
            for constraint, _ in observable
        )
        evidence = _scenario_evidence(
            scenario,
            criteria=constraints,
            slots=_trace_slots(ambiguous),
        )
        if not ambiguous:
            status = BehavioralCoverageStatus.PARTIAL
            reason = "ambiguous_oracle_without_reachable_stimulus"
        elif not constraints:
            status = BehavioralCoverageStatus.PARTIAL
            reason = "ambiguous_stimulus_without_retry_oracle"
        elif not observable:
            status = BehavioralCoverageStatus.PARTIAL
            reason = "ambiguous_retry_lacks_controlled_next_attempt"
        elif hard and hard_observable:
            status = BehavioralCoverageStatus.COVERED
            reason = "required_ambiguous_retry_is_observable"
        else:
            status = BehavioralCoverageStatus.PARTIAL
            reason = (
                "optional_ambiguous_action_path"
                if allowed
                else "nonauthoritative_ambiguous_oracle"
            )
        candidates.append(_Outcome(status, reason, evidence))
    return _best(candidates, "ambiguous_outcome_case_missing")


def _retry_outcome(
    scenarios: Sequence[Scenario],
    tools: Mapping[str, ToolDefinition],
    tool_name: str | None,
) -> _Outcome:
    candidates: list[_Outcome] = []
    for scenario in scenarios:
        constraints = _matching_trajectory(
            scenario,
            tool_name,
            {
                TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT,
                TrajectoryConstraintKind.MAX_RETRIES,
            },
        )
        if not constraints:
            continue
        if tool_name is None:
            candidates.append(
                _Outcome(
                    BehavioralCoverageStatus.PARTIAL,
                    "retry_oracle_without_declared_tool_signature",
                    _scenario_evidence(scenario, criteria=constraints),
                )
            )
            continue
        required, allowed, _ = (
            _action_contracts(scenario, tool_name)
            if tool_name is not None else ((), (), ())
        )
        tool = tools.get(tool_name)
        for constraint in constraints:
            if constraint.kind is TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT:
                needed = _maximum_trace_length(scenario, tool_name)
                traces = tuple(
                    (
                        action,
                        _trace_for_constraint(scenario, tool, action, needed),
                    )
                    for action in (*required, *allowed)
                )
                observable = tuple(
                    (action, trace)
                    for action, trace in traces
                    if _ambiguous_with_controlled_next(trace, tool)
                )
            else:
                maximum = _valid_retry_limit(constraint)
                if maximum is None:
                    candidates.append(
                        _Outcome(
                            BehavioralCoverageStatus.PARTIAL,
                            "retry_limit_parameters_invalid",
                            _scenario_evidence(scenario, criteria=(constraint,)),
                        )
                    )
                    continue
                needed = maximum + 2
                traces = tuple(
                    (
                        action,
                        _trace_for_constraint(scenario, tool, action, needed),
                    )
                    for action in (*required, *allowed)
                )
                observable = tuple(
                    (action, trace)
                    for action, trace in traces
                    if len(trace) >= needed
                )
            hard = constraint.required and _criterion_is_authoritative(
                scenario, constraint
            )
            hard_required = any(
                action in required
                and _criterion_is_authoritative(scenario, action)
                for action, _ in observable
            )
            evidence = _scenario_evidence(
                scenario,
                criteria=(constraint,),
                slots=_trace_slots(traces),
            )
            if not observable:
                status = BehavioralCoverageStatus.PARTIAL
                reason = "retry_limit_lacks_prohibited_attempt_fixture"
            elif hard and hard_required:
                status = BehavioralCoverageStatus.COVERED
                reason = "required_retry_limit_is_observable"
            else:
                status = BehavioralCoverageStatus.PARTIAL
                reason = (
                    "optional_retry_action_path"
                    if allowed
                    else "nonauthoritative_retry_oracle"
                )
            candidates.append(_Outcome(status, reason, evidence))
    return _best(candidates, "retry_control_case_missing")


def _duplicate_outcome(
    scenarios: Sequence[Scenario],
    tools: Mapping[str, ToolDefinition],
    tool_name: str | None,
) -> _Outcome:
    candidates: list[_Outcome] = []
    for scenario in scenarios:
        constraints = _matching_trajectory(
            scenario,
            tool_name,
            {TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT},
        )
        if not constraints:
            continue
        if tool_name is None:
            candidates.append(
                _Outcome(
                    BehavioralCoverageStatus.PARTIAL,
                    "duplicate_oracle_without_declared_tool_signature",
                    _scenario_evidence(scenario, criteria=constraints),
                )
            )
            continue
        required, allowed, _ = _action_contracts(scenario, tool_name)
        traces = _traces_for_constraints(
            scenario, tools, (*required, *allowed), 2
        )
        observable = tuple(
            (action, trace)
            for action, trace in traces
            if len(trace) >= 2
        )
        hard = any(
            constraint.required and _criterion_is_authoritative(scenario, constraint)
            for constraint in constraints
        )
        hard_required = any(
            action in required
            and _criterion_is_authoritative(scenario, action)
            for action, _ in observable
        )
        evidence = _scenario_evidence(
            scenario,
            criteria=constraints,
            slots=_trace_slots(traces),
        )
        if not observable:
            status = BehavioralCoverageStatus.PARTIAL
            reason = "duplicate_oracle_lacks_second_fixture"
        elif hard and hard_required:
            status = BehavioralCoverageStatus.COVERED
            reason = "required_duplicate_attempt_is_observable"
        else:
            status = BehavioralCoverageStatus.PARTIAL
            reason = (
                "optional_duplicate_action_path"
                if allowed
                else "nonauthoritative_duplicate_oracle"
            )
        candidates.append(_Outcome(status, reason, evidence))
    return _best(candidates, "duplicate_action_case_missing")


def _ordering_outcome(
    scenarios: Sequence[Scenario],
    tools: Mapping[str, ToolDefinition],
    tool_name: str | None,
) -> _Outcome:
    """Report whether a declared ordering relation can actually be observed.

    The evaluator proves ordering from what the agent had observed when it
    decided to act, so coverage requires both tools to be reachable under
    control: without a controlled outcome for the prerequisite there is no
    observation for the dependent call to have followed.
    """

    candidates: list[_Outcome] = []
    for scenario in scenarios:
        constraints = _matching_trajectory(
            scenario, tool_name, {TrajectoryConstraintKind.ORDERING}
        )
        if not constraints:
            continue
        if tool_name is None:
            candidates.append(
                _Outcome(
                    BehavioralCoverageStatus.PARTIAL,
                    "ordering_oracle_without_declared_tool_signature",
                    _scenario_evidence(scenario, criteria=constraints),
                )
            )
            continue
        required, allowed, _ = _action_contracts(scenario, tool_name)
        traces = _traces_for_constraints(scenario, tools, (*required, *allowed), 1)
        dependent_reachable = any(trace for _, trace in traces)
        for constraint in constraints:
            before = constraint.parameters.get("required_before")
            if not isinstance(before, str) or not before:
                candidates.append(
                    _Outcome(
                        BehavioralCoverageStatus.PARTIAL,
                        "ordering_prerequisite_not_declared",
                        _scenario_evidence(scenario, criteria=(constraint,)),
                    )
                )
                continue
            prior_required, prior_allowed, _ = _action_contracts(scenario, before)
            prior_traces = _traces_for_constraints(
                scenario, tools, (*prior_required, *prior_allowed), 1
            )
            prior_reachable = any(trace for _, trace in prior_traces)
            hard = constraint.required and _criterion_is_authoritative(
                scenario, constraint
            )
            evidence = _scenario_evidence(
                scenario,
                criteria=(constraint,),
                slots=(*_trace_slots(traces), *_trace_slots(prior_traces)),
            )
            if not (dependent_reachable and prior_reachable):
                status = BehavioralCoverageStatus.PARTIAL
                reason = "ordering_lacks_controlled_outcome_for_both_tools"
            elif hard:
                status = BehavioralCoverageStatus.COVERED
                reason = "required_ordering_is_observable"
            else:
                status = BehavioralCoverageStatus.PARTIAL
                reason = "nonauthoritative_ordering_oracle"
            candidates.append(_Outcome(status, reason, evidence))
    return _best(candidates, "ordering_case_missing")


def _has_explicit_confirmation(scenario: Scenario) -> bool:
    return any(
        turn.metadata.get("explicit_confirmation") is True
        for turn in (*scenario.conversation_turns, *scenario.followup_turns)
    )


def _confirmation_constraints(
    scenario: Scenario, tool_name: str
) -> tuple[object, ...]:
    behavior = tuple(
        constraint
        for constraint in (
            *scenario.required_tool_behavior,
            *scenario.allowed_tool_behavior,
            *scenario.forbidden_tool_behavior,
        )
        if constraint.tool_name == tool_name
        and constraint.confirmation_required_before_call
    )
    trajectory = tuple(
        constraint
        for constraint in scenario.trajectory_constraints
        if constraint.kind is TrajectoryConstraintKind.CONFIRMATION_BEFORE_TOOL
        and constraint.parameters.get("tool_name") == tool_name
    )
    return (*behavior, *trajectory)


def _confirmation_outcome(
    scenarios: Sequence[Scenario],
    tools: Mapping[str, ToolDefinition],
    tool_name: str,
    *,
    with_consent: bool,
) -> _Outcome:
    candidates: list[_Outcome] = []
    for scenario in scenarios:
        if tool_name not in _focal_tools(scenario):
            continue
        confirmed = _has_explicit_confirmation(scenario)
        if confirmed is not with_consent:
            continue
        constraints = _confirmation_constraints(scenario, tool_name)
        required, allowed, forbidden = _action_contracts(scenario, tool_name)
        if not constraints and not required and not allowed and not forbidden:
            continue
        polarity = required if with_consent else forbidden
        traces = _traces_for_constraints(scenario, tools, polarity, 1)
        observable = tuple(
            (action, trace)
            for action, trace in traces
            if trace
        )
        hard = any(
            _criterion_is_authoritative(scenario, item)
            and bool(getattr(item, "required", True))
            for item in constraints
        )
        strong_action = any(
            action in polarity
            and _criterion_is_authoritative(scenario, action)
            for action, _ in observable
        )
        evidence = _scenario_evidence(
            scenario,
            criteria=constraints,
            slots=_trace_slots(traces),
        )
        if not constraints:
            status = BehavioralCoverageStatus.PARTIAL
            reason = "confirmation_opportunity_without_oracle"
        elif not polarity:
            status = BehavioralCoverageStatus.PARTIAL
            reason = "confirmation_action_polarity_not_observable"
        elif not observable:
            status = BehavioralCoverageStatus.PARTIAL
            reason = "confirmation_oracle_without_controlled_tool_fixture"
        elif hard and strong_action:
            status = BehavioralCoverageStatus.COVERED
            reason = (
                "confirmed_required_action_is_observable"
                if with_consent
                else "unconfirmed_forbidden_action_is_observable"
            )
        else:
            status = BehavioralCoverageStatus.PARTIAL
            reason = (
                "optional_confirmed_action_path"
                if allowed
                else "nonauthoritative_confirmation_oracle"
            )
        candidates.append(_Outcome(status, reason, evidence))
    missing = (
        "confirmed_action_case_missing"
        if with_consent
        else "unconfirmed_action_case_missing"
    )
    return _best(candidates, missing)


def _prerequisite_outcome(
    scenarios: Sequence[Scenario],
    tools: Mapping[str, ToolDefinition],
    focal_tool: str,
    prerequisite_tool: str,
    *,
    failure: bool,
) -> _Outcome:
    candidates: list[_Outcome] = []
    pair = (focal_tool, prerequisite_tool)
    for scenario in scenarios:
        if pair not in _prerequisite_pairs(scenario):
            continue
        prerequisite_required = tuple(
            constraint
            for constraint in scenario.required_tool_behavior
            if constraint.tool_name == prerequisite_tool and constraint.min_calls >= 1
        )
        focal_required, _, focal_forbidden = _action_contracts(scenario, focal_tool)
        prerequisite_traces = _traces_for_constraints(
            scenario, tools, prerequisite_required, 1
        )
        selected_prerequisite = tuple(
            (action, trace)
            for action, trace in prerequisite_traces
            if trace
            and (
                trace[0].status
                in {SimulatedToolStatus.ERROR, SimulatedToolStatus.TIMEOUT}
                if failure
                else trace[0].status is SimulatedToolStatus.SUCCESS
            )
        )
        focal_constraints = focal_forbidden if failure else focal_required
        focal_traces = _traces_for_constraints(
            scenario, tools, focal_constraints, 1
        )
        selected_focal = tuple(
            (action, trace) for action, trace in focal_traces if trace
        )
        required_prerequisite = any(
            action in prerequisite_required
            and _criterion_is_authoritative(scenario, action)
            for action, _ in selected_prerequisite
        )
        asserted_focal = any(
            action in focal_constraints
            and _criterion_is_authoritative(scenario, action)
            for action, _ in selected_focal
        )
        if failure:
            asserted = required_prerequisite and asserted_focal
            criteria: tuple[object, ...] = (
                *prerequisite_required,
                *focal_forbidden,
            )
            status = (
                BehavioralCoverageStatus.COVERED
                if asserted
                else BehavioralCoverageStatus.PARTIAL
            )
            reason = (
                "required_prerequisite_failure_forbids_focal_action"
                if asserted
                else (
                    "prerequisite_failure_fixture_not_reachable"
                    if not selected_prerequisite
                    else "prerequisite_failure_path_not_fully_asserted"
                )
            )
        else:
            asserted = required_prerequisite and asserted_focal
            criteria = (*prerequisite_required, *focal_required)
            # Both calls can be asserted today, but not their ordering. Focal
            # first would otherwise create false full prerequisite coverage.
            status = BehavioralCoverageStatus.PARTIAL
            reason = (
                "calls_asserted_but_order_unobservable"
                if asserted
                else (
                    "prerequisite_success_fixture_not_reachable"
                    if not selected_prerequisite
                    else "prerequisite_success_relationship_unasserted"
                )
            )
        candidates.append(
            _Outcome(
                status,
                reason,
                _scenario_evidence(
                    scenario,
                    criteria=criteria,
                    slots=(
                        *_trace_slots(prerequisite_traces),
                        *_trace_slots(focal_traces),
                    ),
                ),
            )
        )
    missing = (
        "prerequisite_failure_path_missing"
        if failure
        else "prerequisite_success_path_missing"
    )
    return _best(candidates, missing)


def _ordering_evidence(
    scenarios: Sequence[Scenario], seed: _Seed
) -> tuple[str, ...]:
    evidence: set[str] = set()
    if seed.tool_name is not None and seed.prerequisite_tool_name is not None:
        pair = (seed.tool_name, seed.prerequisite_tool_name)
        for scenario in scenarios:
            if pair not in _prerequisite_pairs(scenario):
                continue
            evidence.update(
                _scenario_evidence(
                    scenario,
                    slots=_slots_for(scenario, seed.prerequisite_tool_name),
                )
            )
    for scenario in scenarios:
        for constraint in scenario.trajectory_constraints:
            if constraint.kind is not TrajectoryConstraintKind.ORDERING:
                continue
            tool_name = constraint.parameters.get("tool_name")
            subject = (
                _tool_subject(tool_name)
                if isinstance(tool_name, str) and tool_name
                else f"scenario:{scenario.scenario_id}"
            )
            if subject == seed.subject:
                evidence.update(
                    _scenario_evidence(scenario, criteria=(constraint,))
                )
    return tuple(sorted(evidence))


def _subject_scenarios(
    scenarios: Sequence[Scenario], subject: str
) -> tuple[Scenario, ...]:
    """Restrict a scenario-scoped requirement to the scenario that declared it.

    A requirement whose subject is one scenario must never borrow another
    scenario's stimulus as its evidence; a subject whose scenario is absent
    from the evidence set is missing, not partially covered.
    """

    return tuple(
        scenario
        for scenario in scenarios
        if f"scenario:{scenario.scenario_id}" == subject
    )


def _evaluate_seed(
    seed: _Seed,
    scenarios: Sequence[Scenario],
    tools: Mapping[str, ToolDefinition],
) -> _Outcome:
    if seed.applicability is _Applicability.UNKNOWN:
        return _Outcome(
            BehavioralCoverageStatus.UNKNOWN,
            seed.unknown_reason,
            (),
        )
    if seed.applicability is _Applicability.UNSUPPORTED:
        return _Outcome(
            BehavioralCoverageStatus.UNSUPPORTED,
            (
                "prerequisite_ordering_not_supported"
                if seed.prerequisite_tool_name is not None
                else "trajectory_ordering_not_supported"
            ),
            _ordering_evidence(scenarios, seed),
        )

    tool_name = seed.tool_name
    # An undeclared tool signature yields a scenario-scoped subject, so the
    # evidence set narrows to that one scenario.
    scoped = (
        scenarios
        if tool_name is not None
        else _subject_scenarios(scenarios, seed.subject)
    )
    if seed.dimension is BehavioralDimension.SUCCESS_PATH and tool_name is not None:
        return _success_outcome(scenarios, tools, tool_name)
    if seed.dimension is BehavioralDimension.FAILURE_HANDLING and tool_name is not None:
        return _failure_outcome(scenarios, tools, tool_name)
    if seed.dimension is BehavioralDimension.TIMEOUT_HANDLING and tool_name is not None:
        return _timeout_outcome(scenarios, tools, tool_name)
    if (
        seed.dimension is BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE
        and tool_name is not None
    ):
        return _fabrication_outcome(scenarios, tools, tool_name)
    if seed.dimension is BehavioralDimension.AMBIGUOUS_OUTCOME:
        return _ambiguous_outcome(scoped, tools, tool_name)
    if seed.dimension is BehavioralDimension.RETRY_CONTROL:
        return _retry_outcome(scoped, tools, tool_name)
    if seed.dimension is BehavioralDimension.DUPLICATE_ACTION:
        return _duplicate_outcome(scoped, tools, tool_name)
    if seed.dimension is BehavioralDimension.ORDERING:
        return _ordering_outcome(scoped, tools, tool_name)
    if (
        seed.dimension is BehavioralDimension.CONFIRMATION_WITHOUT_CONSENT
        and tool_name is not None
    ):
        return _confirmation_outcome(scenarios, tools, tool_name, with_consent=False)
    if (
        seed.dimension is BehavioralDimension.CONFIRMATION_WITH_CONSENT
        and tool_name is not None
    ):
        return _confirmation_outcome(scenarios, tools, tool_name, with_consent=True)
    if (
        seed.dimension is BehavioralDimension.PREREQUISITE_SUCCESS
        and tool_name is not None
        and seed.prerequisite_tool_name is not None
    ):
        return _prerequisite_outcome(
            scenarios,
            tools,
            tool_name,
            seed.prerequisite_tool_name,
            failure=False,
        )
    if (
        seed.dimension is BehavioralDimension.PREREQUISITE_FAILURE
        and tool_name is not None
        and seed.prerequisite_tool_name is not None
    ):
        return _prerequisite_outcome(
            scenarios,
            tools,
            tool_name,
            seed.prerequisite_tool_name,
            failure=True,
        )
    return _Outcome(
        BehavioralCoverageStatus.UNSUPPORTED,
        "behavioral_dimension_not_supported",
        (),
    )


def _bounded_requirement(seed: _Seed, outcome: _Outcome) -> BehavioralCoverageRequirement:
    evidence = tuple(sorted(set(outcome.evidence)))
    visible = evidence[:MAX_COVERAGE_DETAILS]
    return BehavioralCoverageRequirement(
        subject=seed.subject,
        status=outcome.status,
        reason_code=outcome.reason_code,
        evidence=visible,
        omitted_evidence=max(0, len(evidence) - len(visible)),
    )


def _family(
    dimension: BehavioralDimension,
    seeds: Sequence[_Seed],
    scenarios: Sequence[Scenario],
    tools: Mapping[str, ToolDefinition],
) -> BehavioralCoverageFamily:
    requirements = [
        _bounded_requirement(seed, _evaluate_seed(seed, scenarios, tools))
        for seed in sorted(seeds, key=lambda item: item.subject)
    ]
    counts = Counter(requirement.status for requirement in requirements)
    # Artifact redaction can collapse distinct secret-shaped identifiers to the
    # same safe subject. Keep every requirement in the counts, but omit all
    # colliding detail rows instead of inventing an identity or retaining a
    # secret-derived hash.
    subject_counts = Counter(requirement.subject for requirement in requirements)
    distinct_details = [
        requirement
        for requirement in requirements
        if subject_counts[requirement.subject] == 1
    ]
    visible = tuple(distinct_details[:MAX_COVERAGE_DETAILS])
    return BehavioralCoverageFamily(
        dimension=dimension,
        covered=counts[BehavioralCoverageStatus.COVERED],
        partial=counts[BehavioralCoverageStatus.PARTIAL],
        missing=counts[BehavioralCoverageStatus.MISSING],
        unknown=counts[BehavioralCoverageStatus.UNKNOWN],
        unsupported=counts[BehavioralCoverageStatus.UNSUPPORTED],
        requirements=visible,
        omitted=len(requirements) - len(visible),
    )


def _validate_reference_contains(
    actual: Sequence[Scenario], reference: Sequence[Scenario]
) -> None:
    actual_identities = Counter(
        (scenario.scenario_id, scenario.fingerprint) for scenario in actual
    )
    reference_identities = Counter(
        (scenario.scenario_id, scenario.fingerprint) for scenario in reference
    )
    if any(
        count > reference_identities[identity]
        for identity, count in actual_identities.items()
    ):
        raise ValueError("actual scenarios must be contained in reference_scenarios")


def analyze_behavioral_coverage(
    spec: AgentSpec,
    scenarios: Sequence[Scenario],
    *,
    suite_fingerprint: str | None = None,
    reference_scenarios: Sequence[Scenario] | None = None,
    reference_scope: BehavioralCoverageReferenceScope = (
        BehavioralCoverageReferenceScope.COMPLETE
    ),
) -> BehavioralCoverage:
    """Derive explicit behavioral requirements and selected-suite evidence.

    ``reference_scenarios`` supplies the full requirement universe when
    ``scenarios`` is a selected subset.  Evidence and source binding always use
    only ``scenarios``, so a case excluded by selection becomes missing instead
    of silently shrinking the denominator.
    """

    actual = tuple(scenarios)
    reference = actual if reference_scenarios is None else tuple(reference_scenarios)
    if reference_scenarios is not None:
        _validate_reference_contains(actual, reference)
    declared_tools = [item.value for item in spec.tools.items]
    # A repeated declared name makes tool identity ambiguous, so no definition
    # under that name is authoritative. ToolGateway already refuses to execute
    # such a target; derived coverage is report metadata and must not abort
    # generation, so the affected subjects stay UNKNOWN instead of silently
    # binding to one arbitrary definition.
    ambiguous_tool_names = {
        name
        for name, count in Counter(tool.name for tool in declared_tools).items()
        if count > 1
    }
    tools = {
        tool.name: tool
        for tool in declared_tools
        if tool.name not in ambiguous_tool_names
    }
    seeds: dict[tuple[BehavioralDimension, str], _Seed] = {}
    _seed_from_spec(spec, seeds)
    non_authoritative_risk_tools = {
        item.value.name for item in spec.tools.items if not item.authoritative
    }
    _seed_from_scenarios(reference, seeds, non_authoritative_risk_tools)
    lossy_tool_names = {
        tool.name for tool in declared_tools if not _tool_binding_is_lossless(tool)
    }
    for key, seed in tuple(seeds.items()):
        bound_names = tuple(
            name
            for name in (seed.tool_name, seed.prerequisite_tool_name)
            if name is not None
        )
        if any(name in ambiguous_tool_names for name in bound_names):
            seeds[key] = replace(
                seed,
                applicability=_Applicability.UNKNOWN,
                unknown_reason="declared_tool_name_is_ambiguous",
            )
        elif any(
            name in lossy_tool_names or redact_log_text(name) != name
            for name in bound_names
        ):
            seeds[key] = replace(
                seed,
                applicability=_Applicability.UNKNOWN,
                unknown_reason="coverage_binding_redacted_or_truncated",
            )


    by_dimension: dict[BehavioralDimension, list[_Seed]] = {
        dimension: [] for dimension in BehavioralDimension
    }
    for seed in seeds.values():
        by_dimension[seed.dimension].append(seed)
    families = tuple(
        _family(dimension, by_dimension[dimension], actual, tools)
        for dimension in BehavioralDimension
    )
    return BehavioralCoverage(
        spec_id=spec.spec_id,
        spec_digest=_spec_digest(spec),
        scenario_count=len(actual),
        scenario_digest=_scenario_digest(actual),
        reference_scenario_count=len(reference),
        reference_scenario_digest=_scenario_digest(reference),
        reference_scope=reference_scope,
        suite_fingerprint=suite_fingerprint,
        families=families,
    )


class _UnspecifiedSuiteFingerprint:
    pass


_UNSPECIFIED_SUITE_FINGERPRINT = _UnspecifiedSuiteFingerprint()


class _UnspecifiedReferenceScenarios:
    pass


_UNSPECIFIED_REFERENCE_SCENARIOS = _UnspecifiedReferenceScenarios()


@overload
def verify_behavioral_coverage_binding(
    coverage: BehavioralCoverage,
    spec: AgentSpec,
    scenarios: Sequence[Scenario],
) -> None: ...


@overload
def verify_behavioral_coverage_binding(
    coverage: BehavioralCoverage,
    spec: AgentSpec,
    scenarios: Sequence[Scenario],
    *,
    suite_fingerprint: str | None,
) -> None: ...


@overload
def verify_behavioral_coverage_binding(
    coverage: BehavioralCoverage,
    spec: AgentSpec,
    scenarios: Sequence[Scenario],
    *,
    reference_scenarios: Sequence[Scenario],
) -> None: ...


@overload
def verify_behavioral_coverage_binding(
    coverage: BehavioralCoverage,
    spec: AgentSpec,
    scenarios: Sequence[Scenario],
    *,
    suite_fingerprint: str | None,
    reference_scenarios: Sequence[Scenario],
) -> None: ...


def verify_behavioral_coverage_binding(
    coverage: BehavioralCoverage,
    spec: AgentSpec,
    scenarios: Sequence[Scenario],
    *,
    suite_fingerprint: str | None | _UnspecifiedSuiteFingerprint = (
        _UNSPECIFIED_SUITE_FINGERPRINT
    ),
    reference_scenarios: (
        Sequence[Scenario] | _UnspecifiedReferenceScenarios
    ) = _UNSPECIFIED_REFERENCE_SCENARIOS,
) -> None:
    """Fail closed when derived coverage is presented for different sources."""

    actual = tuple(scenarios)
    complete_reference: tuple[Scenario, ...] | None = (
        actual
        if coverage.reference_scenario_count == coverage.scenario_count
        else None
    )
    mismatches: list[str] = []
    if coverage.spec_id != redact_log_text(spec.spec_id):
        mismatches.append("spec_id")
    if coverage.spec_digest != _spec_digest(spec):
        mismatches.append("spec_digest")
    if coverage.scenario_count != len(actual):
        mismatches.append("scenario_count")
    if coverage.scenario_digest != _scenario_digest(actual):
        mismatches.append("scenario_digest")
    if coverage.reference_scenario_count < coverage.scenario_count:
        mismatches.append("reference_scenario_count")
    if (
        coverage.reference_scenario_count == coverage.scenario_count
        and coverage.reference_scenario_digest != coverage.scenario_digest
    ):
        mismatches.append("reference_scenario_digest")
    if (
        coverage.reference_scope
        is BehavioralCoverageReferenceScope.AVAILABLE_SCENARIOS_ONLY
        and coverage.reference_scenario_count != coverage.scenario_count
    ):
        mismatches.append("reference_scope")
    if not isinstance(
        reference_scenarios, _UnspecifiedReferenceScenarios
    ):
        reference = tuple(reference_scenarios)
        complete_reference = reference
        try:
            _validate_reference_contains(actual, reference)
        except ValueError:
            mismatches.append("reference_scenarios")
        if coverage.reference_scenario_count != len(reference):
            mismatches.append("reference_scenario_count")
        if coverage.reference_scenario_digest != _scenario_digest(reference):
            mismatches.append("reference_scenario_digest")

    try:
        expected_fingerprint = coverage.expected_fingerprint()
    except ValueError:
        mismatches.append("fingerprint")
    else:
        if not hmac.compare_digest(coverage.fingerprint, expected_fingerprint):
            mismatches.append("fingerprint")
    expected_suite_fingerprint = suite_fingerprint
    if isinstance(expected_suite_fingerprint, str):
        expected_suite_fingerprint = redact_log_text(expected_suite_fingerprint)
    if not isinstance(
        suite_fingerprint, _UnspecifiedSuiteFingerprint
    ) and coverage.suite_fingerprint != expected_suite_fingerprint:
        mismatches.append("suite_fingerprint")
    if complete_reference is not None:
        try:
            rederived = analyze_behavioral_coverage(
                spec,
                actual,
                reference_scenarios=complete_reference,
                suite_fingerprint=coverage.suite_fingerprint,
                reference_scope=coverage.reference_scope,
            )
        except ValueError:
            mismatches.append("derived_coverage")
        else:
            if coverage != rederived:
                mismatches.append("derived_coverage")

    if mismatches:
        raise ValueError(
            "behavioral coverage source binding mismatch: "
            + ", ".join(dict.fromkeys(mismatches))
        )


__all__ = ["analyze_behavioral_coverage", "verify_behavioral_coverage_binding"]
