"""Deterministic coverage-based scenario selection.

Selection is greedy set cover over explicit coverage dimensions. It does not
sample, does not call a model, and does not treat inferred action kinds as
authoritative business coverage. The generation seed participates only by
determining the candidate pool; it is not used as a random source here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from pydantic import Field, model_validator

from agentcheck.domain import (
    ActionKind,
    AgentSpec,
    ContractModel,
    Scenario,
    TrajectoryConstraintKind,
)
from agentcheck.errors import ConfigurationError


SELECTION_ALGORITHM: Literal["greedy_set_cover.v1"] = "greedy_set_cover.v1"
MAX_CASES = 256
MAX_SELECTION_CANDIDATES = 512
MAX_DIMENSIONS = 512
MAX_DECISIONS = 512

_MANDATORY_TRAJECTORY_KINDS = {
    TrajectoryConstraintKind.CONFIRMATION_BEFORE_TOOL,
    TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT,
    TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT,
    TrajectoryConstraintKind.NO_SAME_STAGE_DUPLICATE_ACTION,
}


class CoverageReport(ContractModel):
    """Covered, uncovered, unsupported, and unknown dimensions after selection."""

    covered: tuple[str, ...] = ()
    uncovered: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()

    @model_validator(mode="after")
    def bound_and_sort(self) -> "CoverageReport":
        object.__setattr__(self, "covered", _bound_tags(self.covered))
        object.__setattr__(self, "uncovered", _bound_tags(self.uncovered))
        object.__setattr__(self, "unsupported", _bound_tags(self.unsupported))
        object.__setattr__(self, "unknown", _bound_tags(self.unknown))
        overlap = set(self.covered).intersection(self.uncovered)
        if overlap:
            raise ValueError(
                "coverage report cannot list a dimension as both covered and uncovered"
            )
        return self


class CoverageModel(ContractModel):
    """Dimension universe used for selection.

    Tags already stored on ``Scenario.dimension_tags`` are the primary
    dimensions. Tool names, boundary/mutation families, and declared policy
    oracles are added when present. Inferred capability action kinds are never
    coverage goals.
    """

    dimensions: tuple[str, ...] = ()
    extra_goals: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()

    @model_validator(mode="after")
    def bound_and_sort(self) -> "CoverageModel":
        object.__setattr__(self, "dimensions", _bound_tags(self.dimensions))
        object.__setattr__(self, "extra_goals", _bound_tags(self.extra_goals))
        object.__setattr__(self, "unsupported", _bound_tags(self.unsupported))
        object.__setattr__(self, "unknown", _bound_tags(self.unknown))
        return self


class SelectionDecision(ContractModel):
    scenario_id: str = Field(min_length=1, max_length=200)
    selected: bool
    reason: str = Field(min_length=1, max_length=100)
    gained_dimensions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def normalize(self) -> "SelectionDecision":
        object.__setattr__(
            self, "gained_dimensions", _bound_tags(self.gained_dimensions)
        )
        return self


class SelectionPlan(ContractModel):
    """Chosen and excluded scenario IDs with reasons and the resulting coverage."""

    algorithm: Literal["greedy_set_cover.v1"] = SELECTION_ALGORITHM
    max_cases: int | None = Field(default=None, ge=1, le=MAX_CASES)
    selected_ids: tuple[str, ...] = Field(min_length=1)
    excluded_ids: tuple[str, ...] = ()
    decisions: tuple[SelectionDecision, ...] = Field(min_length=1, max_length=MAX_DECISIONS)
    coverage: CoverageReport = Field(default_factory=CoverageReport)

    @model_validator(mode="after")
    def validate_plan_consistency(self) -> "SelectionPlan":
        selected = list(self.selected_ids)
        excluded = list(self.excluded_ids)
        if len(selected) != len(set(selected)):
            raise ValueError("selection plan selected IDs must be unique")
        if len(excluded) != len(set(excluded)):
            raise ValueError("selection plan excluded IDs must be unique")
        overlap = set(selected).intersection(excluded)
        if overlap:
            raise ValueError("a scenario cannot be both selected and excluded")
        decision_ids = [item.scenario_id for item in self.decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("selection plan decisions must be unique")
        if set(decision_ids) != set(selected).union(excluded):
            raise ValueError("selection plan decisions must cover every candidate")
        decided_selected = {
            item.scenario_id for item in self.decisions if item.selected
        }
        if decided_selected != set(selected):
            raise ValueError("selection plan selected IDs must match selected decisions")
        if self.max_cases is not None and len(selected) > self.max_cases:
            raise ValueError("selection plan exceeds max_cases")
        return self


@dataclass(frozen=True, slots=True)
class CoverageUnit:
    """One candidate for greedy selection. Not a persisted contract."""

    scenario_id: str
    fingerprint: str
    dimensions: tuple[str, ...]
    mandatory: bool = False


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_ids: tuple[str, ...]
    plan: SelectionPlan


def validate_max_cases(value: int) -> int:
    if value < 1 or value > MAX_CASES:
        raise ConfigurationError(
            f"max-cases must be between 1 and {MAX_CASES}"
        )
    return value


def scenario_is_mandatory(scenario: Scenario) -> bool:
    """True when a declared policy pack attached a tool-specific hard rule.

    Globally applied output rules such as no-fabricated-success do not mark
    every case mandatory; dropping one of those would still leave the pack
    coverable by another case.
    """

    policy_oracles = {
        oracle.oracle_id
        for oracle in scenario.oracle_provenance
        if oracle.oracle_id.startswith("policy:")
        and oracle.supports_hard_failure
        and oracle.confidence >= 0.8
    }
    if not policy_oracles:
        return False
    for constraint in scenario.trajectory_constraints:
        if constraint.kind not in _MANDATORY_TRAJECTORY_KINDS:
            continue
        if any(oracle_id in policy_oracles for oracle_id in constraint.oracle_ids):
            return True
    return False


def scenario_coverage_dimensions(
    scenario: Scenario, *, extra_tags: Sequence[str] = ()
) -> tuple[str, ...]:
    tags: list[str] = list(scenario.dimension_tags)
    tags.extend(extra_tags)
    for fixture in scenario.tool_fixtures:
        tags.append(f"tool:{fixture.tool_name}")
    for constraint in (
        *scenario.required_tool_behavior,
        *scenario.allowed_tool_behavior,
        *scenario.forbidden_tool_behavior,
    ):
        tags.append(f"tool:{constraint.tool_name}")
    for oracle in scenario.oracle_provenance:
        if oracle.oracle_id.startswith("policy:"):
            parts = oracle.oracle_id.split(":")
            if len(parts) >= 2 and parts[1]:
                tags.append(f"policy:{parts[1]}")
    return _bound_tags(tags)


def lineage_coverage_tags(
    *,
    origin: str | None = None,
    tool_name: str | None = None,
    boundary_kind: str | None = None,
    mutation_kind: str | None = None,
) -> tuple[str, ...]:
    tags: list[str] = []
    if origin:
        tags.append(f"origin:{origin}")
    if tool_name:
        tags.append(f"tool:{tool_name}")
    if boundary_kind:
        tags.append(f"schema:{boundary_kind}")
    if mutation_kind:
        tags.append(f"mutation:{mutation_kind}")
    return tuple(tags)


def spec_selection_context(
    spec: AgentSpec | None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return ``(extra_goals, unsupported, unknown)`` for a spec.

    Extra goals are declared tools and policy packs. Unsupported names schema
    features this generator will not invert. Unknown records inspection
    unknowns and capabilities whose action kind stayed ``other``. Lexical
    action inference is never a coverage goal.
    """

    if spec is None:
        return (), (), ()
    extra: list[str] = [f"tool:{tool.value.name}" for tool in spec.tools.items]
    extra.extend(f"policy:{policy.value.policy_id}" for policy in spec.policies.items)
    unsupported: list[str] = []
    from .boundaries import unsupported_boundary_reasons

    for tool in spec.tools.items:
        unsupported.extend(
            f"unsupported:{reason}" for reason in unsupported_boundary_reasons(tool.value)
        )
    unknown: list[str] = [f"unknown:{item.path}" for item in spec.unknowns]
    for capability_item in spec.capabilities.items:
        capability = capability_item.value
        if capability.action_kind is ActionKind.OTHER:
            unknown.append(f"unknown:capability:{capability.name}")
    return (
        _bound_tags(extra),
        _bound_tags(unsupported),
        _bound_tags(unknown),
    )


def coverage_unit_for_scenario(
    scenario: Scenario, *, extra_tags: Sequence[str] = ()
) -> CoverageUnit:
    return CoverageUnit(
        scenario_id=scenario.scenario_id,
        fingerprint=scenario.fingerprint,
        dimensions=scenario_coverage_dimensions(scenario, extra_tags=extra_tags),
        mandatory=scenario_is_mandatory(scenario),
    )


def select_units(
    units: Sequence[CoverageUnit],
    *,
    max_cases: int | None,
    extra_goals: Sequence[str] = (),
    unsupported: Sequence[str] = (),
    unknown: Sequence[str] = (),
) -> SelectionResult:
    """Greedy set cover with deterministic tie-breaking.

    ``max_cases`` is a cap, not a target. When it is ``None``, redundant cases
    are dropped once every coverable dimension is represented. When it is at
    least the pool size, every candidate is kept.
    """

    if max_cases is not None:
        validate_max_cases(max_cases)
    if not units:
        raise ConfigurationError("no candidates remain for coverage selection")
    if len(units) > MAX_SELECTION_CANDIDATES:
        raise ConfigurationError(
            f"candidate pool exceeds the {MAX_SELECTION_CANDIDATES} case selection bound"
        )
    identifiers = [unit.scenario_id for unit in units]
    if len(identifiers) != len(set(identifiers)):
        raise ConfigurationError("coverage selection requires unique scenario IDs")

    extra = _bound_tags(extra_goals)
    unsupported_tags = _bound_tags(unsupported)
    unknown_tags = _bound_tags(unknown)
    pool_dimensions = set(_bound_tags(tag for unit in units for tag in unit.dimensions))

    if max_cases is not None and max_cases >= len(units):
        selected = list(units)
        excluded: list[CoverageUnit] = []
        gained_by_id = _gained_in_order(selected)
        decisions = tuple(
            SelectionDecision(
                scenario_id=unit.scenario_id,
                selected=True,
                reason="mandatory_policy" if unit.mandatory else "pool_within_budget",
                gained_dimensions=gained_by_id.get(unit.scenario_id, ()),
            )
            for unit in units
        )
    else:
        cap = max_cases if max_cases is not None else len(units)
        mandatory = [unit for unit in units if unit.mandatory]
        if len(mandatory) > cap:
            raise ConfigurationError(
                f"{len(mandatory)} mandatory policy cases exceed max-cases {cap}"
            )
        selected = list(mandatory)
        selected_ids = {unit.scenario_id for unit in selected}
        covered = _dimension_set(selected)
        remaining = [unit for unit in units if unit.scenario_id not in selected_ids]
        while len(selected) < cap and remaining:
            uncovered = pool_dimensions.difference(covered)
            if not uncovered:
                break
            pick = _best_candidate(remaining, uncovered)
            if pick is None:
                break
            selected.append(pick)
            selected_ids.add(pick.scenario_id)
            covered.update(pick.dimensions)
            remaining = [
                unit for unit in remaining if unit.scenario_id not in selected_ids
            ]
        excluded = remaining
        final_covered = _dimension_set(selected)
        gained_by_id = _gained_in_order(selected)
        decisions = tuple(
            _decision_for(unit, selected_ids, final_covered, gained_by_id)
            for unit in units
        )

    if not selected:
        raise ConfigurationError("coverage selection produced an empty suite")

    selected_id_order = tuple(
        unit.scenario_id
        for unit in units
        if unit.scenario_id in {item.scenario_id for item in selected}
    )
    excluded_id_order = tuple(
        unit.scenario_id
        for unit in units
        if unit.scenario_id in {item.scenario_id for item in excluded}
    )
    covered_tags = _bound_tags(tag for unit in selected for tag in unit.dimensions)
    coverable_goals = set(pool_dimensions).union(extra)
    uncovered_tags = _bound_tags(
        tag
        for tag in coverable_goals
        if tag not in covered_tags
        and tag not in unsupported_tags
        and tag not in unknown_tags
    )
    plan = SelectionPlan(
        max_cases=max_cases,
        selected_ids=selected_id_order,
        excluded_ids=excluded_id_order,
        decisions=decisions,
        coverage=CoverageReport(
            covered=covered_tags,
            uncovered=uncovered_tags,
            unsupported=unsupported_tags,
            unknown=unknown_tags,
        ),
    )
    return SelectionResult(selected_ids=plan.selected_ids, plan=plan)


def select_scenarios(
    scenarios: Sequence[Scenario],
    *,
    max_cases: int | None,
    spec: AgentSpec | None = None,
    extra_tags_by_id: Mapping[str, Sequence[str]] | None = None,
) -> tuple[tuple[Scenario, ...], SelectionPlan]:
    extras = extra_tags_by_id or {}
    units = tuple(
        coverage_unit_for_scenario(
            scenario, extra_tags=extras.get(scenario.scenario_id, ())
        )
        for scenario in scenarios
    )
    extra, unsupported, unknown = spec_selection_context(spec)
    result = select_units(
        units,
        max_cases=max_cases,
        extra_goals=extra,
        unsupported=unsupported,
        unknown=unknown,
    )
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    selected = tuple(by_id[scenario_id] for scenario_id in result.selected_ids)
    return selected, result.plan


def _best_candidate(
    remaining: Sequence[CoverageUnit], uncovered: set[str]
) -> CoverageUnit | None:
    best: CoverageUnit | None = None
    best_gain = 0
    for unit in remaining:
        gain = sum(1 for tag in unit.dimensions if tag in uncovered)
        if gain > best_gain:
            best = unit
            best_gain = gain
            continue
        if best is not None and gain == best_gain and gain > 0:
            if (unit.scenario_id, unit.fingerprint) < (best.scenario_id, best.fingerprint):
                best = unit
    if best is None or best_gain == 0:
        return None
    return best


def _decision_for(
    unit: CoverageUnit,
    selected_ids: set[str],
    final_covered: set[str],
    gained_by_id: dict[str, tuple[str, ...]],
) -> SelectionDecision:
    if unit.scenario_id in selected_ids:
        return SelectionDecision(
            scenario_id=unit.scenario_id,
            selected=True,
            reason="mandatory_policy" if unit.mandatory else "covers_new_dimensions",
            gained_dimensions=gained_by_id.get(unit.scenario_id, ()),
        )
    unique = tuple(sorted(tag for tag in unit.dimensions if tag not in final_covered))
    return SelectionDecision(
        scenario_id=unit.scenario_id,
        selected=False,
        reason="budget_exhausted" if unique else "redundant_coverage",
        gained_dimensions=(),
    )


def _gained_in_order(selected: Sequence[CoverageUnit]) -> dict[str, tuple[str, ...]]:
    covered: set[str] = set()
    gained: dict[str, tuple[str, ...]] = {}
    for unit in selected:
        new = tuple(sorted(tag for tag in unit.dimensions if tag not in covered))
        gained[unit.scenario_id] = new
        covered.update(unit.dimensions)
    return gained


def _dimension_set(units: Sequence[CoverageUnit]) -> set[str]:
    return {tag for unit in units for tag in unit.dimensions}


def _bound_tags(tags: Iterable[str]) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if not tag or tag in seen:
            continue
        seen.add(tag)
        unique.append(tag)
        if len(unique) >= MAX_DIMENSIONS:
            break
    return tuple(sorted(unique))


__all__ = [
    "MAX_CASES",
    "MAX_SELECTION_CANDIDATES",
    "SELECTION_ALGORITHM",
    "CoverageModel",
    "CoverageReport",
    "CoverageUnit",
    "SelectionDecision",
    "SelectionPlan",
    "SelectionResult",
    "coverage_unit_for_scenario",
    "lineage_coverage_tags",
    "scenario_coverage_dimensions",
    "scenario_is_mandatory",
    "select_scenarios",
    "select_units",
    "spec_selection_context",
    "validate_max_cases",
]
