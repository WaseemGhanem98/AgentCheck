"""Deterministic scenario reductions. Deletion only; oracles are never weakened."""

from __future__ import annotations

import json
from typing import Any, Literal, Mapping

from agentcheck.domain import Scenario

from .signature import FailureSignature, protected_criterion_ids


ReductionDimension = Literal[
    "conversation_turns",
    "tool_fixtures",
    "injected_faults",
    "initial_world_state",
    "allowed_tool_behavior",
    "output_criteria",
    "trajectory_constraints",
    "required_tool_behavior",
    "forbidden_tool_behavior",
    "expected_postconditions",
]

REDUCTION_ORDER: tuple[ReductionDimension, ...] = (
    "conversation_turns",
    "tool_fixtures",
    "injected_faults",
    "initial_world_state",
    "allowed_tool_behavior",
    "output_criteria",
    "trajectory_constraints",
    "required_tool_behavior",
    "forbidden_tool_behavior",
    "expected_postconditions",
)

_CONSTRAINT_DIMENSIONS = {
    "allowed_tool_behavior",
    "output_criteria",
    "trajectory_constraints",
    "required_tool_behavior",
    "forbidden_tool_behavior",
    "expected_postconditions",
}


def dimension_items(scenario: Scenario, dimension: ReductionDimension) -> tuple[Any, ...]:
    if dimension == "initial_world_state":
        return tuple(scenario.initial_world_state.keys())
    return tuple(getattr(scenario, dimension))


def minimum_keep(dimension: ReductionDimension) -> int:
    return 1 if dimension == "conversation_turns" else 0


def removable_indices(
    scenario: Scenario,
    dimension: ReductionDimension,
    signature: FailureSignature,
) -> tuple[int, ...]:
    """Indices that may be deleted without immediately dropping a failing criterion."""

    items = dimension_items(scenario, dimension)
    if dimension not in _CONSTRAINT_DIMENSIONS:
        return tuple(range(len(items)))
    protected = protected_criterion_ids(signature)
    keep: list[int] = []
    for index, item in enumerate(items):
        criterion_id = getattr(item, "criterion_id", None)
        if criterion_id in protected:
            continue
        keep.append(index)
    return tuple(keep)


def reconstruct_scenario(
    scenario: Scenario,
    dimension: ReductionDimension,
    keep_indices: tuple[int, ...],
) -> Scenario:
    """Return a fingerprint-recomputed scenario keeping the selected items."""

    items = dimension_items(scenario, dimension)
    if any(index < 0 or index >= len(items) for index in keep_indices):
        raise ValueError("keep index is out of range")
    if tuple(sorted(set(keep_indices))) != tuple(sorted(keep_indices)):
        raise ValueError("keep indices must be unique")
    ordered = tuple(index for index in range(len(items)) if index in set(keep_indices))
    payload = json.loads(scenario.model_dump_json())
    if dimension == "initial_world_state":
        keys = items
        payload[dimension] = {keys[index]: payload[dimension][keys[index]] for index in ordered}
    else:
        payload[dimension] = [payload[dimension][index] for index in ordered]
    return _load_pruned(payload)


def _load_pruned(payload: Mapping[str, Any]) -> Scenario:
    data = dict(payload)
    data["fingerprint"] = ""
    data["oracle_provenance"] = _referenced_oracles(data)
    return Scenario.model_validate_json(json.dumps(data, ensure_ascii=False))


def _referenced_oracles(payload: Mapping[str, Any]) -> list[Any]:
    referenced: set[str] = set()
    for group in (
        "expected_postconditions",
        "required_tool_behavior",
        "allowed_tool_behavior",
        "forbidden_tool_behavior",
        "trajectory_constraints",
        "output_criteria",
    ):
        for item in payload.get(group) or ():
            referenced.update(item.get("oracle_ids") or ())
    kept = [
        oracle
        for oracle in payload.get("oracle_provenance") or ()
        if oracle.get("oracle_id") in referenced
    ]
    return kept or list(payload.get("oracle_provenance") or ())


__all__ = [
    "REDUCTION_ORDER",
    "ReductionDimension",
    "dimension_items",
    "minimum_keep",
    "reconstruct_scenario",
    "removable_indices",
]
