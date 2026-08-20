"""Structural complexity for shrink acceptance.

Byte size is not used. A candidate is smaller only when this lexicographic
tuple decreases. The order matches the reduction hierarchy: turns, fixtures,
faults, world state, then constraints, then nested argument nodes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field

from agentcheck.domain import ContractModel, Scenario


class ScenarioComplexity(ContractModel):
    """Counts derived from ``agentcheck.scenario.v1`` fields."""

    conversation_turns: int = Field(ge=0)
    tool_fixtures: int = Field(ge=0)
    injected_faults: int = Field(ge=0)
    world_state_entries: int = Field(ge=0)
    allowed_tool_behavior: int = Field(ge=0)
    output_criteria: int = Field(ge=0)
    trajectory_constraints: int = Field(ge=0)
    required_tool_behavior: int = Field(ge=0)
    forbidden_tool_behavior: int = Field(ge=0)
    expected_postconditions: int = Field(ge=0)
    argument_nodes: int = Field(ge=0)
    oracles: int = Field(ge=1)

    def as_tuple(self) -> tuple[int, ...]:
        return (
            self.conversation_turns,
            self.tool_fixtures,
            self.injected_faults,
            self.world_state_entries,
            self.allowed_tool_behavior,
            self.output_criteria,
            self.trajectory_constraints,
            self.required_tool_behavior,
            self.forbidden_tool_behavior,
            self.expected_postconditions,
            self.argument_nodes,
            self.oracles,
        )

    def total(self) -> int:
        return sum(self.as_tuple())


def measure_complexity(scenario: Scenario) -> ScenarioComplexity:
    argument_nodes = 0
    for fixture in scenario.tool_fixtures:
        argument_nodes += _node_count(fixture.arguments_match)
    for group in (
        scenario.required_tool_behavior,
        scenario.allowed_tool_behavior,
        scenario.forbidden_tool_behavior,
    ):
        for constraint in group:
            argument_nodes += _node_count(constraint.arguments_match)
    return ScenarioComplexity(
        conversation_turns=len(scenario.conversation_turns),
        tool_fixtures=len(scenario.tool_fixtures),
        injected_faults=len(scenario.injected_faults),
        world_state_entries=_node_count(scenario.initial_world_state),
        allowed_tool_behavior=len(scenario.allowed_tool_behavior),
        output_criteria=len(scenario.output_criteria),
        trajectory_constraints=len(scenario.trajectory_constraints),
        required_tool_behavior=len(scenario.required_tool_behavior),
        forbidden_tool_behavior=len(scenario.forbidden_tool_behavior),
        expected_postconditions=len(scenario.expected_postconditions),
        argument_nodes=argument_nodes,
        oracles=len(scenario.oracle_provenance),
    )


def is_strictly_smaller(candidate: ScenarioComplexity, current: ScenarioComplexity) -> bool:
    return candidate.as_tuple() < current.as_tuple()


def _node_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return len(value) + sum(_node_count(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) + sum(_node_count(item) for item in value)
    return 0 if value is None else 1


__all__ = [
    "ScenarioComplexity",
    "is_strictly_smaller",
    "measure_complexity",
]
