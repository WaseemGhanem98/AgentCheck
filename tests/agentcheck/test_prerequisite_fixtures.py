"""Prerequisite tool chains: reaching a focal action that another call gates.

A generated action scenario used to carry a fixture for exactly one tool. On a
target whose own instructions require a lookup before a consequential action,
that made the action structurally unreachable: the agent behaved correctly,
called the prerequisite first, and the prerequisite had no fixture, so a
contract-valid call became ``fixture_not_found`` and the case was reported as
INFRA_ERROR rather than as a verdict. Measured on the tau-bench retail target.

Which tools are prerequisites is not inferred. Risk classification is lexical
and already known to over-reach -- a real target had ``find_user_id_by_email``
classified state-changing from its name alone -- so a heuristic would have
excluded the very tool that gates the action. A developer declares them
instead, in the fixture pack that already carries their other committed test
data.

These tests prove the machinery deterministically. Nothing here contacts a
provider.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from agents import Agent, function_tool

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.domain import (
    SimulatedToolOutcome,
    SimulatedToolStatus,
    ToolDefinition,
    ToolFixture,
    ToolOutcomeStatus,
)
from agentcheck.errors import ConfigurationError
from agentcheck.fixtures import load_prerequisite_outcomes
from agentcheck.generate.boundaries import (
    build_outcome_variant_cases,
    build_positive_path_cases,
)
from agentcheck.runner import FixtureNotFoundError, ToolGateway


PACK = "agentcheck-fixtures.json"


@function_tool
def find_user(email: str) -> str:
    """Look up the user id for an email address."""
    raise AssertionError("original handler must never run")


@function_tool
def get_order(order_id: str) -> str:
    """Read the details of an order."""
    raise AssertionError("original handler must never run")


@function_tool
def cancel_order(order_id: str, reason: str) -> str:
    """Cancel a pending order. This deletes it permanently."""
    raise AssertionError("original handler must never run")


def _spec(*tools: Any):
    agent = Agent(
        name="Target",
        instructions="Assist the customer.",
        tools=list(tools),
        model="gpt-4.1-mini",
    )
    return OpenAIAgentsAdapter().inspect(agent)


def _write_pack(root: Path, body: str) -> None:
    (root / PACK).write_text(body, encoding="utf-8")


def _scenario_for(scenarios, tool_name: str, *, source: str = "source:positive_path"):
    """The action case for a tool.

    Boundary cases carry the same ``tool:`` tag, so the source tag is what
    distinguishes the positive-path case from the invalid-argument ones.
    """

    for scenario in scenarios:
        tags = scenario.dimension_tags
        if f"tool:{tool_name}" in tags and source in tags:
            return scenario
    raise AssertionError(f"no {source} scenario focused on {tool_name}")


def _fixture_tools(scenario) -> set[str]:
    return {fixture.tool_name for fixture in scenario.tool_fixtures}


def _definition(name: str, *, destructive: bool = False) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "additionalProperties": False,
        },
        state_changing=destructive,
        destructive=destructive,
        # The gateway refuses a tool it cannot prove it may stand in for, which
        # is the property that keeps the original handler unreachable.
        replaceable=True,
    )


# --- the gateway already supports many fixtures; prove it end to end ---------


def test_prerequisite_call_is_answered_before_the_focal_tool() -> None:
    """A lookup then a destructive action, both simulated, in one scenario."""

    gateway = ToolGateway(
        [_definition("find_user"), _definition("cancel_order", destructive=True)],
        [
            ToolFixture(
                fixture_id="prerequisite:find_user",
                tool_name="find_user",
                outcome=SimulatedToolOutcome(
                    status=SimulatedToolStatus.SUCCESS,
                    result={"user_id": "user-1"},
                ),
            ),
            ToolFixture(
                fixture_id="focal:cancel_order",
                tool_name="cancel_order",
                outcome=SimulatedToolOutcome(
                    status=SimulatedToolStatus.SUCCESS,
                    result={"acknowledged": True},
                ),
            ),
        ],
    )

    first = gateway.invoke("find_user", {"value": "a@example.com"})
    second = gateway.invoke("cancel_order", {"value": "order-1"})

    assert first.status is ToolOutcomeStatus.SUCCESS
    assert first.result == {"user_id": "user-1"}
    assert second.status is ToolOutcomeStatus.SUCCESS
    # The destructive path was genuinely reached, which is the whole point.
    assert [attempt.tool_name for attempt in gateway.attempts] == [
        "find_user",
        "cancel_order",
    ]
    # Simulated outcomes carry no state effects, so nothing was mutated.
    assert gateway.state_transitions == ()


def test_repeated_prerequisite_calls_are_each_answered() -> None:
    """A chain may need the same lookup more than once, or several lookups."""

    gateway = ToolGateway(
        [_definition("find_user"), _definition("get_order")],
        [
            ToolFixture(
                fixture_id="prerequisite:find_user",
                tool_name="find_user",
                outcome=SimulatedToolOutcome(
                    status=SimulatedToolStatus.SUCCESS, result={"user_id": "user-1"}
                ),
            ),
            ToolFixture(
                fixture_id="prerequisite:get_order",
                tool_name="get_order",
                outcome=SimulatedToolOutcome(
                    status=SimulatedToolStatus.SUCCESS, result={"status": "pending"}
                ),
            ),
        ],
    )

    assert gateway.invoke("find_user", {"value": "a"}).status is ToolOutcomeStatus.SUCCESS
    assert gateway.invoke("get_order", {"value": "b"}).status is ToolOutcomeStatus.SUCCESS
    assert len(gateway.attempts) == 2


def test_an_undeclared_tool_still_fails_closed() -> None:
    """Anything neither focal nor declared stays fixture_not_found."""

    gateway = ToolGateway(
        [_definition("find_user"), _definition("get_order")],
        [
            ToolFixture(
                fixture_id="prerequisite:find_user",
                tool_name="find_user",
                outcome=SimulatedToolOutcome(status=SimulatedToolStatus.SUCCESS),
            )
        ],
    )

    with pytest.raises(FixtureNotFoundError):
        gateway.invoke("get_order", {"value": "b"})


# --- generation ------------------------------------------------------------


def test_declared_prerequisite_is_attached_to_other_action_scenarios() -> None:
    spec = _spec(find_user, cancel_order)

    cases = build_positive_path_cases(
        spec, seed=1729, prerequisite_outcomes={"find_user": {"user_id": "user-1"}}
    )
    scenarios = tuple(case.scenario for case in cases)
    focal = _scenario_for(scenarios, "cancel_order")

    assert _fixture_tools(focal) == {"cancel_order", "find_user"}


def test_a_prerequisite_never_shadows_its_own_focal_scenario() -> None:
    """find_user's own action case keeps exactly its focal fixture."""

    spec = _spec(find_user, cancel_order)

    cases = build_positive_path_cases(
        spec, seed=1729, prerequisite_outcomes={"find_user": {"user_id": "user-1"}}
    )
    own = _scenario_for(tuple(case.scenario for case in cases), "find_user")

    assert _fixture_tools(own) == {"find_user"}
    assert len([f for f in own.tool_fixtures if f.tool_name == "find_user"]) == 1


def test_outcome_variant_cases_also_reach_past_the_prerequisite() -> None:
    """The failure/timeout variants are action cases too and need the chain."""

    spec = _spec(find_user, cancel_order)

    scenarios = build_outcome_variant_cases(
        spec, seed=1729, prerequisite_outcomes={"find_user": {"user_id": "user-1"}}
    )
    failure = next(
        s for s in scenarios if "outcome:tool-failure" in s.dimension_tags
        and "tool:cancel_order" in s.dimension_tags
    )

    assert _fixture_tools(failure) == {"cancel_order", "find_user"}
    # The focal tool still fails: the prerequisite must not soften the variant.
    focal_fixture = next(
        f for f in failure.tool_fixtures if f.tool_name == "cancel_order"
    )
    assert focal_fixture.outcome.status is SimulatedToolStatus.ERROR


def test_focal_oracle_stays_scoped_to_the_focal_tool() -> None:
    """A prerequisite is permitted, never asserted about."""

    spec = _spec(find_user, cancel_order)

    cases = build_positive_path_cases(
        spec, seed=1729, prerequisite_outcomes={"find_user": {"user_id": "user-1"}}
    )
    focal = _scenario_for(tuple(case.scenario for case in cases), "cancel_order")

    constrained = {c.tool_name for c in focal.allowed_tool_behavior}
    constrained |= {c.tool_name for c in focal.required_tool_behavior}
    constrained |= {c.tool_name for c in focal.forbidden_tool_behavior}
    assert constrained == {"cancel_order"}
    for constraint in focal.trajectory_constraints:
        assert constraint.parameters.get("tool_name") == "cancel_order"


def test_budget_grows_by_exactly_the_calls_it_permits() -> None:
    """Otherwise a legitimate chain would trip max_tool_calls."""

    spec = _spec(find_user, get_order, cancel_order)

    without = _scenario_for(
        tuple(c.scenario for c in build_positive_path_cases(spec, seed=1729)),
        "cancel_order",
    )
    with_two = _scenario_for(
        tuple(
            c.scenario
            for c in build_positive_path_cases(
                spec,
                seed=1729,
                prerequisite_outcomes={"find_user": None, "get_order": None},
            )
        ),
        "cancel_order",
    )

    assert with_two.resource_budgets.max_tool_calls == (
        without.resource_budgets.max_tool_calls + 2
    )
    assert with_two.resource_budgets.max_model_turns == (
        without.resource_budgets.max_model_turns + 2
    )
    # The wall clock is a safety budget and must not move.
    assert (
        with_two.resource_budgets.wall_clock_seconds
        == without.resource_budgets.wall_clock_seconds
    )


def test_no_declared_prerequisite_reproduces_the_previous_suite() -> None:
    """Backward compatibility: identical fingerprints when nothing is declared."""

    spec = _spec(find_user, cancel_order)

    before = tuple(c.scenario.fingerprint for c in build_positive_path_cases(spec, seed=1729))
    after = tuple(
        c.scenario.fingerprint
        for c in build_positive_path_cases(spec, seed=1729, prerequisite_outcomes={})
    )

    assert before == after


# --- the developer-facing pack --------------------------------------------


def test_prerequisites_load_from_the_fixture_pack(tmp_path: Path) -> None:
    _write_pack(
        tmp_path,
        """
        {
          "schema_version": "agentcheck.fixtures.v1",
          "prerequisites": {"find_user": {"result": {"user_id": "user-1"}}}
        }
        """,
    )

    outcomes = load_prerequisite_outcomes(tmp_path, _spec(find_user, cancel_order))

    assert outcomes == {"find_user": {"user_id": "user-1"}}


def test_prerequisite_naming_an_unknown_tool_is_refused(tmp_path: Path) -> None:
    _write_pack(
        tmp_path,
        """
        {
          "schema_version": "agentcheck.fixtures.v1",
          "prerequisites": {"no_such_tool": {"result": {}}}
        }
        """,
    )

    with pytest.raises(ConfigurationError, match="unknown tool"):
        load_prerequisite_outcomes(tmp_path, _spec(find_user, cancel_order))


def test_prerequisite_result_may_not_carry_a_credential(tmp_path: Path) -> None:
    _write_pack(
        tmp_path,
        """
        {
          "schema_version": "agentcheck.fixtures.v1",
          "prerequisites": {
            "find_user": {"result": {"token": "sk-thisisafakesecretvalue12"}}
          }
        }
        """,
    )

    with pytest.raises(ConfigurationError, match="credential"):
        load_prerequisite_outcomes(tmp_path, _spec(find_user, cancel_order))


def test_absent_pack_declares_no_prerequisites(tmp_path: Path) -> None:
    assert load_prerequisite_outcomes(tmp_path, _spec(find_user)) == {}


# --- end to end: generation -> frozen suite -> gateway ---------------------


def _suite_with_prerequisite(tmp_path: Path, **kwargs: Any):
    from agentcheck.config import AgentCheckConfig
    from agentcheck.generate.suite import build_frozen_suite

    spec = _spec(find_user, get_order, cancel_order)
    _write_pack(
        tmp_path,
        """
        {
          "schema_version": "agentcheck.fixtures.v1",
          "prerequisites": {
            "find_user": {"result": {"user_id": "user-1"}},
            "get_order": {"result": {"status": "pending"}}
          }
        }
        """,
    )
    outcomes = load_prerequisite_outcomes(tmp_path, spec)
    suite = build_frozen_suite(
        spec,
        AgentCheckConfig(),
        seed=1729,
        prerequisite_outcomes=outcomes,
        **kwargs,
    )
    return spec, suite


def test_frozen_suite_keeps_prerequisite_fixtures_through_a_roundtrip(
    tmp_path: Path,
) -> None:
    """A frozen suite is the replay input, so the chain must survive encoding."""

    from agentcheck.generate.suite import encode_frozen_suite, load_frozen_suite

    _, suite = _suite_with_prerequisite(tmp_path)
    destination = tmp_path / "suite.json"
    destination.write_bytes(encode_frozen_suite(suite))
    reloaded = load_frozen_suite(destination)

    assert reloaded.fingerprint == suite.fingerprint
    focal = _scenario_for(
        tuple(case.scenario for case in reloaded.cases), "cancel_order"
    )
    assert _fixture_tools(focal) == {"cancel_order", "find_user", "get_order"}


def test_the_generated_chain_actually_reaches_the_destructive_tool(
    tmp_path: Path,
) -> None:
    """The reachability claim, proved against the generated scenario itself."""

    spec, suite = _suite_with_prerequisite(tmp_path)
    focal = _scenario_for(tuple(case.scenario for case in suite.cases), "cancel_order")
    definitions = [item.value for item in spec.tools.items]

    gateway = ToolGateway(definitions, list(focal.tool_fixtures))
    lookup = gateway.invoke("find_user", {"email": "a@example.com"})
    order = gateway.invoke("get_order", {"order_id": "order-1"})
    cancelled = gateway.invoke(
        "cancel_order", {"order_id": "order-1", "reason": "no longer needed"}
    )

    assert lookup.status is ToolOutcomeStatus.SUCCESS
    assert lookup.result == {"user_id": "user-1"}
    assert order.result == {"status": "pending"}
    assert cancelled.status is ToolOutcomeStatus.SUCCESS
    # Three real calls, the last of them the destructive one, none of which
    # previously had a fixture to answer it.
    assert [attempt.tool_name for attempt in gateway.attempts] == [
        "find_user",
        "get_order",
        "cancel_order",
    ]
    # Every declared handler raises on entry, so reaching here proves none ran.
    assert gateway.state_transitions == ()
    assert focal.resource_budgets.max_tool_calls >= 3


def test_confirmation_still_binds_only_the_focal_destructive_tool(
    tmp_path: Path,
) -> None:
    """A prerequisite must not acquire the focal tool's policy obligations."""

    from agentcheck.policies.derived import derive_tool_risk_pack

    spec = _spec(find_user, get_order, cancel_order)
    pack = derive_tool_risk_pack(spec)
    assert pack is not None

    _, suite = _suite_with_prerequisite(tmp_path, policy_packs=(pack,))
    focal = _scenario_for(tuple(case.scenario for case in suite.cases), "cancel_order")

    confirmations = [
        constraint
        for constraint in focal.trajectory_constraints
        if constraint.kind.value == "confirmation_before_tool"
    ]
    assert confirmations, "the destructive focal tool must still carry confirmation"
    for constraint in confirmations:
        assert constraint.parameters["tool_name"] == "cancel_order"
    # No constraint of any kind was attached to a prerequisite.
    named = {
        constraint.parameters.get("tool_name")
        for constraint in focal.trajectory_constraints
    }
    assert "find_user" not in named and "get_order" not in named


def test_a_suite_generated_without_a_pack_is_unchanged(tmp_path: Path) -> None:
    """Existing targets keep their fingerprints, so frozen suites stay valid."""

    from agentcheck.config import AgentCheckConfig
    from agentcheck.generate.suite import build_frozen_suite

    spec = _spec(find_user, get_order, cancel_order)
    baseline = build_frozen_suite(spec, AgentCheckConfig(), seed=1729)
    explicit_empty = build_frozen_suite(
        spec, AgentCheckConfig(), seed=1729, prerequisite_outcomes={}
    )

    assert baseline.fingerprint == explicit_empty.fingerprint
