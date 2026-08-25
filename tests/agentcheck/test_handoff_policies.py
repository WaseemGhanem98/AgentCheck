"""Handoff expectations the evaluator already enforces, but nothing could declare.

`agentcheck/evaluate/engine.py` implements five deterministic handoff checks over
adapter-recorded HANDOFF events, and the generation lint has always listed them
as supported kinds. Neither fact helped anyone: generation emits no handoff
constraint, and the policy layer had no rule kind that produced one. The only
route to a handoff assertion was hand-writing raw suite JSON.

That is the same shape as the `ordering` gap fixed in policies v1 -- an
evaluator capability with no path to it -- and it lands on a pattern that is
not rare. Inspecting the official `customer_service` example reports a
three-agent handoff topology, then generates fifteen cases that assert nothing
about handoffs and a coverage report that does not mention them.

These tests pin that a declared handoff rule reaches the evaluator with the
parameters it needs, and try to break the wiring: by redirecting a rule at an
agent the pack never declared, by dropping the ceiling a bound needs, and by
collapsing two rules that assert different things.
"""

from __future__ import annotations

from typing import Any

import pytest
from agents import Agent, function_tool
from pydantic import ValidationError

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.config import AgentCheckConfig
from agentcheck.domain import TrajectoryConstraintKind
from agentcheck.generate.suite import build_frozen_suite
from agentcheck.policies import PolicyPack, PolicyRule, PolicyRuleKind

SEED = 1729


@function_tool
def update_seat(confirmation_number: str, new_seat: str) -> str:
    """Update the seat on a booking."""
    raise AssertionError("original handler must never run")


@function_tool
def faq_lookup(question: str) -> str:
    """Look up an answer."""
    raise AssertionError("original handler must never run")


def _spec(*tools: Any):
    return OpenAIAgentsAdapter().inspect(
        Agent(name="T", instructions="Assist.", tools=list(tools), model="gpt-4.1-mini")
    )


def _pack(*rules: PolicyRule) -> PolicyPack:
    return PolicyPack(
        pack_id="airline_policy_v1",
        version="1",
        title="Airline policy",
        description="Behavioural contracts for a handoff-based airline agent.",
        rules=rules,
    )


def _rule(kind: PolicyRuleKind, **kw: Any) -> PolicyRule:
    return PolicyRule(
        rule_id=kw.pop("rule_id", f"{kind.value}__rule"),
        kind=kind,
        description=kw.pop("description", "declared behavioural contract"),
        **kw,
    )


def _constraints(pack: PolicyPack):
    spec = _spec(update_seat, faq_lookup)
    suite = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED, policy_packs=(pack,))
    return [
        constraint
        for case in suite.cases
        for constraint in case.scenario.trajectory_constraints
    ]


def _of_kind(pack: PolicyPack, kind: TrajectoryConstraintKind):
    return [c for c in _constraints(pack) if c.kind is kind]


# --- each declared handoff rule reaches the evaluator ----------------------


def test_a_forbidden_handoff_rule_reaches_the_evaluator() -> None:
    pack = _pack(
        _rule(
            PolicyRuleKind.FORBIDDEN_HANDOFF,
            parameters={"to_agent": "Billing Agent"},
        )
    )

    attached = _of_kind(pack, TrajectoryConstraintKind.FORBIDDEN_HANDOFF)

    assert attached, "no forbidden-handoff constraint was attached"
    for constraint in attached:
        assert constraint.parameters["to_agent"] == "Billing Agent"


def test_a_required_handoff_rule_carries_its_target_and_minimum() -> None:
    pack = _pack(
        _rule(
            PolicyRuleKind.REQUIRED_HANDOFF,
            parameters={"to_agent": "Seat Booking Agent", "minimum": 1},
        )
    )

    attached = _of_kind(pack, TrajectoryConstraintKind.REQUIRED_HANDOFF)

    assert attached, "no required-handoff constraint was attached"
    for constraint in attached:
        assert constraint.parameters["to_agent"] == "Seat Booking Agent"
        assert constraint.parameters["minimum"] == 1


def test_a_handoff_ceiling_reaches_the_evaluator() -> None:
    pack = _pack(_rule(PolicyRuleKind.MAX_HANDOFFS, parameters={"maximum": 2}))

    attached = _of_kind(pack, TrajectoryConstraintKind.MAX_HANDOFFS)

    assert attached, "no handoff ceiling was attached"
    for constraint in attached:
        assert constraint.parameters["maximum"] == 2


def test_a_loop_bound_reaches_the_evaluator() -> None:
    """Two agents bouncing a customer back and forth is the failure this names."""
    pack = _pack(
        _rule(PolicyRuleKind.NO_HANDOFF_LOOP, parameters={"max_edge_repeats": 1})
    )

    attached = _of_kind(pack, TrajectoryConstraintKind.NO_HANDOFF_LOOP)

    assert attached, "no handoff-loop bound was attached"
    for constraint in attached:
        assert constraint.parameters["max_edge_repeats"] == 1


def test_handoff_before_tool_binds_to_the_declared_tool() -> None:
    """"Escalate before you touch the booking" is a real airline contract."""
    pack = _pack(
        _rule(
            PolicyRuleKind.HANDOFF_BEFORE_TOOL,
            tool_name="update_seat",
            parameters={"to_agent": "Seat Booking Agent"},
        )
    )

    attached = _of_kind(pack, TrajectoryConstraintKind.HANDOFF_BEFORE_TOOL)

    assert attached, "no handoff-before-tool constraint was attached"
    for constraint in attached:
        assert constraint.parameters["tool_name"] == "update_seat"
        assert constraint.parameters["to_agent"] == "Seat Booking Agent"


# --- the wiring must not be fooled -----------------------------------------


def test_a_parameter_block_cannot_redirect_the_rule_at_another_tool() -> None:
    """tool_name is the pack's claim, not the parameter block's."""
    pack = _pack(
        _rule(
            PolicyRuleKind.HANDOFF_BEFORE_TOOL,
            tool_name="update_seat",
            parameters={"tool_name": "faq_lookup", "to_agent": "X"},
        )
    )

    for constraint in _of_kind(pack, TrajectoryConstraintKind.HANDOFF_BEFORE_TOOL):
        assert constraint.parameters["tool_name"] == "update_seat"


def test_two_handoff_ceilings_that_differ_are_not_collapsed() -> None:
    """Dedup keyed only on kind would silently drop one of these."""
    pack = _pack(
        _rule(PolicyRuleKind.MAX_HANDOFFS, rule_id="a", parameters={"maximum": 1}),
        _rule(PolicyRuleKind.MAX_HANDOFFS, rule_id="b", parameters={"maximum": 5}),
    )

    maxima = {
        c.parameters["maximum"]
        for c in _of_kind(pack, TrajectoryConstraintKind.MAX_HANDOFFS)
    }

    assert maxima == {1, 5}


def test_a_handoff_ceiling_without_a_maximum_is_rejected() -> None:
    """A bound with no number would be INCONCLUSIVE forever and read as enforced."""
    with pytest.raises(ValidationError):
        _rule(PolicyRuleKind.MAX_HANDOFFS, parameters={})


def test_handoff_before_tool_still_requires_a_tool_name() -> None:
    """It asserts a handoff happened before a specific call. Without one it says nothing."""
    with pytest.raises(ValidationError):
        _rule(PolicyRuleKind.HANDOFF_BEFORE_TOOL, parameters={"to_agent": "X"})


def test_an_empty_agent_name_is_rejected_rather_than_matching_everything() -> None:
    """`""` would fall through as "no filter" and quietly widen the rule."""
    with pytest.raises(ValidationError):
        _rule(PolicyRuleKind.FORBIDDEN_HANDOFF, parameters={"to_agent": ""})


def test_a_negative_handoff_ceiling_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _rule(PolicyRuleKind.MAX_HANDOFFS, parameters={"maximum": -1})


def test_run_scoped_handoff_rules_need_no_tool_name() -> None:
    """A handoff is a property of the run, not of one call."""
    for kind in (
        PolicyRuleKind.REQUIRED_HANDOFF,
        PolicyRuleKind.FORBIDDEN_HANDOFF,
        PolicyRuleKind.NO_HANDOFF_LOOP,
    ):
        _rule(kind, parameters={})
    _rule(PolicyRuleKind.MAX_HANDOFFS, parameters={"maximum": 1})
