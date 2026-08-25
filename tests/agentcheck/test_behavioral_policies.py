"""Authoring the behavioural contracts the evaluator could already judge.

The trajectory evaluator has understood ordering relations, retry ceilings and
run budgets for some time. No declared rule could reach them: policy packs
carried only four kinds, and the one path that turned a rule into a constraint
wrote ``{"tool_name": ...}`` and discarded everything else the rule said. A rule
asking for at most two retries produced a constraint with no ceiling in it, and
the evaluator answered INCONCLUSIVE for every run -- which reads like an
enforced policy and is not one.

These tests cover the authoring surface and, more importantly, try to make a
policy look enforced when it is not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from agents import Agent, function_tool
from pydantic import ValidationError

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.config import AgentCheckConfig
from agentcheck.domain import TrajectoryConstraintKind
from agentcheck.generate.suite import build_frozen_suite
from agentcheck.policies import (
    PolicyPack,
    PolicyRule,
    PolicyRuleKind,
)


SEED = 1729


@function_tool
def refund_order(order_id: str) -> str:
    """Refund an order permanently. This cannot be undone."""
    raise AssertionError("original handler must never run")


@function_tool
def verify_customer(customer_id: str) -> str:
    """Verify a customer identity."""
    raise AssertionError("original handler must never run")


def _spec(*tools: Any):
    return OpenAIAgentsAdapter().inspect(
        Agent(name="T", instructions="Assist.", tools=list(tools), model="gpt-4.1-mini")
    )


def _pack(*rules: PolicyRule) -> PolicyPack:
    return PolicyPack(
        pack_id="shop_policy_v1",
        version="1",
        title="Shop policy",
        description="Behavioural contracts for the shop agent.",
        rules=rules,
    )


def _rule(kind: PolicyRuleKind, **kw: Any) -> PolicyRule:
    return PolicyRule(
        rule_id=kw.pop("rule_id", f"{kind.value}__refund_order"),
        kind=kind,
        description=kw.pop("description", "declared behavioural contract"),
        **kw,
    )


def _constraints(pack: PolicyPack, spec: Any = None):
    spec = spec or _spec(refund_order, verify_customer)
    suite = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED, policy_packs=(pack,))
    return [
        constraint
        for case in suite.cases
        for constraint in case.scenario.trajectory_constraints
    ]


# --- the parameters actually reach the evaluator ---------------------------


def test_an_ordering_rule_carries_the_tool_that_must_come_first() -> None:
    """Without this the evaluator has no relation to check."""

    pack = _pack(
        _rule(
            PolicyRuleKind.ORDERING,
            tool_name="refund_order",
            parameters={"required_before": "verify_customer"},
        )
    )

    ordering = [
        c for c in _constraints(pack) if c.kind is TrajectoryConstraintKind.ORDERING
    ]

    assert ordering, "no ordering constraint was attached"
    for constraint in ordering:
        assert constraint.parameters["tool_name"] == "refund_order"
        assert constraint.parameters["required_before"] == "verify_customer"


def test_a_retry_ceiling_reaches_the_constraint() -> None:
    pack = _pack(
        _rule(
            PolicyRuleKind.MAX_RETRIES,
            tool_name="refund_order",
            parameters={"max_retries": 2},
        )
    )

    limits = [
        c for c in _constraints(pack) if c.kind is TrajectoryConstraintKind.MAX_RETRIES
    ]

    assert limits
    for constraint in limits:
        assert constraint.parameters["max_retries"] == 2


@pytest.mark.parametrize(
    "kind,expected",
    [
        (PolicyRuleKind.MAX_TOOL_CALLS, TrajectoryConstraintKind.MAX_TOOL_CALLS),
        (PolicyRuleKind.MAX_MODEL_TURNS, TrajectoryConstraintKind.MAX_MODEL_TURNS),
    ],
)
def test_a_run_budget_needs_no_tool_and_keeps_its_ceiling(kind, expected) -> None:
    """Budgets belong to the run, so demanding a tool_name would be wrong."""

    pack = _pack(_rule(kind, rule_id=f"{kind.value}__run", parameters={"maximum": 4}))

    budgets = [c for c in _constraints(pack) if c.kind is expected]

    assert budgets
    for constraint in budgets:
        assert constraint.parameters["maximum"] == 4


def test_two_orderings_on_one_tool_stay_separate_constraints() -> None:
    """"verify before refund" and "authorize before refund" are not one rule."""

    @function_tool
    def authorize_refund(order_id: str) -> str:
        """Authorize a refund."""
        raise AssertionError

    pack = _pack(
        _rule(
            PolicyRuleKind.ORDERING,
            rule_id="ordering__verify",
            tool_name="refund_order",
            parameters={"required_before": "verify_customer"},
        ),
        _rule(
            PolicyRuleKind.ORDERING,
            rule_id="ordering__authorize",
            tool_name="refund_order",
            parameters={"required_before": "authorize_refund"},
        ),
    )

    ordering = [
        c
        for c in _constraints(pack, _spec(refund_order, verify_customer, authorize_refund))
        if c.kind is TrajectoryConstraintKind.ORDERING
    ]
    required = {c.parameters.get("required_before") for c in ordering}

    assert {"verify_customer", "authorize_refund"} <= required


# --- invalid policies fail closed ------------------------------------------


@pytest.mark.parametrize(
    "kind,parameters",
    [
        (PolicyRuleKind.ORDERING, {}),
        (PolicyRuleKind.MAX_RETRIES, {}),
        (PolicyRuleKind.MAX_TOOL_CALLS, {}),
        (PolicyRuleKind.MAX_MODEL_TURNS, {}),
    ],
)
def test_a_rule_missing_what_its_evaluator_needs_is_refused(kind, parameters) -> None:
    """It must not parse into a rule that can only ever be INCONCLUSIVE."""

    with pytest.raises(ValidationError):
        _rule(kind, rule_id="r", tool_name="refund_order", parameters=parameters)


@pytest.mark.parametrize("value", [-1, "two", 2.5, True, None])
def test_a_nonsense_ceiling_is_refused(value) -> None:
    with pytest.raises(ValidationError):
        _rule(
            PolicyRuleKind.MAX_RETRIES,
            rule_id="r",
            tool_name="refund_order",
            parameters={"max_retries": value},
        )


def test_a_tool_ordered_before_itself_is_refused() -> None:
    with pytest.raises(ValidationError):
        _rule(
            PolicyRuleKind.ORDERING,
            rule_id="r",
            tool_name="refund_order",
            parameters={"required_before": "refund_order"},
        )


@pytest.mark.parametrize("value", ["", 7, None])
def test_an_unusable_ordering_target_is_refused(value) -> None:
    with pytest.raises(ValidationError):
        _rule(
            PolicyRuleKind.ORDERING,
            rule_id="r",
            tool_name="refund_order",
            parameters={"required_before": value},
        )


def test_a_rule_that_names_no_tool_is_still_refused_where_one_is_needed() -> None:
    with pytest.raises(ValidationError):
        _rule(
            PolicyRuleKind.ORDERING,
            rule_id="r",
            parameters={"required_before": "verify_customer"},
        )


def test_a_parameter_block_cannot_redirect_a_rule_to_another_tool() -> None:
    """The declared tool_name wins; parameters must not smuggle a substitute."""

    pack = _pack(
        _rule(
            PolicyRuleKind.ORDERING,
            tool_name="refund_order",
            parameters={
                "required_before": "verify_customer",
                "tool_name": "verify_customer",
            },
        )
    )

    ordering = [
        c for c in _constraints(pack) if c.kind is TrajectoryConstraintKind.ORDERING
    ]

    assert ordering
    for constraint in ordering:
        assert constraint.parameters["tool_name"] == "refund_order"


# --- identity and compatibility --------------------------------------------


def test_a_declared_policy_changes_suite_identity() -> None:
    """A suite asserting more is a different suite."""

    spec = _spec(refund_order, verify_customer)
    plain = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)
    packed = build_frozen_suite(
        spec,
        AgentCheckConfig(),
        seed=SEED,
        policy_packs=(
            _pack(
                _rule(
                    PolicyRuleKind.ORDERING,
                    tool_name="refund_order",
                    parameters={"required_before": "verify_customer"},
                )
            ),
        ),
    )

    assert packed.fingerprint != plain.fingerprint
    assert packed.provenance.policy_packs == ("shop_policy_v1",)


def test_the_same_policy_produces_the_same_suite() -> None:
    spec = _spec(refund_order, verify_customer)
    pack = _pack(
        _rule(
            PolicyRuleKind.MAX_RETRIES,
            tool_name="refund_order",
            parameters={"max_retries": 2},
        )
    )

    first = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED, policy_packs=(pack,))
    second = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED, policy_packs=(pack,))

    assert first.fingerprint == second.fingerprint


def test_changing_a_ceiling_changes_the_suite() -> None:
    """Otherwise a tightened policy would silently reuse a stale baseline."""

    spec = _spec(refund_order, verify_customer)

    def suite(limit: int):
        return build_frozen_suite(
            spec,
            AgentCheckConfig(),
            seed=SEED,
            policy_packs=(
                _pack(
                    _rule(
                        PolicyRuleKind.MAX_RETRIES,
                        tool_name="refund_order",
                        parameters={"max_retries": limit},
                    )
                ),
            ),
        )

    assert suite(1).fingerprint != suite(2).fingerprint


def test_the_existing_four_kinds_are_unchanged() -> None:
    """Policies v1 adds kinds; it must not alter the ones already in use."""

    pack = _pack(
        _rule(PolicyRuleKind.NO_DUPLICATE_SIDE_EFFECT, tool_name="refund_order")
    )

    # The positive-path generator already adds a no-duplicate constraint per
    # focal tool, so the declared rule is proven by its presence on the tool it
    # names rather than by the absence of the generator's own.
    duplicates = {
        c.parameters.get("tool_name")
        for c in _constraints(pack)
        if c.kind is TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT
    }

    assert "refund_order" in duplicates


def test_a_pack_file_round_trips_through_json(tmp_path: Path) -> None:
    pack = _pack(
        _rule(
            PolicyRuleKind.ORDERING,
            tool_name="refund_order",
            parameters={"required_before": "verify_customer"},
        )
    )
    path = tmp_path / "policy.json"
    path.write_text(pack.model_dump_json(), encoding="utf-8")

    reloaded = PolicyPack.model_validate_json(path.read_text(encoding="utf-8"))

    assert reloaded == pack


def test_an_unknown_rule_kind_is_refused() -> None:
    raw = json.loads(
        _pack(
            _rule(PolicyRuleKind.NO_DUPLICATE_SIDE_EFFECT, tool_name="refund_order")
        ).model_dump_json()
    )
    raw["rules"][0]["kind"] = "obey_the_vibes"

    with pytest.raises(ValidationError):
        PolicyPack.model_validate(raw)
