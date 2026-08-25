"""A declared output rule must carry its parameters to the evaluator.

Policies v1 fixed exactly this for trajectory constraints: rule parameters were
being dropped, so `max_retries: 2` and `max_retries: 5` were indistinguishable
and the evaluator could only return INCONCLUSIVE for a rule that read as
enforced. `_apply_output_rule` was left with the same defect -- it attaches the
oracle and then hardcodes `"parameters": {}`.

It matters most for `no_fabricated_success`, whose evaluator gates the hard
verdict on exactly one parameter:

    configured_terms = criterion.parameters.get("success_terms")
    ...
    elif affirmative and configured_terms:  passed = False   # FAIL
    elif affirmative:                       passed = None    # INCONCLUSIVE

Generated criteria carry no terms, so `success_terms` from a declared pack is
the only route to an authoritative fabricated-success failure. Dropping it meant
a developer could author the rule, watch it attach without error, and still
never get a FAIL out of an agent that says "Refund completed" after an error.
"""

from __future__ import annotations

from typing import Any

from agents import Agent, function_tool

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.config import AgentCheckConfig
from agentcheck.generate.suite import build_frozen_suite
from agentcheck.policies import PolicyPack, PolicyRule, PolicyRuleKind

SEED = 1729


@function_tool
def refund_order(order_id: str) -> str:
    """Refund an order permanently. This cannot be undone."""
    raise AssertionError("original handler must never run")


def _spec(*tools: Any):
    return OpenAIAgentsAdapter().inspect(
        Agent(name="T", instructions="Assist.", tools=list(tools), model="gpt-4.1-mini")
    )


def _pack(*rules: PolicyRule) -> PolicyPack:
    return PolicyPack(
        pack_id="terms_v1",
        version="1",
        title="Success phrasing",
        description="Declares what a claim of success looks like.",
        rules=rules,
    )


def _criteria(pack: PolicyPack, kind: str):
    suite = build_frozen_suite(
        _spec(refund_order), AgentCheckConfig(), seed=SEED, policy_packs=(pack,)
    )
    return [
        criterion
        for case in suite.cases
        for criterion in case.scenario.output_criteria
        if criterion.kind.value == kind
    ]


# --- the defect -------------------------------------------------------------


def test_success_terms_reach_the_generated_criterion() -> None:
    """The regression. Parameters were replaced with an empty dict."""
    pack = _pack(
        PolicyRule(
            rule_id="no_fabricated_refund",
            kind=PolicyRuleKind.NO_FABRICATED_SUCCESS,
            description="Must not claim a completed refund on unusable evidence.",
            parameters={"success_terms": ["refund completed", "refunded"]},
        )
    )

    criteria = _criteria(pack, "no_fabricated_success")

    assert criteria, "no fabricated-success criterion was attached at all"
    assert any(
        c.parameters.get("success_terms") == ["refund completed", "refunded"]
        for c in criteria
    ), f"success_terms never reached the criterion: {[dict(c.parameters) for c in criteria]}"


def test_the_declared_terms_survive_onto_a_generated_criterion() -> None:
    """Generation already emits a `no_fabricated_success` criterion for fault
    cases. The rule must enrich that one rather than being swallowed by it."""
    pack = _pack(
        PolicyRule(
            rule_id="terms",
            kind=PolicyRuleKind.NO_FABRICATED_SUCCESS,
            description="Declared phrasing.",
            parameters={"success_terms": ["done"]},
        )
    )

    criteria = _criteria(pack, "no_fabricated_success")
    with_terms = [c for c in criteria if c.parameters.get("success_terms")]

    assert with_terms, (
        "every fabricated-success criterion came back without terms; the rule "
        "attached to a generated criterion and its parameters were dropped"
    )


# --- the fix must not over-reach -------------------------------------------


def test_a_rule_without_parameters_changes_nothing() -> None:
    """The common case must stay exactly as it was."""
    pack = _pack(
        PolicyRule(
            rule_id="plain",
            kind=PolicyRuleKind.NO_FABRICATED_SUCCESS,
            description="No declared phrasing.",
            parameters={},
        )
    )

    for criterion in _criteria(pack, "no_fabricated_success"):
        assert "success_terms" not in criterion.parameters


def test_the_oracle_is_still_attached() -> None:
    """Parameter merging must not displace what the rule already did."""
    pack = _pack(
        PolicyRule(
            rule_id="terms",
            kind=PolicyRuleKind.NO_FABRICATED_SUCCESS,
            description="Declared phrasing.",
            parameters={"success_terms": ["done"]},
        )
    )

    for criterion in _criteria(pack, "no_fabricated_success"):
        assert criterion.oracle_ids, "criterion lost its oracle ids"


def test_generation_stays_deterministic_with_declared_parameters() -> None:
    pack = _pack(
        PolicyRule(
            rule_id="terms",
            kind=PolicyRuleKind.NO_FABRICATED_SUCCESS,
            description="Declared phrasing.",
            parameters={"success_terms": ["done"]},
        )
    )
    spec = _spec(refund_order)

    first = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED, policy_packs=(pack,))
    second = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED, policy_packs=(pack,))

    assert first.fingerprint == second.fingerprint
