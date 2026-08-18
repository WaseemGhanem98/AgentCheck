"""Policy scope derived from a target's own inferred tool risk.

The rule kinds were always general; only the built-in packs' scope was
hardcoded to the bundled example's tool names. These tests pin the derivation:
it is scoped to whatever tools the target declares, deterministic, conservative
about the stronger claim, and opt-in — because a policy the developer did not
ask for is a product bug, not extra coverage.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from agents import Agent, function_tool

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.config import AgentCheckConfig
from agentcheck.errors import ConfigurationError
from agentcheck.policies import DERIVED_TOOL_RISK_PACK_ID, derive_tool_risk_pack
from agentcheck.policies.loader import resolve_policy_packs
from agentcheck.policies.pack import PolicyRuleKind


def _spec(agent: Agent):
    return OpenAIAgentsAdapter().inspect(agent)


def _agent_with(*tools) -> Agent:
    return Agent(name="Target", instructions="Act.", tools=list(tools), model="gpt-4.1-mini")


@function_tool
def delete_account(account_id: str) -> str:
    """Permanently delete the account."""
    raise AssertionError("original handler must never run")


@function_tool
def update_email(account_id: str, new_email: str) -> str:
    """Update the account email address."""
    raise AssertionError("original handler must never run")


@function_tool
def lookup_account(account_id: str) -> str:
    """Look up an account."""
    raise AssertionError("original handler must never run")


def _rules(agent: Agent):
    pack = derive_tool_risk_pack(_spec(agent))
    assert pack is not None
    return {(rule.kind, rule.tool_name) for rule in pack.rules}


def test_rules_are_scoped_to_the_targets_own_tools() -> None:
    """Scope comes from the inspected target, not from a hardcoded name list."""

    rules = _rules(_agent_with(delete_account, update_email, lookup_account))

    assert (PolicyRuleKind.NO_DUPLICATE_SIDE_EFFECT, "delete_account") in rules
    assert (PolicyRuleKind.NO_DUPLICATE_SIDE_EFFECT, "update_email") in rules
    # A read-only lookup is not a side effect and must not acquire rules.
    assert not any(tool == "lookup_account" for _kind, tool in rules)


def test_confirmation_is_derived_only_for_destructive_tools() -> None:
    """Requiring confirmation is the stronger claim, so it needs the stronger signal."""

    rules = _rules(_agent_with(delete_account, update_email))

    assert (PolicyRuleKind.CONFIRMATION_BEFORE_TOOL, "delete_account") in rules
    # update_email changes state but is not destructive: demanding confirmation
    # would invent an expectation the target never made.
    assert (PolicyRuleKind.CONFIRMATION_BEFORE_TOOL, "update_email") not in rules


def test_targets_without_side_effects_derive_nothing() -> None:
    """No state-changing tool means no behavioural rule to scope."""

    assert derive_tool_risk_pack(_spec(_agent_with(lookup_account))) is None
    assert derive_tool_risk_pack(_spec(Agent(name="Chat", instructions="c", model="gpt-4.1-mini"))) is None


def test_derivation_is_deterministic() -> None:
    """Same spec in, byte-identical pack out, with a stable rule order."""

    spec = _spec(_agent_with(update_email, delete_account))
    first = derive_tool_risk_pack(spec)
    second = derive_tool_risk_pack(spec)

    assert first is not None and second is not None
    assert first.model_dump_json() == second.model_dump_json()
    assert [rule.rule_id for rule in first.rules] == [
        rule.rule_id for rule in second.rules
    ]


def test_every_rule_records_the_tool_and_property_that_produced_it() -> None:
    """Provenance: a developer must be able to see why a rule exists."""

    pack = derive_tool_risk_pack(_spec(_agent_with(delete_account)))
    assert pack is not None
    by_kind = {rule.kind: rule for rule in pack.rules}

    confirmation = by_kind[PolicyRuleKind.CONFIRMATION_BEFORE_TOOL]
    assert "delete_account" in confirmation.description
    assert "destructive" in confirmation.description

    duplicate = by_kind[PolicyRuleKind.NO_DUPLICATE_SIDE_EFFECT]
    assert "state-changing" in duplicate.description


def test_pack_is_opt_in(tmp_path: Path) -> None:
    """Not requesting the pack must derive nothing, whatever the tools are."""

    spec = _spec(_agent_with(delete_account))

    assert resolve_policy_packs(tmp_path, AgentCheckConfig(), spec=spec) == ()

    config = AgentCheckConfig(policy_packs=(DERIVED_TOOL_RISK_PACK_ID,))
    resolved = resolve_policy_packs(tmp_path, config, spec=spec)
    assert [pack.pack_id for pack in resolved] == [DERIVED_TOOL_RISK_PACK_ID]


def test_requesting_the_derived_pack_without_a_spec_fails_closed(tmp_path: Path) -> None:
    """It is scoped to an inspected agent, so it cannot resolve before inspection."""

    config = AgentCheckConfig(policy_packs=(DERIVED_TOOL_RISK_PACK_ID,))

    with pytest.raises(ConfigurationError, match="cannot be resolved before inspection"):
        resolve_policy_packs(tmp_path, config, spec=None)


def test_requesting_it_for_a_target_with_no_side_effects_fails_closed(
    tmp_path: Path,
) -> None:
    """Silently contributing zero rules would look like coverage that is not there."""

    config = AgentCheckConfig(policy_packs=(DERIVED_TOOL_RISK_PACK_ID,))

    with pytest.raises(ConfigurationError, match="no state-changing tool"):
        resolve_policy_packs(tmp_path, config, spec=_spec(_agent_with(lookup_account)))


def test_derivation_never_invokes_a_tool_handler() -> None:
    """Risk is read from declarations; the handlers above raise if executed."""

    pack = derive_tool_risk_pack(_spec(_agent_with(delete_account, update_email)))

    assert pack is not None
    assert pack.rules
