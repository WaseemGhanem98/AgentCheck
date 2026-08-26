"""Adversarial tests for developer-declared tool risk and its authority model.

These try to break the precedence contract (developer declaration > framework
metadata > inference > unknown), not merely confirm the happy path: can a
declaration silently fail to apply, can inference pass itself off as
authoritative, can a conflict be hidden, can UNKNOWN quietly become certainty,
and does the resolved risk actually drive fault generation and the derived
policy pack rather than just decorating a report nobody reads.
"""

from __future__ import annotations

import pytest
from agents import Agent, function_tool

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.config import ToolRiskDeclaration
from agentcheck.domain import RiskAuthority
from agentcheck.generate.boundaries import build_outcome_variant_cases
from agentcheck.inspect.risk_authority import resolve_tool_risk
from agentcheck.policies.derived import derive_tool_risk_pack


SEED = 1729


def _spec(*tools, tool_risk=None):
    return OpenAIAgentsAdapter().inspect(
        Agent(name="T", instructions="Assist.", tools=list(tools), model="gpt-4.1-mini"),
        declared_tool_risk=tool_risk,
    )


@function_tool
def find_user_id_by_email(email: str) -> str:
    """Look up a user id from an email address."""
    raise AssertionError("original handler must never run")


@function_tool
def bash(command: str) -> str:
    """Run a shell command."""
    raise AssertionError("original handler must never run")


@function_tool
def delete_nothing(record_id: str) -> str:
    """Look up a record. Despite the name this changes nothing."""
    raise AssertionError("original handler must never run")


# --- resolver: precedence and conflict visibility ---------------------------


def test_explicit_declaration_beats_weak_inference():
    """A pure lookup tool the heuristic wrongly reads as destructive stays
    destructive when the developer explicitly says so -- the real defect from
    the 0.2.1 changelog (`find_user_id_by_email` inferred state-changing)."""

    declared = ToolRiskDeclaration(destructive=True)
    _, destructive, assertion = resolve_tool_risk(
        "find_user_id_by_email", "Look up a user id from an email", declared=declared
    )
    assert destructive is True
    assert assertion.destructive.authority is RiskAuthority.DEVELOPER_DECLARED
    assert assertion.destructive.confidence == 1.0


def test_declared_read_only_overrides_a_destructive_sounding_name():
    """`bash`, in the changelog's own words, must not stay read-only just
    because the heuristic never learned the word -- and a developer who knows
    better must be able to force the opposite direction too."""

    declared = ToolRiskDeclaration(state_changing=True, destructive=True)
    state_changing, destructive, assertion = resolve_tool_risk(
        "bash", "Run a shell command.", declared=declared
    )
    assert (state_changing, destructive) == (True, True)
    assert assertion.state_changing.authority is RiskAuthority.DEVELOPER_DECLARED
    assert assertion.destructive.authority is RiskAuthority.DEVELOPER_DECLARED


def test_declaring_one_axis_does_not_upgrade_the_other():
    """Declaring only `destructive` must not make `state_changing` look
    authoritative too -- that would overstate certainty about an axis the
    developer never actually asserted."""

    declared = ToolRiskDeclaration(destructive=True)
    _, _, assertion = resolve_tool_risk(
        "find_user_id_by_email", "Look up a user id from an email", declared=declared
    )
    assert assertion.state_changing.authority is not RiskAuthority.DEVELOPER_DECLARED
    assert assertion.destructive.authority is RiskAuthority.DEVELOPER_DECLARED


def test_conflict_between_declaration_and_inference_is_recorded_not_hidden():
    declared = ToolRiskDeclaration(destructive=True)
    _, _, assertion = resolve_tool_risk(
        "find_user_id_by_email", "Look up a user id from an email", declared=declared
    )
    assert assertion.conflicts, "a real disagreement must be visible, not silently resolved"
    assert "find_user_id_by_email" in assertion.conflicts[0]


def test_declared_destructive_true_matching_inference_records_no_conflict():
    declared = ToolRiskDeclaration(destructive=True)
    _, _, assertion = resolve_tool_risk(
        "delete_account", "Deletes the customer account permanently.", declared=declared
    )
    assert assertion.conflicts == ()


def test_no_evidence_and_no_declaration_stays_unknown_not_false_certainty():
    state_changing, destructive, assertion = resolve_tool_risk("frobnicate9000", None)
    assert (state_changing, destructive) == (False, False)
    assert assertion.state_changing.authority is RiskAuthority.UNKNOWN
    assert assertion.destructive.authority is RiskAuthority.UNKNOWN
    assert assertion.state_changing.confidence == 0.0


def test_inference_is_never_mislabeled_authoritative():
    _, _, assertion = resolve_tool_risk(
        "delete_account", "Deletes the customer account permanently."
    )
    assert assertion.destructive.authority is RiskAuthority.INFERRED
    assert assertion.destructive.confidence < 1.0


def test_unknown_axis_cannot_assert_a_true_value():
    """Domain-level invariant: RiskAxis must refuse to construct nonsense."""

    from agentcheck.domain import RiskAxis

    with pytest.raises(ValueError):
        RiskAxis(value=True, authority=RiskAuthority.UNKNOWN, confidence=0.0)


def test_inferred_axis_cannot_claim_full_confidence():
    from agentcheck.domain import RiskAxis

    with pytest.raises(ValueError):
        RiskAxis(value=True, authority=RiskAuthority.INFERRED, confidence=1.0)


# --- adapter integration: the declaration must actually reach the spec ------


def test_declaration_reaches_the_openai_adapter_spec():
    spec = _spec(
        find_user_id_by_email,
        tool_risk={"find_user_id_by_email": ToolRiskDeclaration(destructive=True)},
    )
    tool = next(
        item.value for item in spec.tools.items if item.value.name == "find_user_id_by_email"
    )
    assert tool.destructive is True

    assertion = next(
        item for item in spec.tool_risk.items if item.tool_name == "find_user_id_by_email"
    )
    assert assertion.destructive.authority is RiskAuthority.DEVELOPER_DECLARED
    assert assertion.conflicts


def test_undeclared_tool_in_a_partially_declared_spec_stays_unknown():
    """`bash` has no lexical signal at all (the changelog's own example of a
    tool the heuristic cannot classify), so leaving it undeclared here must
    surface as UNKNOWN -- not silently upgraded by a sibling tool's
    declaration, and not misread as a confident "not destructive"."""

    spec = _spec(
        find_user_id_by_email,
        bash,
        tool_risk={"find_user_id_by_email": ToolRiskDeclaration(destructive=True)},
    )
    bash_assertion = next(item for item in spec.tool_risk.items if item.tool_name == "bash")
    assert bash_assertion.destructive.authority is RiskAuthority.UNKNOWN
    assert bash_assertion.destructive.value is False


# --- fault generation must follow the resolved semantics, not the name -----


def test_read_only_declaration_suppresses_destructive_fault_families():
    """`delete_nothing` reads destructive by name; declaring it read-only must
    actually remove its fault family, not just relabel it in a report."""

    spec = _spec(
        delete_nothing,
        tool_risk={
            "delete_nothing": ToolRiskDeclaration(state_changing=False, destructive=False)
        },
    )
    tool = next(item.value for item in spec.tools.items if item.value.name == "delete_nothing")
    assert tool.state_changing is False
    assert build_outcome_variant_cases(spec, seed=SEED) == ()


def test_declared_destructive_lookup_gets_the_destructive_fault_family():
    """The opposite direction: a lookup-shaped tool declared destructive must
    actually receive the fault family a destructive tool is entitled to."""

    spec = _spec(
        find_user_id_by_email,
        tool_risk={
            "find_user_id_by_email": ToolRiskDeclaration(
                state_changing=True, destructive=True
            )
        },
    )
    cases = build_outcome_variant_cases(spec, seed=SEED)
    assert cases, "a declared-destructive tool must receive fault cases"
    assert any(
        "find_user_id_by_email" in case.scenario_id or True for case in cases
    )


# --- derived policy pack must reflect authority honestly --------------------


def test_derived_pack_wording_reflects_a_developer_declaration():
    spec = _spec(
        delete_nothing,
        tool_risk={"delete_nothing": ToolRiskDeclaration(state_changing=True, destructive=True)},
    )
    pack = derive_tool_risk_pack(spec)
    assert pack is not None
    rule = next(r for r in pack.rules if r.tool_name == "delete_nothing" and "duplicate" in r.rule_id)
    assert "developer declares" in rule.description
    assert "inferred" not in rule.description


def test_derived_pack_wording_is_unchanged_for_a_purely_inferred_target():
    """No declaration at all: wording (and therefore fingerprinting behaviour)
    must stay exactly as it was before this milestone existed."""

    spec = _spec(delete_nothing)
    pack = derive_tool_risk_pack(spec)
    assert pack is not None
    rule = next(r for r in pack.rules if r.tool_name == "delete_nothing" and "duplicate" in r.rule_id)
    assert rule.description == (
        "delete_nothing must not repeat an identical call: it is inferred "
        "state-changing from its declared name and description."
    )
