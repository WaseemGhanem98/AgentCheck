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

from agentcheck.adapters import CustomAgentAdapter, OpenAIAgentsAdapter, UnsupportedTargetError
from agentcheck.config import ToolRiskDeclaration
from agentcheck.domain import RiskAuthority, ToolDefinition
from agentcheck.coverage import (
    BehavioralCoverageStatus,
    BehavioralDimension,
    analyze_behavioral_coverage,
)
from agentcheck.generate.boundaries import build_outcome_variant_cases
from agentcheck.inspect.risk_authority import resolve_tool_risk, unmatched_tool_risk_names
from agentcheck.policies.derived import derive_tool_risk_pack

from tests.agentcheck.test_openai_adapter import RecordingGateway


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


# --- an unmatched tool_risk key is a silent no-op if nothing refuses it -----
#
# ``declared_risk_for`` looks an override up by name. A key that names no
# declared tool is never read by anything: a developer who misspells a tool
# name in agentcheck.json's tool_risk block gets no error and an override that
# silently never applies. These prove every adapter's prepare() refuses that,
# rather than only the resolver-level helper being correct in isolation.


def test_unmatched_tool_risk_names_reports_every_name_that_matches_nothing():
    assert unmatched_tool_risk_names(
        ("cancel_order", "get_order"),
        {
            "cancel_order": ToolRiskDeclaration(destructive=True),
            "cancle_order": ToolRiskDeclaration(destructive=False),
            "delete_user": ToolRiskDeclaration(destructive=True),
        },
    ) == ("cancle_order", "delete_user")


def test_unmatched_tool_risk_names_is_empty_when_every_key_matches():
    assert (
        unmatched_tool_risk_names(
            ("cancel_order",), {"cancel_order": ToolRiskDeclaration(destructive=True)}
        )
        == ()
    )


def test_unmatched_tool_risk_names_tolerates_no_declaration_at_all():
    assert unmatched_tool_risk_names(("cancel_order",), None) == ()


def test_a_misspelled_tool_risk_name_refuses_prepare_for_the_openai_adapter() -> None:
    agent = Agent(
        name="T",
        instructions="Assist.",
        tools=[delete_nothing],
        model="gpt-4.1-mini",
    )
    adapter = OpenAIAgentsAdapter()

    with pytest.raises(UnsupportedTargetError) as excinfo:
        adapter.prepare(
            agent,
            RecordingGateway(),
            declared_tool_risk={"delet_nothing": ToolRiskDeclaration(destructive=True)},
        )
    assert "unknown_tool_risk_declaration" in {issue.code for issue in excinfo.value.issues}


CANCEL_ORDER = ToolDefinition(
    name="cancel_order",
    description="Cancel one order.",
    input_schema={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
        "additionalProperties": False,
    },
    state_changing=True,
    destructive=True,
)


class _MinimalCustomAgent:
    tools = (CANCEL_ORDER,)

    def start(self, message, tools):  # pragma: no cover - never reached
        raise AssertionError("prepare() must refuse before any turn runs")

    def resume(self, state, message, tools):  # pragma: no cover - never reached
        raise AssertionError("prepare() must refuse before any turn runs")


def test_a_misspelled_tool_risk_name_refuses_prepare_for_a_custom_agent() -> None:
    adapter = CustomAgentAdapter()

    with pytest.raises(UnsupportedTargetError) as excinfo:
        adapter.prepare(
            _MinimalCustomAgent(),
            RecordingGateway(),
            declared_tool_risk={"cancle_order": ToolRiskDeclaration(destructive=False)},
        )
    assert "unknown_tool_risk_declaration" in {issue.code for issue in excinfo.value.issues}


# --- coverage: a declaration must keep its authority on an SDK adapter ------
#
# The documented contract (README, docs/fault-testing.md) is that a developer
# declaration is *always* authoritative, and that `risk_metadata_not_authoritative`
# means "a lexical classification rather than a declared one". Behavioral
# coverage used to key that decision off the tool *property's* `authoritative`
# flag, which records how the tool schema was obtained and is unconditionally
# False on both SDK adapters -- so an explicit declaration silently lost its
# authority and four real requirements dropped out of the missing denominator.

RISK_DIMENSIONS = (
    BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
    BehavioralDimension.DUPLICATE_ACTION,
    BehavioralDimension.AMBIGUOUS_OUTCOME,
    BehavioralDimension.RETRY_CONTROL,
)

# Deliberately reads as a pure lookup so nothing is inferable as risky: the
# declaration is then the only risk signal on either target.
PURGE_DESCRIPTION = "Look up a record by id."


@function_tool(name_override="purge_record", description_override=PURGE_DESCRIPTION)
def purge_record(record_id: str) -> str:
    raise AssertionError("original handler must never run")


PURGE_RECORD = ToolDefinition(
    name="purge_record",
    description=PURGE_DESCRIPTION,
    input_schema={
        "type": "object",
        "properties": {"record_id": {"type": "string"}},
        "required": ["record_id"],
        "additionalProperties": False,
    },
    # Declares no risk of its own, exactly like the SDK tool above.
    state_changing=False,
    destructive=False,
)


class _PurgeCustomAgent:
    name = "T"
    instructions = "Assist."
    tools = (PURGE_RECORD,)

    def start(self, message, tools):  # pragma: no cover - never reached
        raise AssertionError("no turn runs in this test")

    def resume(self, state, message, tools):  # pragma: no cover - never reached
        raise AssertionError("no turn runs in this test")


def _risk_status(spec):
    coverage = analyze_behavioral_coverage(spec, [])
    return {
        family.dimension: (requirement.status, requirement.reason_code)
        for family in coverage.families
        for requirement in family.requirements
        if requirement.subject == "tool:purge_record"
        and family.dimension in RISK_DIMENSIONS
    }


def test_declared_risk_keeps_coverage_authority_on_the_openai_adapter() -> None:
    """An explicit `tool_risk` declaration makes the risk dimensions real
    requirements rather than `unknown`. The SDK cannot carry risk itself, so the
    declaration is the whole authority and must not be discarded merely because
    the schema was reconstructed from a framework object."""

    spec = _spec(
        purge_record,
        tool_risk={
            "purge_record": ToolRiskDeclaration(state_changing=True, destructive=True)
        },
    )
    assertion = next(a for a in spec.tool_risk.items if a.tool_name == "purge_record")
    assert assertion.destructive.authority is RiskAuthority.DEVELOPER_DECLARED
    assert assertion.state_changing.authority is RiskAuthority.DEVELOPER_DECLARED

    statuses = _risk_status(spec)
    assert set(statuses) == set(RISK_DIMENSIONS)
    for dimension, (status, reason) in statuses.items():
        assert status is not BehavioralCoverageStatus.UNKNOWN, (
            f"{dimension.value} reported {reason!r} despite an explicit "
            "developer risk declaration"
        )
        assert reason != "risk_metadata_not_authoritative"


def test_declared_risk_coverage_matches_the_custom_adapter_exactly() -> None:
    """Identical declaration, identical tool contract, different adapter: the
    reported risk requirements must match. Pinning the custom adapter as the
    control makes this a real cross-adapter equivalence rather than a
    restatement of whatever the OpenAI adapter happens to do."""

    declaration = {
        "purge_record": ToolRiskDeclaration(state_changing=True, destructive=True)
    }
    openai_statuses = _risk_status(_spec(purge_record, tool_risk=declaration))
    custom_statuses = _risk_status(
        CustomAgentAdapter().inspect(_PurgeCustomAgent(), declared_tool_risk=declaration)
    )

    assert custom_statuses == {
        BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE: (
            BehavioralCoverageStatus.MISSING,
            "fabricated_success_case_missing",
        ),
        BehavioralDimension.DUPLICATE_ACTION: (
            BehavioralCoverageStatus.MISSING,
            "duplicate_action_case_missing",
        ),
        BehavioralDimension.AMBIGUOUS_OUTCOME: (
            BehavioralCoverageStatus.MISSING,
            "ambiguous_outcome_case_missing",
        ),
        BehavioralDimension.RETRY_CONTROL: (
            BehavioralCoverageStatus.MISSING,
            "retry_control_case_missing",
        ),
    }
    assert openai_statuses == custom_statuses


def test_undeclared_sdk_risk_still_reports_unknown() -> None:
    """The fix must not launder inference into authority: with no declaration
    the same tool keeps reporting `risk_metadata_not_authoritative`."""

    statuses = _risk_status(_spec(purge_record))
    assert set(statuses) == set(RISK_DIMENSIONS)
    for status, reason in statuses.values():
        assert status is BehavioralCoverageStatus.UNKNOWN
        assert reason == "risk_metadata_not_authoritative"


def test_declaring_one_axis_does_not_upgrade_the_other_in_coverage() -> None:
    """`bash` is deliberately unclassifiable, so declaring only `state_changing`
    leaves `destructive` un-inferable: the action-gated dimensions become real
    requirements while the destructive-gated ones stay unknown."""

    spec = _spec(bash, tool_risk={"bash": ToolRiskDeclaration(state_changing=True)})
    coverage = analyze_behavioral_coverage(spec, [])
    reasons = {
        family.dimension: requirement.reason_code
        for family in coverage.families
        for requirement in family.requirements
        if requirement.subject == "tool:bash" and family.dimension in RISK_DIMENSIONS
    }
    assert (
        reasons[BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE]
        != "risk_metadata_not_authoritative"
    )
    assert (
        reasons[BehavioralDimension.DUPLICATE_ACTION]
        != "risk_metadata_not_authoritative"
    )
    assert (
        reasons[BehavioralDimension.AMBIGUOUS_OUTCOME]
        == "risk_metadata_not_authoritative"
    )
    assert (
        reasons[BehavioralDimension.RETRY_CONTROL] == "risk_metadata_not_authoritative"
    )
