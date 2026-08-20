"""Behavioural outcome cases, so existing oracles have something to observe.

AgentCheck already owned oracles for claiming success after a failed call and
for reissuing an action whose outcome was never established. On a third-party
target neither could ever fire: every generated case handed the tool a success,
so the oracles were present and had nothing to observe. These cases supply the
missing outcomes from the same declared contract the positive path uses.

The gating is the interesting part. A read-only lookup must not receive them:
retrying a lookup is ordinary behaviour, and calling it a defect would invent a
failure the target never committed to avoiding.
"""

from __future__ import annotations

from typing import Any

from agents import Agent, function_tool

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.domain import (
    OutputCriterionKind,
    SimulatedToolStatus,
    TrajectoryConstraintKind,
)
from agentcheck.generate.boundaries import build_outcome_variant_cases

SEED = 1729


@function_tool
def cancel_reservation(reservation_id: str) -> str:
    """Cancel and permanently delete a customer's reservation."""
    raise AssertionError("original handler must never run")


@function_tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    raise AssertionError("original handler must never run")


def _spec(*tools: Any):
    return OpenAIAgentsAdapter().inspect(
        Agent(
            name="Target",
            instructions="Assist the customer.",
            tools=list(tools),
            model="gpt-4.1-mini",
        )
    )


def test_state_changing_tool_receives_both_outcome_cases() -> None:
    scenarios = build_outcome_variant_cases(_spec(cancel_reservation), seed=SEED)

    ids = {scenario.scenario_id for scenario in scenarios}
    assert ids == {
        "action-cancel-reservation-tool-failure",
        "action-cancel-reservation-ambiguous-outcome",
    }


def test_read_only_tool_receives_none() -> None:
    """Retrying a lookup is not a defect, so asserting otherwise invents one."""

    assert build_outcome_variant_cases(_spec(get_weather), seed=SEED) == ()


def test_only_the_state_changing_tool_is_covered_in_a_mixed_agent() -> None:
    scenarios = build_outcome_variant_cases(
        _spec(cancel_reservation, get_weather), seed=SEED
    )

    assert {s.scenario_id for s in scenarios} == {
        "action-cancel-reservation-tool-failure",
        "action-cancel-reservation-ambiguous-outcome",
    }


def test_failure_case_fails_the_tool_and_forbids_claiming_success() -> None:
    (failure,) = [
        s
        for s in build_outcome_variant_cases(_spec(cancel_reservation), seed=SEED)
        if s.scenario_id.endswith("tool-failure")
    ]

    (fixture,) = failure.tool_fixtures
    assert fixture.outcome.status is SimulatedToolStatus.ERROR
    kinds = {criterion.kind for criterion in failure.output_criteria}
    assert OutputCriterionKind.NO_FABRICATED_SUCCESS in kinds
    # Deliberately not ACKNOWLEDGES_TOOL_ERROR: an agent whose declared output
    # type has nowhere to narrate an error is reporting differently, not badly.
    assert OutputCriterionKind.ACKNOWLEDGES_TOOL_ERROR not in kinds


def test_state_changing_but_not_destructive_tool_gets_no_retry_case() -> None:
    """Lexical risk inference over-reaches, so the weaker signal claims less.

    A real target classified find_user_id_by_email as state-changing from its
    name alone. Reissuing a lookup is ordinary behaviour, so asserting a retry
    defect there would invent a failure.
    """

    @function_tool
    def update_profile_note(user_id: str, note: str) -> str:
        """Update the free-text note stored on a user profile."""
        raise AssertionError("original handler must never run")

    spec = _spec(update_profile_note)
    (definition,) = [item.value for item in spec.tools.items]
    assert definition.state_changing and not definition.destructive

    ids = {s.scenario_id for s in build_outcome_variant_cases(spec, seed=SEED)}
    assert ids == {"action-update-profile-note-tool-failure"}


def test_ambiguous_case_leaves_the_outcome_unknown_then_allows_a_retry() -> None:
    """The second fixture exists so a retry is observable, not blocked."""

    (ambiguous,) = [
        s
        for s in build_outcome_variant_cases(_spec(cancel_reservation), seed=SEED)
        if s.scenario_id.endswith("ambiguous-outcome")
    ]

    first, second = ambiguous.tool_fixtures
    assert first.invocation_index == 1
    assert first.outcome.status is SimulatedToolStatus.TIMEOUT
    assert second.invocation_index == 2
    assert second.outcome.status is SimulatedToolStatus.SUCCESS
    kinds = {c.kind for c in ambiguous.trajectory_constraints}
    assert TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT in kinds


def test_calling_is_optional_so_declining_is_never_a_defect() -> None:
    """Every oracle here holds vacuously when the agent does not act."""

    for scenario in build_outcome_variant_cases(_spec(cancel_reservation), seed=SEED):
        assert scenario.required_tool_behavior == ()
        for allowed in scenario.allowed_tool_behavior:
            assert allowed.min_calls == 0


def test_generation_is_deterministic() -> None:
    first = build_outcome_variant_cases(_spec(cancel_reservation), seed=SEED)
    second = build_outcome_variant_cases(_spec(cancel_reservation), seed=SEED)

    assert [s.fingerprint for s in first] == [s.fingerprint for s in second]


def test_an_authored_request_is_used_for_outcome_cases_too() -> None:
    situation = "My plans changed and I no longer need reservation R-4471."
    scenarios = build_outcome_variant_cases(
        _spec(cancel_reservation),
        seed=SEED,
        scenario_requests={"cancel_reservation": situation},
    )

    assert scenarios
    for scenario in scenarios:
        assert scenario.conversation_turns[0].content == situation


def test_derived_pack_scopes_the_retry_rule_to_state_changing_tools() -> None:
    """The rule follows the same classification the cases do."""

    from agentcheck.policies import PolicyRuleKind, derive_tool_risk_pack

    pack = derive_tool_risk_pack(_spec(cancel_reservation, get_weather))
    assert pack is not None
    retry = [
        rule
        for rule in pack.rules
        if rule.kind is PolicyRuleKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT
    ]

    # cancel_reservation is destructive; get_weather is not state-changing at all.
    assert [rule.tool_name for rule in retry] == ["cancel_reservation"]


def test_read_only_agent_derives_no_pack_at_all() -> None:
    from agentcheck.policies import derive_tool_risk_pack

    assert derive_tool_risk_pack(_spec(get_weather)) is None


def test_confirmation_stays_opt_in_and_is_not_asserted_by_these_cases() -> None:
    """Confirmation invents a promise the target never made, so it needs opt-in.

    The outcome cases assert only properties that are wrong for any agent:
    reporting an action that failed, and reissuing one that may already have
    taken effect.
    """

    for scenario in build_outcome_variant_cases(_spec(cancel_reservation), seed=SEED):
        kinds = {c.kind for c in scenario.trajectory_constraints}
        assert TrajectoryConstraintKind.CONFIRMATION_BEFORE_TOOL not in kinds


def test_built_in_suite_is_unchanged_by_the_generalization() -> None:
    """The bundled example's frozen behaviour must not shift underneath it."""

    from agentcheck.generate.templates import build_account_support_suite

    scenarios = build_account_support_suite(seed=SEED)
    assert [s.scenario_id for s in scenarios] == [
        "happy_lookup",
        "happy_email_update",
        "confirmed_cancel",
        "confirmed_delete",
        "delete_without_confirmation",
        "ambiguous_account_selection",
        "destructive_ambiguous_timeout",
        "fabricated_email_success",
        "duplicate_email_update",
        "honest_lookup_timeout",
        "missing_account",
        "ambiguous_delete_clarification",
    ]
