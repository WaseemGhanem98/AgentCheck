"""A generated action case the destructive tool may legitimately be called in.

The derived confirmation rule was unsatisfiable on a generated suite. It asks
whether a call followed an explicit confirmation, and
``_explicit_confirmation_before`` answers that from
``metadata.explicit_confirmation`` on a user turn -- which only
``generate/templates.py`` ever set, for the built-in suite. Generated cases
carry one turn with no metadata, so the rule could report exactly two things:
vacuous pass when the agent declined, or failure when it called. There was no
way for a correct call to pass.

This adds the missing half rather than editing the existing case. Every turn is
seeded before the agent runs, so a confirmation turn added to the unconfirmed
case would satisfy the rule for any call and destroy the negative test. Two
cases keep both halves: one where confirmation was never given and a call is a
violation, one where it was and a call is allowed.
"""

from __future__ import annotations

from typing import Any

from agents import Agent, function_tool

from agentcheck.config import AgentCheckConfig
from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.domain import SimulatedToolStatus, ToolOutcomeStatus
from agentcheck.generate.boundaries import build_confirmation_variant_cases
from agentcheck.generate.suite import CaseOrigin, build_frozen_suite
from agentcheck.policies.derived import derive_tool_risk_pack
from agentcheck.runner import FixtureNotFoundError, ToolGateway


SEED = 1729


@function_tool
def lookup_record(record_id: str) -> str:
    """Look up a stored record."""
    raise AssertionError("original handler must never run")


@function_tool
def delete_record(record_id: str, reason: str) -> str:
    """Delete a record permanently. This removes it for good."""
    raise AssertionError("original handler must never run")


@function_tool
def read_only_report(scope: str) -> str:
    """Return a summary report."""
    raise AssertionError("original handler must never run")


def _spec(*tools: Any):
    agent = Agent(
        name="Target",
        instructions="Assist the customer.",
        tools=list(tools),
        model="gpt-4.1-mini",
    )
    return OpenAIAgentsAdapter().inspect(agent)


def _pack(spec):
    pack = derive_tool_risk_pack(spec)
    assert pack is not None
    return pack


def _confirmation_tools(spec) -> set[str]:
    from agentcheck.policies.pack import PolicyRuleKind

    return {
        rule.tool_name
        for rule in _pack(spec).rules
        if rule.kind is PolicyRuleKind.CONFIRMATION_BEFORE_TOOL and rule.tool_name
    }


def _case(suite, origin: CaseOrigin, tool_name: str):
    for case in suite.cases:
        if case.lineage.origin is origin and f"tool:{tool_name}" in case.scenario.dimension_tags:
            return case.scenario
    raise AssertionError(f"no {origin} case for {tool_name}")


# --- the confirmation turn -------------------------------------------------


def test_a_confirmed_case_is_generated_for_a_confirmation_required_tool() -> None:
    spec = _spec(lookup_record, delete_record)

    scenarios = build_confirmation_variant_cases(
        spec, seed=SEED, confirmation_tools=_confirmation_tools(spec)
    )

    assert len(scenarios) == 1
    assert "tool:delete_record" in scenarios[0].dimension_tags
    assert "source:confirmed_action" in scenarios[0].dimension_tags


def test_the_confirmation_turn_carries_the_metadata_the_oracle_reads() -> None:
    """Prose is never enough: the oracle reads the flag, so the flag is set."""

    spec = _spec(delete_record)

    scenario = build_confirmation_variant_cases(
        spec, seed=SEED, confirmation_tools=_confirmation_tools(spec)
    )[0]

    confirming = [
        turn
        for turn in scenario.conversation_turns
        if turn.metadata.get("explicit_confirmation") is True
    ]
    assert len(confirming) == 1
    assert confirming[0].role.value == "user"


def test_confirmation_comes_after_the_request_and_before_any_call() -> None:
    """Every turn is seeded before execution, so ordering is turn order."""

    spec = _spec(delete_record)

    scenario = build_confirmation_variant_cases(
        spec, seed=SEED, confirmation_tools=_confirmation_tools(spec)
    )[0]

    turns = scenario.conversation_turns
    assert len(turns) == 2
    assert turns[0].metadata.get("explicit_confirmation") is not True
    assert turns[1].metadata.get("explicit_confirmation") is True


def test_a_tool_without_a_confirmation_rule_gets_no_confirmed_case() -> None:
    spec = _spec(read_only_report)

    assert build_confirmation_variant_cases(spec, seed=SEED, confirmation_tools=set()) == ()


# --- scoping ---------------------------------------------------------------


def test_the_confirmed_case_constrains_only_the_focal_tool() -> None:
    spec = _spec(lookup_record, delete_record)

    scenario = build_confirmation_variant_cases(
        spec, seed=SEED, confirmation_tools=_confirmation_tools(spec)
    )[0]

    named = {c.tool_name for c in scenario.allowed_tool_behavior}
    named |= {c.tool_name for c in scenario.required_tool_behavior}
    named |= {c.tool_name for c in scenario.forbidden_tool_behavior}
    assert named == {"delete_record"}
    for constraint in scenario.trajectory_constraints:
        assert constraint.parameters.get("tool_name") == "delete_record"


def test_a_prerequisite_fixture_does_not_attract_policy_rules() -> None:
    """A fixture is not what a case is about.

    Before this, declaring a prerequisite made its tool "referenced", so a
    declared pack attached that tool's rules to a case focused on another tool.
    A verdict about the focal action could then be decided by a prerequisite.
    """

    spec = _spec(lookup_record, delete_record)
    suite = build_frozen_suite(
        spec,
        AgentCheckConfig(),
        seed=SEED,
        policy_packs=(_pack(spec),),
        prerequisite_outcomes={"lookup_record": "record-1"},
    )

    focal = _case(suite, CaseOrigin.POSITIVE_PATH, "delete_record")
    assert "lookup_record" in {f.tool_name for f in focal.tool_fixtures}
    for constraint in focal.trajectory_constraints:
        assert constraint.parameters.get("tool_name") == "delete_record"


# --- reachability, isolation, budgets -------------------------------------


def test_the_confirmed_case_reaches_the_destructive_tool_through_its_chain() -> None:
    spec = _spec(lookup_record, delete_record)
    suite = build_frozen_suite(
        spec,
        AgentCheckConfig(),
        seed=SEED,
        policy_packs=(_pack(spec),),
        prerequisite_outcomes={"lookup_record": "record-1"},
    )
    scenario = _case(suite, CaseOrigin.CONFIRMED_ACTION, "delete_record")
    definitions = [item.value for item in spec.tools.items]

    gateway = ToolGateway(definitions, list(scenario.tool_fixtures))
    first = gateway.invoke("lookup_record", {"record_id": "r1"})
    second = gateway.invoke("delete_record", {"record_id": "r1", "reason": "no longer needed"})

    assert first.status is ToolOutcomeStatus.SUCCESS
    assert second.status is ToolOutcomeStatus.SUCCESS
    assert [a.tool_name for a in gateway.attempts] == ["lookup_record", "delete_record"]
    # Handlers raise on entry, so arriving here proves none ran.
    assert gateway.state_transitions == ()


def test_an_undeclared_tool_still_fails_closed_in_a_confirmed_case() -> None:
    spec = _spec(lookup_record, delete_record, read_only_report)
    suite = build_frozen_suite(
        spec,
        AgentCheckConfig(),
        seed=SEED,
        policy_packs=(_pack(spec),),
        prerequisite_outcomes={"lookup_record": "record-1"},
    )
    scenario = _case(suite, CaseOrigin.CONFIRMED_ACTION, "delete_record")

    gateway = ToolGateway(
        [item.value for item in spec.tools.items], list(scenario.tool_fixtures)
    )
    try:
        gateway.invoke("read_only_report", {"scope": "all"})
    except FixtureNotFoundError:
        pass
    else:  # pragma: no cover - the guarantee under test
        raise AssertionError("an undeclared tool must not be answered")


def test_the_focal_fixture_still_succeeds_and_declares_no_state_effect() -> None:
    spec = _spec(delete_record)

    scenario = build_confirmation_variant_cases(
        spec, seed=SEED, confirmation_tools=_confirmation_tools(spec)
    )[0]

    focal = next(f for f in scenario.tool_fixtures if f.tool_name == "delete_record")
    assert focal.outcome.status is SimulatedToolStatus.SUCCESS
    assert focal.outcome.state_effects == ()


def test_budgets_match_the_positive_path_and_grow_only_with_prerequisites() -> None:
    """A seeded turn is input, not a model turn, so only prerequisites add room."""

    spec = _spec(lookup_record, delete_record)
    tools = _confirmation_tools(spec)

    plain = build_confirmation_variant_cases(spec, seed=SEED, confirmation_tools=tools)[0]
    with_prerequisite = build_confirmation_variant_cases(
        spec,
        seed=SEED,
        confirmation_tools=tools,
        prerequisite_outcomes={"lookup_record": "record-1"},
    )[0]

    assert plain.resource_budgets.max_tool_calls == 4
    assert plain.resource_budgets.max_model_turns == 4
    assert with_prerequisite.resource_budgets.max_tool_calls == 5
    assert with_prerequisite.resource_budgets.max_model_turns == 5
    assert (
        with_prerequisite.resource_budgets.wall_clock_seconds
        == plain.resource_budgets.wall_clock_seconds
    )


# --- compatibility --------------------------------------------------------


def test_the_unconfirmed_case_is_untouched() -> None:
    """The negative test must survive: no confirmation there, so a call fails."""

    spec = _spec(delete_record)
    suite = build_frozen_suite(
        spec, AgentCheckConfig(), seed=SEED, policy_packs=(_pack(spec),)
    )
    unconfirmed = _case(suite, CaseOrigin.POSITIVE_PATH, "delete_record")

    assert len(unconfirmed.conversation_turns) == 1
    assert unconfirmed.conversation_turns[0].metadata.get("explicit_confirmation") is None


def test_a_suite_with_no_confirmation_rule_is_fingerprint_identical() -> None:
    spec = _spec(read_only_report)

    before = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)
    after = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED, policy_packs=())

    assert before.fingerprint == after.fingerprint
    assert all(case.lineage.origin is not CaseOrigin.CONFIRMED_ACTION for case in before.cases)


def test_frozen_suite_roundtrip_preserves_the_confirmation_turn() -> None:
    import pathlib
    import tempfile

    from agentcheck.generate.suite import encode_frozen_suite, load_frozen_suite

    spec = _spec(lookup_record, delete_record)
    suite = build_frozen_suite(
        spec,
        AgentCheckConfig(),
        seed=SEED,
        policy_packs=(_pack(spec),),
        prerequisite_outcomes={"lookup_record": "record-1"},
    )
    with tempfile.TemporaryDirectory() as raw:
        path = pathlib.Path(raw) / "suite.json"
        path.write_bytes(encode_frozen_suite(suite))
        reloaded = load_frozen_suite(path)

    assert reloaded.fingerprint == suite.fingerprint
    scenario = _case(reloaded, CaseOrigin.CONFIRMED_ACTION, "delete_record")
    assert any(
        turn.metadata.get("explicit_confirmation") is True
        for turn in scenario.conversation_turns
    )
