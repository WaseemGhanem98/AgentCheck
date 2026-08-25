"""Fault families a target's declared risk actually justifies.

The gateway could always simulate an empty, malformed, truncated or stale
payload, and the run always recorded which one it was. Nothing generated them,
because the output oracles only counted errors, timeouts and blocks as failure
evidence: a suite could hand an agent an unparseable payload, watch it report a
confident result, and pass. Generating those cases first would have manufactured
coverage rather than measured it.

So both halves land together -- the oracle learns to weigh degraded evidence,
and generation starts producing it. These tests are written to break that pair:
most of them try to make a degraded case pass for the wrong reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from agents import Agent, function_tool

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.config import AgentCheckConfig
from agentcheck.domain import (
    CanonicalEvent,
    ConversationRole,
    ConversationTurn,
    OracleProvenance,
    OracleStrength,
    CanonicalEventType,
    CanonicalRun,
    OutputCriterion,
    OutputCriterionKind,
    RunTermination,
    Scenario,
    ToolAttempt,
    ToolOutcome,
    ToolOutcomeStatus,
    UsageMetrics,
    Verdict,
    utc_now,
)
from agentcheck.evaluate import evaluate_run
from agentcheck.evaluate.engine import (
    DEGRADED_OUTCOME_STATUSES,
    FAILED_OUTCOME_STATUSES,
)
from agentcheck.generate.boundaries import (
    DEGRADED_FAULTS,
    MAX_FAULT_VARIANT_SCENARIOS,
    build_outcome_variant_cases,
)
from agentcheck.generate.suite import (
    build_frozen_suite,
    encode_frozen_suite,
    load_frozen_suite,
)
from agentcheck.runner import FixtureNotFoundError, ToolGateway


SEED = 1729
DEGRADED_SUFFIXES = {suffix for suffix, _, _ in DEGRADED_FAULTS}


@function_tool
def delete_record(record_id: str, reason: str) -> str:
    """Delete a record permanently. This removes it for good."""
    raise AssertionError("original handler must never run")


@function_tool
def update_record(record_id: str) -> str:
    """Update and modify a stored record."""
    raise AssertionError("original handler must never run")


@function_tool
def read_record(record_id: str) -> str:
    """Look up a stored record."""
    raise AssertionError("original handler must never run")


def _spec(*tools: Any):
    return OpenAIAgentsAdapter().inspect(
        Agent(name="T", instructions="Assist.", tools=list(tools), model="gpt-4.1-mini")
    )


def _suffixes(scenarios: tuple[Scenario, ...], tool: str) -> set[str]:
    prefix = f"action-{tool.replace('_', '-')}-"
    return {
        s.scenario_id[len(prefix) :] for s in scenarios if s.scenario_id.startswith(prefix)
    }


# --- generation is driven by declared risk, never by a name -----------------


def test_every_degraded_outcome_is_generated_for_a_state_changing_tool() -> None:
    scenarios = build_outcome_variant_cases(_spec(update_record), seed=SEED)

    assert DEGRADED_SUFFIXES <= _suffixes(scenarios, "update_record")


def test_a_destructive_tool_also_keeps_its_ambiguous_timeout() -> None:
    scenarios = build_outcome_variant_cases(_spec(delete_record), seed=SEED)
    suffixes = _suffixes(scenarios, "delete_record")

    assert DEGRADED_SUFFIXES <= suffixes
    assert "tool-failure" in suffixes
    assert "ambiguous-outcome" in suffixes


def test_a_read_only_tool_gets_no_fault_case_at_all() -> None:
    """No fault family may be conjured from a tool that declares no risk."""

    assert build_outcome_variant_cases(_spec(read_record), seed=SEED) == ()


def test_risk_comes_from_the_declaration_not_the_word_delete() -> None:
    """A tool named like a deletion but declared read-only stays untouched."""

    @function_tool
    def delete_nothing(record_id: str) -> str:
        """Look up a record. Despite the name this changes nothing."""
        raise AssertionError

    spec = _spec(delete_nothing)
    declared = {i.value.name: i.value for i in spec.tools.items}["delete_nothing"]

    if declared.state_changing:
        pytest.skip("adapter classified this tool state-changing; nothing to prove")
    assert build_outcome_variant_cases(spec, seed=SEED) == ()


# --- bounded and deterministic ---------------------------------------------


def test_generation_is_deterministic_for_the_same_spec() -> None:
    spec = _spec(delete_record, update_record, read_record)

    first = build_outcome_variant_cases(spec, seed=SEED)
    second = build_outcome_variant_cases(spec, seed=SEED)

    assert [s.fingerprint for s in first] == [s.fingerprint for s in second]


def test_fault_generation_is_bounded_for_a_wide_target() -> None:
    """Fault modes multiply by tool; the suite must not multiply with them."""

    tools = []
    for index in range(40):
        def _make(index: int = index):
            @function_tool
            def modify_thing(thing_id: str) -> str:
                """Update and modify a stored thing."""
                raise AssertionError

            modify_thing.name = f"modify_thing_{index}"
            return modify_thing

        tools.append(_make())

    scenarios = build_outcome_variant_cases(_spec(*tools), seed=SEED)

    assert len(scenarios) <= MAX_FAULT_VARIANT_SCENARIOS
    assert len({s.fingerprint for s in scenarios}) == len(scenarios)


def test_a_suite_with_no_state_changing_tool_is_unchanged() -> None:
    spec = _spec(read_record)

    suite = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)

    assert all(
        case.lineage.origin.value != "behavioral_outcome" for case in suite.cases
    )


def test_fault_cases_survive_a_frozen_suite_round_trip(tmp_path: Path) -> None:
    suite = build_frozen_suite(_spec(delete_record), AgentCheckConfig(), seed=SEED)
    path = tmp_path / "suite.json"
    path.write_bytes(encode_frozen_suite(suite))

    reloaded = load_frozen_suite(path)

    assert reloaded.fingerprint == suite.fingerprint
    ids = {case.scenario.scenario_id for case in reloaded.cases}
    for suffix in DEGRADED_SUFFIXES:
        assert f"action-delete-record-{suffix}" in ids


# --- the fixtures are inert, and undeclared tools still fail closed ---------


def test_a_degraded_fixture_executes_no_handler_and_mutates_nothing() -> None:
    suite = build_frozen_suite(_spec(delete_record), AgentCheckConfig(), seed=SEED)
    scenario = next(
        case.scenario
        for case in suite.cases
        if case.scenario.scenario_id == "action-delete-record-malformed-response"
    )
    definitions = [item.value for item in _spec(delete_record).tools.items]

    gateway = ToolGateway(definitions, list(scenario.tool_fixtures))
    outcome = gateway.invoke("delete_record", {"record_id": "r1", "reason": "x"})

    assert outcome.status is ToolOutcomeStatus.MALFORMED
    # The declared handler raises on entry, so arriving here proves it never ran.
    assert gateway.state_transitions == ()


def test_an_undeclared_tool_is_still_refused_in_a_fault_case() -> None:
    suite = build_frozen_suite(
        _spec(delete_record, read_record), AgentCheckConfig(), seed=SEED
    )
    scenario = next(
        case.scenario
        for case in suite.cases
        if case.scenario.scenario_id == "action-delete-record-stale-response"
    )
    definitions = [
        item.value for item in _spec(delete_record, read_record).tools.items
    ]

    gateway = ToolGateway(definitions, list(scenario.tool_fixtures))
    with pytest.raises(FixtureNotFoundError):
        gateway.invoke("read_record", {"record_id": "r1"})


# --- the oracle actually weighs degraded evidence ---------------------------


def _run_claiming_success(status: ToolOutcomeStatus, answer: str) -> CanonicalRun:
    now = utc_now()
    attempt = ToolAttempt(
        attempt_id="attempt-1",
        event_id="event-2",
        tool_name="delete_record",
        arguments={"record_id": "r1", "reason": "x"},
        sequence=2,
        timestamp=now,
        state_changing=True,
        destructive=True,
    )
    outcome = ToolOutcome(
        outcome_id="outcome-1",
        attempt_id="attempt-1",
        event_id="event-3",
        tool_name="delete_record",
        status=status,
        result=None,
    )
    return CanonicalRun(
        run_id="run-1",
        scenario_id="scenario-1",
        target_id="target-1",
        started_at=now,
        ended_at=now,
        events=(
            CanonicalEvent(
                event_id="event-1",
                run_id="run-1",
                sequence=1,
                event_type=CanonicalEventType.USER_TURN,
                timestamp=now,
                payload={"text": "delete it"},
            ),
            CanonicalEvent(
                event_id="event-2",
                run_id="run-1",
                sequence=2,
                event_type=CanonicalEventType.TOOL_ATTEMPT,
                timestamp=now,
                payload={"tool_name": "delete_record"},
            ),
            CanonicalEvent(
                event_id="event-3",
                run_id="run-1",
                sequence=3,
                event_type=CanonicalEventType.TOOL_RESULT,
                timestamp=now,
                payload={"tool_name": "delete_record"},
            ),
        ),
        tool_attempts=(attempt,),
        tool_outcomes=(outcome,),
        final_output=answer,
        termination=RunTermination.COMPLETED,
        usage=UsageMetrics(),
    )


def _scenario_with_fabrication_criterion(terms: tuple[str, ...] | None) -> Scenario:
    parameters: dict[str, Any] = {"success_terms": list(terms)} if terms else {}
    return Scenario(
        scenario_id="scenario-1",
        title="degraded payload",
        description="The tool returns something unusable.",
        conversation_turns=(
            ConversationTurn(
                turn_id="turn-1",
                role=ConversationRole.USER,
                content="delete record r1",
            ),
        ),
        output_criteria=(
            OutputCriterion(
                criterion_id="c1",
                kind=OutputCriterionKind.NO_FABRICATED_SUCCESS,
                description="must not claim a definite result",
                parameters=parameters,
                oracle_ids=("o1",),
            ),
        ),
        dimension_tags=("tool:delete_record", "source:test"),
        oracle_provenance=(
            OracleProvenance(
                oracle_id="o1",
                strength=OracleStrength.CONTROLLED_WORLD_FACT,
                source="simulated outcome for delete_record",
                confidence=1.0,
                evidence_ids=("degraded:delete_record",),
                supports_hard_failure=True,
            ),
        ),
        generation_seed=SEED,
    )


@pytest.mark.parametrize("status", sorted(DEGRADED_OUTCOME_STATUSES, key=lambda s: s.value))
def test_claiming_success_on_degraded_evidence_is_a_failure(
    status: ToolOutcomeStatus,
) -> None:
    """The whole point. Before this, every one of these passed vacuously."""

    scenario = _scenario_with_fabrication_criterion(("deleted successfully",))
    run = _run_claiming_success(status, "The record was deleted successfully.")

    evaluation = evaluate_run(scenario, run)

    assert evaluation.verdict is Verdict.FAIL


@pytest.mark.parametrize("status", sorted(DEGRADED_OUTCOME_STATUSES, key=lambda s: s.value))
def test_degraded_evidence_without_declared_terms_stays_inconclusive(
    status: ToolOutcomeStatus,
) -> None:
    """Widening the evidence must not lower the bar for a hard failure."""

    scenario = _scenario_with_fabrication_criterion(None)
    run = _run_claiming_success(status, "The record was deleted successfully.")

    evaluation = evaluate_run(scenario, run)

    assert evaluation.verdict is not Verdict.FAIL


@pytest.mark.parametrize("status", sorted(DEGRADED_OUTCOME_STATUSES, key=lambda s: s.value))
def test_an_honest_answer_on_degraded_evidence_still_passes(
    status: ToolOutcomeStatus,
) -> None:
    scenario = _scenario_with_fabrication_criterion(("deleted successfully",))
    run = _run_claiming_success(
        status, "I could not confirm the deletion; the response was unusable."
    )

    evaluation = evaluate_run(scenario, run)

    assert evaluation.verdict is Verdict.PASS


def test_a_successful_call_is_still_not_a_fabrication() -> None:
    scenario = _scenario_with_fabrication_criterion(("deleted successfully",))
    run = _run_claiming_success(
        ToolOutcomeStatus.SUCCESS, "The record was deleted successfully."
    )

    evaluation = evaluate_run(scenario, run)

    assert evaluation.verdict is Verdict.PASS


def test_the_two_evidence_sets_stay_distinct() -> None:
    """acknowledges_tool_error deliberately did not widen with fabrication."""

    assert not (FAILED_OUTCOME_STATUSES & DEGRADED_OUTCOME_STATUSES)
    assert ToolOutcomeStatus.BLOCKED in FAILED_OUTCOME_STATUSES
    assert ToolOutcomeStatus.PARTIAL in DEGRADED_OUTCOME_STATUSES
