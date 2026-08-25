"""A run that attempted no tool observed nothing, and must not be scored FAIL.

`zero-input` cases seed a user turn asking for the call and attach a
`required_tool_behavior`. ControlledModel -- AgentCheck's own neutral offline
model -- is documented to "never choose a tool", so every zero-argument tool on
every target produced a guaranteed FAIL under the documented offline
configuration, and `gate` returned exit 1 on it.

Proven with a controlled experiment against the same suite and the same tool:
under ControlledModel the case FAILs and action-path coverage is 0/8; under a
scripted model that calls the tool once it PASSes, 8/8, coverage 7/7. The verdict
was tracking the harness, not the agent.

So a required-call constraint is now undecided rather than violated when the run
attempted no tool at all. These tests hold that line from both sides: the
silence must stop being a FAIL, and every observed choice must keep being one.
"""

from __future__ import annotations

from agentcheck.domain import (
    CanonicalEvent,
    CanonicalEventType,
    CanonicalRun,
    ConversationRole,
    ConversationTurn,
    OracleProvenance,
    OracleStrength,
    RunTermination,
    Scenario,
    ToolAttempt,
    ToolBehaviorConstraint,
    UsageMetrics,
    Verdict,
    utc_now,
)
from agentcheck.evaluate import evaluate_run

NOW = utc_now()


def _scenario() -> Scenario:
    return Scenario(
        scenario_id="zero-input-cancel-trip",
        title="cancel_trip may be invoked with no arguments",
        description="Exercises the in-contract zero-input invocation.",
        conversation_turns=(
            ConversationTurn(
                turn_id="turn-1",
                role=ConversationRole.USER,
                content="Use the cancel_trip tool with no arguments: {}",
            ),
        ),
        required_tool_behavior=(
            ToolBehaviorConstraint(
                criterion_id="zero-input-cancel-trip:required",
                tool_name="cancel_trip",
                arguments_match={},
                min_calls=1,
                oracle_ids=("zero-input-cancel-trip:oracle",),
            ),
        ),
        dimension_tags=("source:zero_input_invocation", "tool:cancel_trip"),
        oracle_provenance=(
            OracleProvenance(
                oracle_id="zero-input-cancel-trip:oracle",
                strength=OracleStrength.CONTROLLED_WORLD_FACT,
                source="declared schema accepts the empty object",
                confidence=1.0,
                evidence_ids=("zero-input-cancel-trip:required",),
                supports_hard_failure=True,
            ),
        ),
        generation_seed=1729,
    )


def _run(*attempts: ToolAttempt) -> CanonicalRun:
    events = [
        CanonicalEvent(
            event_id="event-1",
            run_id="run-1",
            sequence=1,
            event_type=CanonicalEventType.USER_TURN,
            timestamp=NOW,
            payload={"text": "Use the cancel_trip tool with no arguments: {}"},
        )
    ]
    for index, attempt in enumerate(attempts, start=2):
        events.append(
            CanonicalEvent(
                event_id=f"event-{index}",
                run_id="run-1",
                sequence=index,
                event_type=CanonicalEventType.TOOL_ATTEMPT,
                timestamp=NOW,
                payload={"tool_name": attempt.tool_name},
            )
        )
    return CanonicalRun(
        run_id="run-1",
        scenario_id="zero-input-cancel-trip",
        target_id="target-1",
        started_at=NOW,
        ended_at=NOW,
        events=tuple(events),
        tool_attempts=attempts,
        tool_outcomes=(),
        termination=RunTermination.COMPLETED,
        final_output="acknowledged",
        usage=UsageMetrics(),
    )


def _attempt(tool_name: str, arguments: dict, index: int = 2) -> ToolAttempt:
    return ToolAttempt(
        attempt_id=f"attempt-{index}",
        event_id=f"event-{index}",
        tool_name=tool_name,
        arguments=arguments,
        sequence=index,
        timestamp=NOW,
    )


# --- the false positive -----------------------------------------------------


def test_a_run_that_attempted_nothing_is_undecided_not_failed() -> None:
    """The regression. This was FAIL, blaming the agent for the harness."""
    result = evaluate_run(_scenario(), _run())

    assert result.verdict is Verdict.INCONCLUSIVE


def test_the_undecided_case_says_why() -> None:
    result = evaluate_run(_scenario(), _run())

    rationales = " ".join(a.rationale for a in result.assertions)
    assert "No tool was attempted at all" in rationales
    missing = [m for a in result.assertions for m in a.missing_evidence]
    assert missing, "an inconclusive verdict must name the evidence it lacks"


# --- everything the run DID observe must still fail -------------------------


def test_calling_the_right_tool_with_wrong_arguments_still_fails() -> None:
    """An out-of-contract call is an observed choice, not silence.

    Uses a contract that actually constrains arguments. An empty
    ``arguments_match`` means "no argument constraint" at this layer -- for a
    zero-parameter schema the gateway is what refuses undeclared properties.
    """
    scenario = _scenario().model_copy(
        update={
            "required_tool_behavior": (
                ToolBehaviorConstraint(
                    criterion_id="zero-input-cancel-trip:required",
                    tool_name="cancel_trip",
                    arguments_match={"order_id": "A-1"},
                    min_calls=1,
                    oracle_ids=("zero-input-cancel-trip:oracle",),
                ),
            )
        }
    )

    result = evaluate_run(scenario, _run(_attempt("cancel_trip", {"order_id": "B-2"})))

    assert result.verdict is Verdict.FAIL


def test_calling_some_other_tool_instead_still_fails() -> None:
    """The agent chose. That is evidence, and the contract is unmet."""
    result = evaluate_run(_scenario(), _run(_attempt("change_seat", {"seat": "1A"})))

    assert result.verdict is Verdict.FAIL


def test_the_satisfied_contract_still_passes() -> None:
    result = evaluate_run(_scenario(), _run(_attempt("cancel_trip", {})))

    assert result.verdict is Verdict.PASS


def test_inconclusive_is_not_a_pass() -> None:
    """`gate` must still refuse the run. This is a relabel, not a softening."""
    from agentcheck.gate import EXIT_INCONCLUSIVE, EXIT_PASS

    result = evaluate_run(_scenario(), _run())

    assert result.verdict is not Verdict.PASS
    assert EXIT_INCONCLUSIVE != EXIT_PASS
