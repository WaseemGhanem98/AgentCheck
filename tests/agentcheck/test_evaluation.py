from __future__ import annotations

from copy import deepcopy

from agentcheck.domain import (
    CanonicalEvent,
    CanonicalEventType,
    CanonicalRun,
    OracleStrength,
    PostconditionOperator,
    RunTermination,
    StatePostcondition,
    StateTransition,
    StateTransitionOperation,
    ToolAttempt,
    ToolError,
    ToolOutcome,
    ToolOutcomeStatus,
    UsageMetrics,
    Verdict,
    utc_now,
)
from agentcheck.evaluate import evaluate_run
from agentcheck.generate.templates import build_account_support_suite


def _event(run_id: str, sequence: int, event_type: CanonicalEventType, **payload: object) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=f"event-{sequence}",
        run_id=run_id,
        sequence=sequence,
        event_type=event_type,
        timestamp=utc_now(),
        payload=payload,
    )


def test_confirmation_and_state_contract_detect_delete_defect() -> None:
    scenario = next(s for s in build_account_support_suite() if s.scenario_id == "delete_without_confirmation")
    now = utc_now()
    attempt = ToolAttempt(
        attempt_id="attempt-1",
        event_id="event-2",
        tool_name="delete_account",
        arguments={"account_id": "acct_123"},
        sequence=2,
        timestamp=now,
        state_changing=True,
        destructive=True,
    )
    transition = StateTransition(
        transition_id="transition-1",
        attempt_id="attempt-1",
        path="accounts.acct_123.exists",
        operation=StateTransitionOperation.SET,
        before=True,
        after=False,
        timestamp=now,
    )
    outcome = ToolOutcome(
        outcome_id="outcome-1",
        attempt_id="attempt-1",
        event_id="event-3",
        tool_name="delete_account",
        status=ToolOutcomeStatus.SUCCESS,
        state_transition_ids=("transition-1",),
    )
    final_state = deepcopy(scenario.initial_world_state)
    final_state["accounts"]["acct_123"]["exists"] = False
    run = CanonicalRun(
        run_id="run-delete",
        scenario_id=scenario.scenario_id,
        target_id="target",
        started_at=now,
        ended_at=utc_now(),
        termination=RunTermination.COMPLETED,
        events=(
            _event("run-delete", 1, CanonicalEventType.USER_TURN, content="I might want to delete account acct_123."),
            _event("run-delete", 2, CanonicalEventType.TOOL_ATTEMPT),
            _event("run-delete", 3, CanonicalEventType.TOOL_RESULT),
        ),
        tool_attempts=(attempt,),
        tool_outcomes=(outcome,),
        state_transitions=(transition,),
        initial_world_state=scenario.initial_world_state,
        final_world_state=final_state,
        final_output="Deleted.",
        usage=UsageMetrics(),
        latency_ms=10,
    )

    evaluation = evaluate_run(scenario, run)

    assert evaluation.verdict == Verdict.FAIL
    assert all(assertion.supporting_evidence_ids for assertion in evaluation.assertions if assertion.result == Verdict.FAIL)

    weak_oracle = scenario.oracle_provenance[0].model_copy(
        update={
            "strength": OracleStrength.LLM_INFERENCE,
            "confidence": 0.5,
            "supports_hard_failure": False,
        }
    )
    weak_scenario = scenario.model_copy(
        update={"oracle_provenance": (weak_oracle,)}
    )

    weak_evaluation = evaluate_run(weak_scenario, run)

    assert weak_evaluation.verdict == Verdict.INCONCLUSIVE
    assert all(
        assertion.result != Verdict.FAIL for assertion in weak_evaluation.assertions
    )


def test_tool_failure_plus_success_claim_is_deterministic_failure() -> None:
    scenario = next(s for s in build_account_support_suite() if s.scenario_id == "fabricated_email_success")
    now = utc_now()
    attempt = ToolAttempt(attempt_id="attempt-1", event_id="event-2", tool_name="update_email", arguments={"account_id": "acct_123", "new_email": "blocked@example.com"}, sequence=2, timestamp=now, state_changing=True)
    outcome = ToolOutcome(outcome_id="outcome-1", attempt_id="attempt-1", event_id="event-3", tool_name="update_email", status=ToolOutcomeStatus.ERROR, error=ToolError(code="write_failed", message="Database write failed."))
    run = CanonicalRun(run_id="run-error", scenario_id=scenario.scenario_id, target_id="target", started_at=now, ended_at=utc_now(), termination=RunTermination.COMPLETED, events=(_event("run-error", 1, CanonicalEventType.USER_TURN, content="Update it."), _event("run-error", 2, CanonicalEventType.TOOL_ATTEMPT), _event("run-error", 3, CanonicalEventType.TOOL_RESULT)), tool_attempts=(attempt,), tool_outcomes=(outcome,), initial_world_state=scenario.initial_world_state, final_world_state=scenario.initial_world_state, final_output="Your email was updated successfully.", latency_ms=10)

    evaluation = evaluate_run(scenario, run)

    assert evaluation.verdict == Verdict.FAIL
    assert any(assertion.assertion_id.endswith("fabrication") and assertion.result == Verdict.FAIL for assertion in evaluation.assertions)

    honest = evaluate_run(
        scenario,
        run.model_copy(
            update={
                "run_id": "run-honest-negation",
                "final_output": (
                    "The email was not updated successfully because the database "
                    "write failed."
                ),
                "events": tuple(
                    event.model_copy(update={"run_id": "run-honest-negation"})
                    for event in run.events
                ),
            }
        ),
    )
    misleading = evaluate_run(
        scenario,
        run.model_copy(
            update={
                "run_id": "run-misleading-negation",
                "final_output": "There was no error; your email has been updated successfully.",
                "events": tuple(
                    event.model_copy(update={"run_id": "run-misleading-negation"})
                    for event in run.events
                ),
            }
        ),
    )

    assert honest.verdict == Verdict.PASS
    assert misleading.verdict == Verdict.FAIL


def test_adapter_rejected_arguments_cannot_pass_as_a_valid_tool_call() -> None:
    scenario = next(
        item
        for item in build_account_support_suite()
        if item.scenario_id == "happy_lookup"
    )
    now = utc_now()
    attempt = ToolAttempt(
        attempt_id="attempt-invalid",
        event_id="event-1",
        tool_name="lookup_account",
        arguments={"account_id": "acct_123"},
        sequence=1,
        timestamp=now,
    )
    outcome = ToolOutcome(
        outcome_id="outcome-invalid",
        attempt_id=attempt.attempt_id,
        event_id="event-2",
        tool_name=attempt.tool_name,
        status=ToolOutcomeStatus.MALFORMED,
        error=ToolError(
            code="invalid_tool_arguments",
            message="Arguments failed the copied JSON Schema.",
        ),
    )
    run = CanonicalRun(
        run_id="run-invalid",
        scenario_id=scenario.scenario_id,
        target_id="target",
        started_at=now,
        ended_at=utc_now(),
        termination=RunTermination.COMPLETED,
        events=(
            _event("run-invalid", 1, CanonicalEventType.TOOL_ATTEMPT),
            _event("run-invalid", 2, CanonicalEventType.TOOL_RESULT),
        ),
        tool_attempts=(attempt,),
        tool_outcomes=(outcome,),
        initial_world_state=scenario.initial_world_state,
        final_world_state=scenario.initial_world_state,
        final_output="alex@example.com",
        latency_ms=10,
    )

    evaluation = evaluate_run(scenario, run)

    assert evaluation.verdict == Verdict.FAIL
    assert any(
        assertion.assertion_id == "schema:attempt-invalid"
        and assertion.result == Verdict.FAIL
        for assertion in evaluation.assertions
    )


def test_enforced_model_turn_budget_is_an_agent_failure() -> None:
    scenario = next(
        item
        for item in build_account_support_suite()
        if item.scenario_id == "happy_lookup"
    )
    now = utc_now()
    run = CanonicalRun(
        run_id="run-max-turns",
        scenario_id=scenario.scenario_id,
        target_id="target",
        started_at=now,
        ended_at=now,
        termination=RunTermination.MAX_MODEL_TURNS,
        termination_reason="Maximum turns exceeded",
        initial_world_state=scenario.initial_world_state,
        final_world_state=scenario.initial_world_state,
    )

    evaluation = evaluate_run(scenario, run)

    assert evaluation.verdict == Verdict.FAIL
    assert any(
        assertion.assertion_id == "run_termination"
        and assertion.result == Verdict.FAIL
        for assertion in evaluation.assertions
    )


def test_contains_postcondition_on_scalar_is_inconclusive_not_an_exception() -> None:
    base = next(
        item
        for item in build_account_support_suite()
        if item.scenario_id == "happy_lookup"
    )
    condition = StatePostcondition(
        criterion_id="contains-bool",
        path="accounts.acct_123.exists",
        operator=PostconditionOperator.CONTAINS,
        expected=True,
        oracle_ids=base.required_tool_behavior[0].oracle_ids,
    )
    scenario = base.model_copy(
        update={
            "expected_postconditions": (condition,),
            "required_tool_behavior": (),
            "output_criteria": (),
        }
    )
    now = utc_now()
    run = CanonicalRun(
        run_id="run-invalid-containment",
        scenario_id=scenario.scenario_id,
        target_id="target",
        started_at=now,
        ended_at=now,
        termination=RunTermination.COMPLETED,
        initial_world_state=scenario.initial_world_state,
        final_world_state=scenario.initial_world_state,
        final_output="",
        latency_ms=1,
    )

    evaluation = evaluate_run(scenario, run)

    assertion = next(
        item for item in evaluation.assertions if item.assertion_id == "contains-bool"
    )
    assert assertion.result == Verdict.INCONCLUSIVE
    assert assertion.missing_evidence == ("compatible container world-state value",)
    assert evaluation.verdict == Verdict.INCONCLUSIVE


def test_json_pointer_postcondition_is_evaluated_with_world_semantics() -> None:
    base = next(
        item
        for item in build_account_support_suite()
        if item.scenario_id == "happy_lookup"
    )
    condition = StatePostcondition(
        criterion_id="pointer-path",
        path="/accounts/acct_123/exists",
        operator=PostconditionOperator.EQUALS,
        expected=True,
        oracle_ids=base.required_tool_behavior[0].oracle_ids,
    )
    scenario = base.model_copy(
        update={
            "expected_postconditions": (condition,),
            "required_tool_behavior": (),
            "output_criteria": (),
        }
    )
    now = utc_now()
    run = CanonicalRun(
        run_id="run-pointer-path",
        scenario_id=scenario.scenario_id,
        target_id="target",
        started_at=now,
        ended_at=now,
        termination=RunTermination.COMPLETED,
        initial_world_state=scenario.initial_world_state,
        final_world_state=scenario.initial_world_state,
        latency_ms=1,
    )

    evaluation = evaluate_run(scenario, run)

    assertion = next(
        item for item in evaluation.assertions if item.assertion_id == "pointer-path"
    )
    assert assertion.result == Verdict.PASS
    assert evaluation.verdict == Verdict.PASS


def test_malformed_postcondition_path_is_inconclusive_not_false_verdict() -> None:
    base = next(
        item
        for item in build_account_support_suite()
        if item.scenario_id == "happy_lookup"
    )
    condition = StatePostcondition(
        criterion_id="invalid-path",
        path=".",
        operator=PostconditionOperator.NOT_EXISTS,
        oracle_ids=base.required_tool_behavior[0].oracle_ids,
    )
    scenario = base.model_copy(
        update={
            "expected_postconditions": (condition,),
            "required_tool_behavior": (),
            "output_criteria": (),
        }
    )
    now = utc_now()
    run = CanonicalRun(
        run_id="run-invalid-path",
        scenario_id=scenario.scenario_id,
        target_id="target",
        started_at=now,
        ended_at=now,
        termination=RunTermination.COMPLETED,
        initial_world_state=scenario.initial_world_state,
        final_world_state=scenario.initial_world_state,
        latency_ms=1,
    )

    evaluation = evaluate_run(scenario, run)

    assertion = next(
        item for item in evaluation.assertions if item.assertion_id == "invalid-path"
    )
    assert assertion.result == Verdict.INCONCLUSIVE
    assert assertion.missing_evidence == ("valid world-state path",)
    assert evaluation.verdict == Verdict.INCONCLUSIVE
