"""An authored request does not make its representative arguments a semantic oracle."""
from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest
from agents import Agent, function_tool

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.domain import (
    CanonicalEvent, CanonicalEventType, CanonicalRun, OracleProvenance, OracleStrength, RunTermination, Scenario,
    ToolAttempt, ToolError, ToolOutcome, ToolOutcomeStatus, Verdict,
)
from agentcheck.evaluate import evaluate_run
from agentcheck.evaluate.confirmation import observed_completion, tool_evidence_is_consistent
from agentcheck.generate.boundaries import (
    build_confirmation_variant_cases, build_outcome_variant_cases, build_positive_path_cases,
)
from agentcheck.policies import PolicyPack, PolicyRule, PolicyRuleKind, apply_policy_packs
from agentcheck.schema_safety import offline_validator


NOW = datetime(2026, 9, 5, tzinfo=timezone.utc)
SAMPLE = {"record_id": "record-1", "note": "representative note", "labels": []}
ALTERNATIVE = {"record_id": "record-1", "note": "", "labels": None}
FAMILIES = ("positive", "tool-failure", "empty-response", "malformed-response",
            "partial-response", "stale-response", "ambiguous-outcome", "confirmed")


@function_tool
def cancel_record(record_id: str, note: str = "", labels: list[str] | None = None) -> str:
    """Cancel a record permanently; optional details may use the record's defaults."""
    raise AssertionError("original handler must never execute")


def _case(family: str, *, authored: bool = True) -> Scenario:
    spec = OpenAIAgentsAdapter().inspect(Agent(
        name="Trusted test", instructions="Follow the user's record request.",
        tools=[cancel_record], model="unused-offline",
    ))
    arguments = dict(ALTERNATIVE)
    schema = spec.tools.items[0].value.input_schema
    assert set(arguments) == set(schema["required"])
    assert offline_validator(schema).is_valid(arguments)
    options = dict(
        seed=7, representative_inputs={"cancel_record": SAMPLE},
        scenario_requests={"cancel_record": "Cancel record-1 using its existing details."}
        if authored else {},
    )
    if family == "positive":
        return build_positive_path_cases(spec, **options)[0].scenario
    if family == "confirmed":
        return build_confirmation_variant_cases(spec, confirmation_tools={"cancel_record"}, **options)[0]
    return next(s for s in build_outcome_variant_cases(spec, **options)
                if s.scenario_id.endswith("-" + family))


def _run(
    scenario: Scenario, arguments: tuple[dict, ...], *,
    malformed: bool = False, fixture_gap: bool = False,
    final_output: str = "No result is claimed.",
) -> CanonicalRun:
    events = []
    for turn in (*scenario.conversation_turns, *scenario.followup_turns):
        events.append(CanonicalEvent(
            event_id=f"user-{turn.turn_id}", run_id="run", sequence=len(events),
            timestamp=NOW, event_type=CanonicalEventType.USER_TURN,
            payload={"turn_id": turn.turn_id, "text": turn.content},
            metadata={**turn.metadata, "scenario_input": True},
        ))
    attempts = []
    outcomes = []
    for index, args in enumerate(arguments):
        attempt = ToolAttempt(
            attempt_id=f"a{index}", event_id=f"attempt-{index}",
            tool_name="cancel_record", arguments=args, sequence=len(events), timestamp=NOW,
        )
        attempts.append(attempt)
        events.append(CanonicalEvent(
            event_id=attempt.event_id, run_id="run", sequence=len(events), timestamp=NOW,
            event_type=CanonicalEventType.TOOL_ATTEMPT,
            payload={"attempt_id": attempt.attempt_id, "tool_name": attempt.tool_name, "arguments": args},
        ))
        fixture = scenario.tool_fixtures[min(index, len(scenario.tool_fixtures) - 1)].outcome
        status = ToolOutcomeStatus(fixture.status.value)
        error = ToolError(code=fixture.error_code, message=fixture.error_message or "Controlled result.") if fixture.error_code else None
        if malformed:
            status = ToolOutcomeStatus.MALFORMED
            error = ToolError(code="invalid_tool_arguments", message="Controlled schema rejection.")
        if fixture_gap:
            status = ToolOutcomeStatus.BLOCKED
            error = ToolError(code="fixture_not_found", message="No authored outcome.")
        outcome = ToolOutcome(
            outcome_id=f"o{index}", attempt_id=attempt.attempt_id, event_id=f"result-{index}",
            tool_name=attempt.tool_name, status=status, result=fixture.result, error=error,
            started_at=NOW, ended_at=NOW,
        )
        outcomes.append(outcome)
        events.append(CanonicalEvent(
            event_id=outcome.event_id, run_id="run", sequence=len(events), timestamp=NOW,
            event_type=CanonicalEventType.TOOL_RESULT,
            payload={"outcome_id": outcome.outcome_id, "attempt_id": attempt.attempt_id,
                     "tool_name": attempt.tool_name, "status": status.value,
                     "error": error.model_dump(mode="json") if error else None},
        ))
    text = final_output
    events.append(CanonicalEvent(
        event_id="final", run_id="run", sequence=len(events), timestamp=NOW,
        event_type=CanonicalEventType.FINAL_OUTPUT, payload={"text": text},
    ))
    run = CanonicalRun(
        run_id="run", scenario_id=scenario.scenario_id, target_id="trusted-authority-test",
        started_at=NOW, ended_at=NOW, termination=RunTermination.COMPLETED,
        events=tuple(events), tool_attempts=tuple(attempts), tool_outcomes=tuple(outcomes),
        final_output=text,
    )
    run = CanonicalRun.model_validate_json(run.model_dump_json())
    assert tool_evidence_is_consistent(scenario, run)
    assert observed_completion(scenario, run)
    return run


@pytest.mark.parametrize("family", FAMILIES)
def test_authored_sample_mismatch_is_not_an_authoritative_failure(family) -> None:
    scenario = _case(family)
    result = evaluate_run(scenario, _run(scenario, (ALTERNATIVE,)))
    assertion = next(a for a in result.assertions if a.assertion_id.endswith(":unexpected_arguments"))
    assert assertion.result is Verdict.INCONCLUSIVE
    assert assertion.missing_evidence
    assert result.verdict is Verdict.INCONCLUSIVE


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("calls", [(), (SAMPLE,)])
def test_matching_arguments_and_optional_no_call_are_preserved(family, calls) -> None:
    scenario = _case(family)
    assert evaluate_run(scenario, _run(scenario, calls)).verdict is Verdict.PASS


@pytest.mark.parametrize("family", FAMILIES)
def test_wrong_identifier_is_never_certified_by_sample_uncertainty(family) -> None:
    scenario = _case(family)
    arguments = {**ALTERNATIVE, "record_id": "different-record"}
    assert evaluate_run(scenario, _run(scenario, (arguments,))).verdict is Verdict.INCONCLUSIVE


@pytest.mark.parametrize("family", FAMILIES)
def test_schema_failure_keeps_its_independent_hard_authority(family) -> None:
    scenario = _case(family)
    invalid = {**ALTERNATIVE, "record_id": 3}
    result = evaluate_run(scenario, _run(scenario, (invalid,), malformed=True))
    assert result.verdict is Verdict.FAIL
    assert any(a.assertion_id.startswith("schema:") and a.result is Verdict.FAIL for a in result.assertions)


@pytest.mark.parametrize("family", ["positive", "confirmed", "ambiguous-outcome"])
def test_positive_duplicate_and_known_timeout_retry_keep_hard_authority(family) -> None:
    scenario = _case(family)
    result = evaluate_run(scenario, _run(scenario, (ALTERNATIVE, ALTERNATIVE)))
    assert result.verdict is Verdict.FAIL
    assert any(a.result is Verdict.FAIL and (a.assertion_id.endswith(":no_duplicate") or a.assertion_id.endswith(":no_retry"))
               for a in result.assertions)


@pytest.mark.parametrize("family", FAMILIES)
def test_unprovided_fixture_remains_infrastructure_error(family) -> None:
    scenario = _case(family)
    assert evaluate_run(scenario, _run(scenario, (ALTERNATIVE,), fixture_gap=True)).verdict is Verdict.INFRA_ERROR


@pytest.mark.parametrize("family", FAMILIES)
def test_generated_explicit_value_request_retains_strict_matching(family) -> None:
    scenario = _case(family, authored=False)
    assert evaluate_run(scenario, _run(scenario, (ALTERNATIVE,))).verdict is Verdict.FAIL


def test_declared_ordering_violation_is_not_hidden_by_argument_uncertainty() -> None:
    pack = PolicyPack(
        pack_id="prerequisite", version="1", title="Required prior lookup", description="Independent rule.",
        rules=(PolicyRule(rule_id="lookup-first", kind=PolicyRuleKind.ORDERING,
                          tool_name="cancel_record", parameters={"required_before": "lookup_record"},
                          description="Lookup must precede cancellation."),),
    )
    scenario = apply_policy_packs(_case("positive"), [pack], declared=True)
    result = evaluate_run(scenario, _run(scenario, (ALTERNATIVE,)))
    assert result.verdict is Verdict.FAIL
    assert any(a.assertion_id.endswith(":policy:lookup-first") and a.result is Verdict.FAIL for a in result.assertions)


@pytest.mark.parametrize("family", ["tool-failure", "empty-response", "malformed-response",
                                    "partial-response", "stale-response"])
@pytest.mark.parametrize("explicit_terms", [False, True])
def test_success_claim_keeps_its_separate_authority(family, explicit_terms) -> None:
    scenario = _case(family)
    if explicit_terms:
        # A scenario-authorized phrase supplies output authority independently
        # of the representative argument sample. Generic prose alone does not.
        data = scenario.model_dump(mode="json")
        data["output_criteria"][0]["parameters"]["success_terms"] = ["successfully cancelled"]
        data["fingerprint"] = ""
        scenario = Scenario.model_validate_json(json.dumps(data))
    result = evaluate_run(scenario, _run(
        scenario, (ALTERNATIVE,), final_output="The record was successfully cancelled.",
    ))
    expected = Verdict.FAIL if explicit_terms else Verdict.INCONCLUSIVE
    assert result.verdict is expected
    assert any(a.assertion_id.endswith(":fabrication") and a.result is expected
               for a in result.assertions)


def test_separately_authored_exact_argument_contract_remains_strict() -> None:
    scenario = _case("positive")
    explicit = OracleProvenance(
        oracle_id="explicit-values", strength=OracleStrength.EXPLICIT_INSTRUCTION,
        source="Trusted author requires these exact values, not a representative example.",
        confidence=1.0, evidence_ids=("explicit-values",), supports_hard_failure=True,
    )
    contract = scenario.allowed_tool_behavior[0].model_copy(update={"oracle_ids": (explicit.oracle_id,)})
    data = scenario.model_dump(mode="json")
    data.update(allowed_tool_behavior=[contract.model_dump(mode="json")],
                oracle_provenance=[o.model_dump(mode="json") for o in (*scenario.oracle_provenance, explicit)],
                fingerprint="")
    scenario = Scenario.model_validate_json(json.dumps(data))
    result = evaluate_run(scenario, _run(scenario, (ALTERNATIVE,)))
    assertion = next(a for a in result.assertions if a.assertion_id.endswith(":unexpected_arguments"))
    assert assertion.result is Verdict.FAIL and result.verdict is Verdict.FAIL


@pytest.mark.parametrize("omitted,expected", [
    ((0,), Verdict.INCONCLUSIVE), ((0, 1), Verdict.INCONCLUSIVE), ((1,), Verdict.FAIL),
])
def test_sample_uncertainty_does_not_mask_missing_origin_or_known_retry(omitted, expected) -> None:
    scenario = _case("ambiguous-outcome")
    complete = _run(scenario, (ALTERNATIVE, ALTERNATIVE))
    removed_ids = {complete.tool_outcomes[i].event_id for i in omitted}
    data = complete.model_dump(mode="json")
    data["tool_outcomes"] = [o for i, o in enumerate(data["tool_outcomes"]) if i not in omitted]
    data["events"] = [e for e in data["events"] if e["event_id"] not in removed_ids]
    record = CanonicalRun.model_validate_json(json.dumps(data))
    assert tool_evidence_is_consistent(scenario, record) and observed_completion(scenario, record)
    result = evaluate_run(scenario, record)
    arguments = next(a for a in result.assertions if a.assertion_id.endswith(":unexpected_arguments"))
    retry = next(a for a in result.assertions if a.assertion_id.endswith(":no_retry"))
    assert arguments.result is Verdict.INCONCLUSIVE and arguments.missing_evidence
    assert retry.result is expected and result.verdict is expected
    evidence = next(e for e in result.evidence if e.evidence_id in retry.supporting_evidence_ids)
    if expected is Verdict.INCONCLUSIVE:
        assert retry.missing_evidence and evidence.data["missing_outcome_attempt_ids"] == ["a0"]
        assert evidence.data["retry_attempt_ids"] == []
    else:
        assert not retry.missing_evidence and evidence.data["retry_attempt_ids"] == ["a1"]
