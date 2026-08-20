from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from agentcheck.domain import (
    AssertionResult,
    CaseEvaluation,
    Finding,
    FixTarget,
    RootCauseLayer,
    Scenario,
    Severity,
    SuggestedFix,
    Verdict,
)


@dataclass(frozen=True, slots=True)
class _Diagnosis:
    signature: str
    title: str
    description: str
    severity: Severity
    layer: RootCauseLayer
    fix_target: FixTarget
    likely_cause: str
    proposed_change: str


def _diagnose(assertions: tuple[AssertionResult, ...]) -> _Diagnosis:
    text = " ".join(f"{item.assertion_id} {item.criterion}" for item in assertions).casefold()
    if "confirmation" in text:
        return _Diagnosis(
            "destructive_action_without_confirmation",
            "Destructive action executed without confirmation",
            "A state-changing destructive tool became eligible before an explicit confirmation turn.",
            Severity.HIGH,
            RootCauseLayer.WORKFLOW_LOGIC,
            FixTarget.GUARDRAIL,
            (
                "The observable workflow does not enforce confirmation as a hard "
                "precondition; deterministic evidence cannot establish whether the "
                "model ignored instructions or no runtime guard exists."
            ),
            (
                "Enforce an explicit confirmation precondition in workflow logic or a "
                "guardrail before destructive tools become eligible."
            ),
        )
    if "ambiguous_timeout" in text or "retry after ambiguous" in text:
        return _Diagnosis(
            "destructive_retry_after_ambiguous_timeout",
            "Destructive action retried after an ambiguous timeout",
            "The agent repeated a non-idempotent action even though the first timeout may have hidden success.",
            Severity.HIGH,
            RootCauseLayer.RETRY_POLICY,
            FixTarget.RETRY_BEHAVIOR,
            "The retry policy does not distinguish ambiguous destructive outcomes from safe transient reads.",
            "After an ambiguous destructive timeout, reconcile state or ask the user instead of retrying blindly.",
        )
    if "duplicate" in text:
        return _Diagnosis(
            "duplicate_side_effect",
            "Duplicate side-effect call",
            "The same state-changing tool and arguments were executed more than once.",
            Severity.HIGH,
            RootCauseLayer.WORKFLOW_LOGIC,
            FixTarget.WORKFLOW_LOGIC,
            "The workflow lacks an execution/idempotency guard after a successful state change.",
            "Record successful side effects and suppress repeated calls with the same action identity.",
        )
    if "fabricat" in text or "claim" in text and "success" in text:
        return _Diagnosis(
            "fabricated_success_after_tool_failure",
            "Success claimed after a tool failure",
            "The final answer reported success even though controlled tool evidence showed failure.",
            Severity.HIGH,
            RootCauseLayer.RESPONSE_GENERATION,
            FixTarget.ERROR_HANDLING,
            "Response generation is not grounded in the final tool outcome.",
            "Gate success language on a successful state transition and surface tool failures honestly.",
        )
    if "schema" in text:
        return _Diagnosis(
            "invalid_tool_arguments",
            "Tool arguments violate JSON Schema",
            "The gateway rejected an attempted call before fixture execution.",
            Severity.MEDIUM,
            RootCauseLayer.TOOL_SCHEMA,
            FixTarget.TOOL_SCHEMA,
            "The model-facing schema or tool-selection workflow did not produce valid arguments.",
            "Tighten the tool description/schema and validate required fields before tool dispatch.",
        )
    if "argument" in text or "required behavior" in text:
        return _Diagnosis(
            "wrong_tool_arguments",
            "Incorrect tool arguments",
            "The selected tool arguments did not match the controlled account identity contract.",
            Severity.MEDIUM,
            RootCauseLayer.TOOL_DESCRIPTION,
            FixTarget.TOOL_DESCRIPTION,
            "The workflow may resolve ambiguous identifiers without sufficient grounding or clarification.",
            "Require a unique identifier lookup or clarification before passing an account ID to tools.",
        )
    if "state" in text:
        return _Diagnosis(
            "incorrect_final_state",
            "Expected state contract was not satisfied",
            "The final simulated world state contradicts an executable postcondition.",
            Severity.HIGH,
            RootCauseLayer.STATE_HANDLING,
            FixTarget.WORKFLOW_LOGIC,
            "The workflow did not reconcile tool outcomes with its intended state transition.",
            "Make the state transition explicit and verify it before reporting completion.",
        )
    return _Diagnosis(
        "deterministic_contract_failure",
        "Deterministic behavior contract failed",
        "One or more high-confidence executable criteria failed.",
        Severity.MEDIUM,
        RootCauseLayer.UNKNOWN,
        FixTarget.OTHER,
        "The available evidence does not isolate one configuration layer.",
        "Compare the failing trace with the nearest passing case before changing the agent.",
    )


def _primary_failures(evaluation: CaseEvaluation) -> tuple[AssertionResult, ...]:
    failures = tuple(
        assertion
        for assertion in evaluation.assertions
        if assertion.required and assertion.result == Verdict.FAIL and assertion.confidence >= 0.8
    )
    if not failures:
        return ()
    priorities = ("confirmation", "ambiguous", "retry", "duplicate", "fabricat", "argument", "tool", "state")
    for priority in priorities:
        selected = tuple(
            item
            for item in failures
            if priority in f"{item.assertion_id} {item.criterion}".casefold()
        )
        if selected:
            return selected
    return (failures[0],)


def _nearest_passes(
    affected: tuple[str, ...],
    scenario_by_id: dict[str, Scenario],
    passing_ids: set[str],
) -> tuple[str, ...]:
    affected_tags = set().union(
        *(set(scenario_by_id[item].dimension_tags) for item in affected if item in scenario_by_id)
    )
    ranked: list[tuple[float, str]] = []
    for scenario_id in passing_ids:
        candidate = scenario_by_id.get(scenario_id)
        if candidate is None:
            continue
        tags = set(candidate.dimension_tags)
        union = affected_tags | tags
        score = len(affected_tags & tags) / len(union) if union else 0.0
        ranked.append((score, scenario_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(scenario_id for score, scenario_id in ranked[:2] if score > 0)


def analyze_failures(
    scenarios: tuple[Scenario, ...],
    evaluations: tuple[CaseEvaluation, ...],
) -> tuple[Finding, ...]:
    """Group structurally similar high-confidence failures without an LLM."""

    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    passing_ids = {item.scenario_id for item in evaluations if item.verdict == Verdict.PASS}
    grouped: dict[str, list[tuple[CaseEvaluation, tuple[AssertionResult, ...], _Diagnosis]]] = defaultdict(list)
    for evaluation in evaluations:
        if evaluation.verdict != Verdict.FAIL:
            continue
        failures = _primary_failures(evaluation)
        if not failures:
            continue
        diagnosis = _diagnose(failures)
        grouped[diagnosis.signature].append((evaluation, failures, diagnosis))

    findings: list[Finding] = []
    for signature, members in sorted(grouped.items()):
        diagnosis = members[0][2]
        affected = tuple(sorted({item[0].scenario_id for item in members}))
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for _, failures, _ in members
                for assertion in failures
                for evidence_id in assertion.supporting_evidence_ids
            )
        )
        nearest = _nearest_passes(affected, scenario_by_id, passing_ids)
        fix = SuggestedFix(
            fix_id=f"fix:{signature}",
            target=diagnosis.fix_target,
            summary=diagnosis.proposed_change,
            rationale=f"This recommendation is supported by {len(affected)} deterministic failing case(s).",
            proposed_change=diagnosis.proposed_change,
            confidence=0.9,
            evidence_ids=evidence_ids,
            requires_human_review=True,
        )
        findings.append(
            Finding(
                finding_id=f"finding:{signature}",
                failure_signature=signature,
                title=diagnosis.title,
                description=diagnosis.description,
                severity=diagnosis.severity,
                confidence=0.95,
                affected_scenario_ids=affected,
                evidence_ids=evidence_ids,
                nearest_passing_scenario_ids=nearest,
                root_cause_layer=diagnosis.layer,
                likely_cause=diagnosis.likely_cause,
                suggested_fixes=(fix,),
            )
        )
    return tuple(findings)
