from __future__ import annotations

from agentcheck.analyze import analyze_failures
from agentcheck.domain import AssertionResult, CaseEvaluation, Evidence, EvidenceKind, Verdict, utc_now
from agentcheck.generate.templates import build_account_support_suite


def _evaluation(scenario_id: str, verdict: Verdict, assertion_id: str) -> CaseEvaluation:
    now = utc_now()
    evidence = Evidence(evidence_id=f"{scenario_id}:evidence", kind=EvidenceKind.TOOL_ATTEMPT, summary="Observed deterministic failure.", source_ids=("attempt-1",))
    assertion = AssertionResult(assertion_id=assertion_id, criterion=assertion_id, result=verdict, oracle_ids=(f"{scenario_id}:oracle",), supporting_evidence_ids=(evidence.evidence_id,) if verdict == Verdict.FAIL else (), rationale="Deterministic test assertion.")
    return CaseEvaluation(evaluation_id=f"eval-{scenario_id}", scenario_id=scenario_id, run_id=f"run-{scenario_id}", verdict=verdict, assertions=(assertion,), evidence=(evidence,), started_at=now, completed_at=now, summary=verdict.value)


def test_failure_analysis_groups_and_recommends_without_patching() -> None:
    suite = build_account_support_suite()
    evaluations = (
        _evaluation("delete_without_confirmation", Verdict.FAIL, "delete_without_confirmation:confirmation"),
        _evaluation("confirmed_delete", Verdict.PASS, "confirmed_delete:state"),
    )

    findings = analyze_failures(suite, evaluations)

    assert len(findings) == 1
    assert findings[0].failure_signature == "destructive_action_without_confirmation"
    assert findings[0].suggested_fixes[0].requires_human_review is True
    assert "confirmed_delete" in findings[0].nearest_passing_scenario_ids


def test_failure_analysis_attributes_schema_argument_failures_to_schema() -> None:
    suite = build_account_support_suite()
    evaluations = (
        _evaluation(
            "happy_lookup",
            Verdict.FAIL,
            "tool_contract:lookup_account:argument_schema",
        ),
    )

    findings = analyze_failures(suite, evaluations)

    assert len(findings) == 1
    assert findings[0].failure_signature == "invalid_tool_arguments"
    assert findings[0].root_cause_layer.value == "tool_schema"
    assert findings[0].suggested_fixes[0].target.value == "tool_schema"
