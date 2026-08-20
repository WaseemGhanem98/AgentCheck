"""Evidence-backed deterministic case evaluation contracts."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from .base import ContractModel, JsonObject, UtcDatetime
from .verdict import Verdict


CASE_EVALUATION_CONTRACT_VERSION: Literal["agentcheck.case_evaluation.v1"] = (
    "agentcheck.case_evaluation.v1"
)


class EvidenceKind(str, Enum):
    EVENT = "event"
    TOOL_ATTEMPT = "tool_attempt"
    TOOL_OUTCOME = "tool_outcome"
    WORLD_STATE = "world_state"
    STATE_TRANSITION = "state_transition"
    OUTPUT = "output"
    BUDGET = "budget"
    POLICY = "policy"
    AGENT_SPEC = "agent_spec"
    ERROR = "error"


class Evidence(ContractModel):
    evidence_id: str = Field(min_length=1, max_length=200)
    kind: EvidenceKind
    summary: str = Field(min_length=1, max_length=4_000)
    source_ids: tuple[str, ...] = Field(min_length=1)
    data: JsonObject = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    deterministic: bool = True
    sensitive: bool = False


class AssertionResult(ContractModel):
    assertion_id: str = Field(min_length=1, max_length=200)
    criterion: str = Field(min_length=1, max_length=4_000)
    result: Verdict
    required: bool = True
    oracle_ids: tuple[str, ...] = Field(min_length=1)
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    rationale: str = Field(min_length=1, max_length=8_000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    deterministic: bool = True

    @model_validator(mode="after")
    def require_failure_evidence(self) -> "AssertionResult":
        if self.result == Verdict.FAIL and not (
            self.supporting_evidence_ids or self.contradicting_evidence_ids
        ):
            raise ValueError("failed assertions require linked evidence")
        if self.result == Verdict.INCONCLUSIVE and not self.missing_evidence:
            raise ValueError("inconclusive assertions must state what evidence is missing")
        return self


class InfrastructureError(ContractModel):
    code: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=4_000)
    phase: str = Field(min_length=1, max_length=200)
    retryable: bool = False
    details: JsonObject = Field(default_factory=dict)


class CaseEvaluation(ContractModel):
    contract_version: Literal["agentcheck.case_evaluation.v1"] = (
        CASE_EVALUATION_CONTRACT_VERSION
    )
    evaluation_id: str = Field(min_length=1, max_length=200)
    scenario_id: str = Field(min_length=1, max_length=200)
    run_id: str | None = Field(default=None, max_length=200)
    verdict: Verdict
    assertions: tuple[AssertionResult, ...] = Field(min_length=1)
    evidence: tuple[Evidence, ...] = ()
    started_at: UtcDatetime
    completed_at: UtcDatetime
    summary: str = Field(min_length=1, max_length=8_000)
    infrastructure_error: InfrastructureError | None = None

    @model_validator(mode="after")
    def validate_verdict_and_evidence(self) -> "CaseEvaluation":
        if self.completed_at < self.started_at:
            raise ValueError("case evaluation cannot complete before it starts")

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique")
        known_evidence = set(evidence_ids)

        assertion_ids = [item.assertion_id for item in self.assertions]
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("assertion IDs must be unique")
        for assertion in self.assertions:
            references = set(assertion.supporting_evidence_ids).union(
                assertion.contradicting_evidence_ids
            )
            unknown = references.difference(known_evidence)
            if unknown:
                raise ValueError(
                    f"assertion {assertion.assertion_id!r} references unknown evidence: "
                    f"{sorted(unknown)}"
                )

        required = [assertion for assertion in self.assertions if assertion.required]
        hard_failures = [
            assertion
            for assertion in required
            if assertion.result == Verdict.FAIL and assertion.confidence >= 0.8
        ]
        if self.verdict == Verdict.PASS:
            if not required or any(assertion.result != Verdict.PASS for assertion in required):
                raise ValueError("PASS requires every required assertion to pass")
            if self.infrastructure_error is not None:
                raise ValueError("PASS cannot include an infrastructure error")
        elif self.verdict == Verdict.FAIL:
            if not hard_failures:
                raise ValueError("FAIL requires a high-confidence required assertion failure")
            if self.infrastructure_error is not None:
                raise ValueError("agent FAIL cannot also be an infrastructure error")
        elif self.verdict == Verdict.INCONCLUSIVE:
            if hard_failures:
                raise ValueError("high-confidence required failures must produce FAIL")
            if self.infrastructure_error is not None:
                raise ValueError("INCONCLUSIVE cannot include an infrastructure error")
        elif self.verdict == Verdict.INFRA_ERROR:
            if self.infrastructure_error is None:
                raise ValueError("INFRA_ERROR requires infrastructure error details")
            if hard_failures:
                raise ValueError("broken runs cannot also classify agent behavior as FAIL")
        return self
