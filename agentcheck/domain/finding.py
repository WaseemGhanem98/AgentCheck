"""Failure findings and human-controlled remediation proposals."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from .base import ContractModel


FINDING_CONTRACT_VERSION: Literal["agentcheck.finding.v1"] = "agentcheck.finding.v1"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RootCauseLayer(str, Enum):
    SYSTEM_PROMPT = "system_prompt"
    TOOL_DESCRIPTION = "tool_description"
    TOOL_SCHEMA = "tool_schema"
    WORKFLOW_LOGIC = "workflow_logic"
    GUARDRAIL = "guardrail"
    RETRY_POLICY = "retry_policy"
    STATE_HANDLING = "state_handling"
    RETRIEVAL_BEHAVIOR = "retrieval_behavior"
    RESPONSE_GENERATION = "response_generation"
    UNKNOWN = "unknown"


class FixTarget(str, Enum):
    SYSTEM_PROMPT = "system_prompt"
    TOOL_DESCRIPTION = "tool_description"
    TOOL_SCHEMA = "tool_schema"
    GUARDRAIL = "guardrail"
    RETRY_BEHAVIOR = "retry_behavior"
    WORKFLOW_LOGIC = "workflow_logic"
    ERROR_HANDLING = "error_handling"
    OTHER = "other"


class SuggestedFix(ContractModel):
    fix_id: str = Field(min_length=1, max_length=200)
    target: FixTarget
    summary: str = Field(min_length=1, max_length=4_000)
    rationale: str = Field(min_length=1, max_length=8_000)
    proposed_change: str | None = Field(default=None, max_length=100_000)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    requires_human_review: Literal[True] = True


class CriticalFindingBasis(str, Enum):
    DETERMINISTIC_EVIDENCE = "deterministic_evidence"
    HUMAN_CONFIRMED = "human_confirmed"


class Finding(ContractModel):
    """Structural grouping of related automated FAILs.

    ``failure_signature`` is a lexical analysis key (for example
    ``duplicate_side_effect``). It is not the versioned shrink/regression
    contract ``agentcheck.failure_signature.v1``.
    """

    contract_version: Literal["agentcheck.finding.v1"] = FINDING_CONTRACT_VERSION
    finding_id: str = Field(min_length=1, max_length=200)
    failure_signature: str = Field(min_length=1, max_length=500)
    title: str = Field(min_length=1, max_length=500)
    description: str = Field(min_length=1, max_length=8_000)
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    affected_scenario_ids: tuple[str, ...] = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    nearest_passing_scenario_ids: tuple[str, ...] = ()
    root_cause_layer: RootCauseLayer = RootCauseLayer.UNKNOWN
    likely_cause: str | None = Field(default=None, max_length=8_000)
    suggested_fixes: tuple[SuggestedFix, ...] = ()
    critical_basis: CriticalFindingBasis | None = None

    @model_validator(mode="after")
    def require_critical_confirmation(self) -> "Finding":
        if self.severity == Severity.CRITICAL and self.critical_basis is None:
            raise ValueError(
                "critical findings require deterministic evidence or explicit human confirmation"
            )
        if self.severity != Severity.CRITICAL and self.critical_basis is not None:
            raise ValueError("critical_basis is only valid for critical findings")
        return self
