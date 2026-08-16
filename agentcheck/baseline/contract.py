"""Versioned evaluation baseline and comparison contracts.

Baselines are built from stored JSON/JSONL run artifacts. HTML, SQLite, and
human reviews are not comparison inputs.
"""

from __future__ import annotations

import hmac
from typing import Any, Literal

from pydantic import Field, model_serializer, model_validator

from agentcheck.domain import ContractModel, canonical_hash
from agentcheck.shrink.signature import FailureSignature


BASELINE_CONTRACT_VERSION: Literal["agentcheck.baseline.v1"] = (
    "agentcheck.baseline.v1"
)
COMPARISON_CONTRACT_VERSION: Literal["agentcheck.baseline_comparison.v1"] = (
    "agentcheck.baseline_comparison.v1"
)
DEFAULT_BASELINE_FILENAME = "agentcheck-baseline.json"
MAX_BASELINE_BYTES = 4 * 1024 * 1024
BLOCKING_CATEGORIES = ("new_regression", "changed_failure")
ComparisonCategory = Literal[
    "unchanged_failure",
    "resolved_failure",
    "new_regression",
    "changed_failure",
    "new_scenario",
    "removed_scenario",
    "changed_scenario",
    "inconclusive_change",
    "infra_change",
    "unchanged",
]


class BaselineCase(ContractModel):
    """One evaluated scenario in a baseline, keyed by scenario fingerprint."""

    scenario_id: str = Field(min_length=1, max_length=200)
    fingerprint: str = Field(min_length=1, max_length=200)
    verdict: Literal["PASS", "FAIL", "INCONCLUSIVE", "INFRA_ERROR"]
    failure_signature: FailureSignature | None = None

    @model_serializer(mode="wrap")
    def omit_absent_signature(self, serializer: Any) -> dict[str, Any]:
        data = serializer(self)
        if data.get("failure_signature") is None:
            data.pop("failure_signature", None)
        return data


class BaselineFinding(ContractModel):
    """Automated finding identity. Human reviews are excluded."""

    finding_id: str = Field(min_length=1, max_length=200)
    finding_fingerprint: str = Field(min_length=1, max_length=200)
    failure_signature: str = Field(min_length=1, max_length=500)


class EvaluationBaseline(ContractModel):
    """Trusted snapshot of automated AgentCheck results for later comparison."""

    schema_version: Literal["agentcheck.baseline.v1"] = BASELINE_CONTRACT_VERSION
    baseline_id: str = Field(default="", max_length=200)
    created_from_run_id: str = Field(min_length=1, max_length=120)
    spec_id: str = Field(min_length=1, max_length=200)
    suite_id: str | None = Field(default=None, max_length=200)
    suite_fingerprint: str | None = Field(default=None, max_length=200)
    seed: int = Field(ge=0, le=2**63 - 1)
    policy_pack_ids: tuple[str, ...] = ()
    selection_algorithm: str | None = Field(default=None, max_length=200)
    selected_fingerprints: tuple[str, ...] = ()
    cases: tuple[BaselineCase, ...] = Field(min_length=1)
    findings: tuple[BaselineFinding, ...] = ()
    agentcheck_version: str = Field(min_length=1, max_length=100)
    fingerprint: str = ""

    @model_serializer(mode="wrap")
    def omit_absent_optional(self, serializer: Any) -> dict[str, Any]:
        data = serializer(self)
        for key in ("suite_id", "suite_fingerprint", "selection_algorithm"):
            if data.get(key) is None:
                data.pop(key, None)
        if not data.get("policy_pack_ids"):
            data.pop("policy_pack_ids", None)
        if not data.get("selected_fingerprints"):
            data.pop("selected_fingerprints", None)
        if not data.get("findings"):
            data.pop("findings", None)
        return data

    def expected_fingerprint(self) -> str:
        return canonical_hash(
            self.model_dump(
                mode="json",
                exclude={"fingerprint", "baseline_id", "created_from_run_id"},
            )
        )

    def expected_baseline_id(self) -> str:
        return f"baseline-{self.expected_fingerprint().split(':', 1)[1][:24]}"

    @model_validator(mode="after")
    def validate_identity(self) -> "EvaluationBaseline":
        fingerprints = [item.fingerprint for item in self.cases]
        if tuple(sorted(fingerprints)) != tuple(fingerprints):
            raise ValueError("baseline cases must be sorted by scenario fingerprint")
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("baseline scenario fingerprints must be unique")
        finding_ids = [item.finding_id for item in self.findings]
        if tuple(sorted(finding_ids)) != tuple(finding_ids):
            raise ValueError("baseline findings must be sorted by finding_id")
        expected = self.expected_fingerprint()
        if self.fingerprint and not _digest_equal(self.fingerprint, expected):
            raise ValueError("baseline fingerprint does not match its contents")
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", expected)
        expected_id = self.expected_baseline_id()
        if self.baseline_id and not _digest_equal(self.baseline_id, expected_id):
            raise ValueError("baseline ID does not match its contents")
        if not self.baseline_id:
            object.__setattr__(self, "baseline_id", expected_id)
        return self


class ComparisonItem(ContractModel):
    category: ComparisonCategory
    scenario_id: str = Field(min_length=1, max_length=200)
    fingerprint: str = Field(min_length=1, max_length=200)
    baseline_verdict: str | None = Field(default=None, max_length=20)
    current_verdict: str | None = Field(default=None, max_length=20)
    baseline_failure_fingerprint: str | None = Field(default=None, max_length=200)
    current_failure_fingerprint: str | None = Field(default=None, max_length=200)
    blocking: bool = False

    @model_serializer(mode="wrap")
    def omit_absent_optional(self, serializer: Any) -> dict[str, Any]:
        data = serializer(self)
        for key in (
            "baseline_verdict",
            "current_verdict",
            "baseline_failure_fingerprint",
            "current_failure_fingerprint",
        ):
            if data.get(key) is None:
                data.pop(key, None)
        if not data.get("blocking"):
            data.pop("blocking", None)
        return data


class BaselineComparison(ContractModel):
    schema_version: Literal["agentcheck.baseline_comparison.v1"] = (
        COMPARISON_CONTRACT_VERSION
    )
    baseline_id: str = Field(min_length=1, max_length=200)
    current_run_id: str = Field(min_length=1, max_length=120)
    baseline_scenario_count: int = Field(ge=0)
    current_scenario_count: int = Field(ge=0)
    baseline_failure_count: int = Field(ge=0)
    current_failure_count: int = Field(ge=0)
    new_regression_count: int = Field(ge=0)
    resolved_count: int = Field(ge=0)
    unchanged_failure_count: int = Field(ge=0)
    changed_failure_count: int = Field(ge=0)
    items: tuple[ComparisonItem, ...] = ()
    blocking_categories: tuple[str, ...] = BLOCKING_CATEGORIES
    fingerprint: str = ""

    def expected_fingerprint(self) -> str:
        return canonical_hash(
            self.model_dump(mode="json", exclude={"fingerprint"})
        )

    @model_validator(mode="after")
    def validate_identity(self) -> "BaselineComparison":
        expected = self.expected_fingerprint()
        if self.fingerprint and not _digest_equal(self.fingerprint, expected):
            raise ValueError("comparison fingerprint does not match its contents")
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", expected)
        return self


def _digest_equal(left: str, right: str) -> bool:
    left_bytes = left.encode("utf-8")
    right_bytes = right.encode("utf-8")
    if len(left_bytes) != len(right_bytes):
        hmac.compare_digest(left_bytes, left_bytes)
        return False
    return hmac.compare_digest(left_bytes, right_bytes)


__all__ = [
    "BASELINE_CONTRACT_VERSION",
    "BLOCKING_CATEGORIES",
    "COMPARISON_CONTRACT_VERSION",
    "DEFAULT_BASELINE_FILENAME",
    "MAX_BASELINE_BYTES",
    "BaselineCase",
    "BaselineComparison",
    "BaselineFinding",
    "ComparisonCategory",
    "ComparisonItem",
    "EvaluationBaseline",
]
