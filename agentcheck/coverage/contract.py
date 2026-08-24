"""Strict, independently versioned behavioral-coverage contracts."""

from __future__ import annotations

import hmac
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from agentcheck.domain import ContractModel, canonical_hash
from agentcheck.privacy import redact_artifact, redact_log_text


BEHAVIORAL_COVERAGE_CONTRACT_VERSION: Literal[
    "agentcheck.behavioral_coverage.v1"
] = "agentcheck.behavioral_coverage.v1"

# This one bound applies at both nested levels.  At 24, all twelve v1 families
# can carry 24 requirements with 24 evidence IDs each while remaining well
# below privacy's 100,000-node and 8 MiB artifact limits even at maximum string
# lengths. Omitted counts make the loss of display detail explicit.
MAX_COVERAGE_DETAILS = 24

CoverageIdentifier = Annotated[str, Field(min_length=1, max_length=500)]


class BehavioralDimension(str, Enum):
    SUCCESS_PATH = "success_path"
    FAILURE_HANDLING = "failure_handling"
    TIMEOUT_HANDLING = "timeout_handling"
    FABRICATED_SUCCESS_AFTER_FAILURE = "fabricated_success_after_failure"
    AMBIGUOUS_OUTCOME = "ambiguous_outcome"
    RETRY_CONTROL = "retry_control"
    DUPLICATE_ACTION = "duplicate_action"
    CONFIRMATION_WITHOUT_CONSENT = "confirmation_without_consent"
    CONFIRMATION_WITH_CONSENT = "confirmation_with_consent"
    PREREQUISITE_SUCCESS = "prerequisite_success"
    PREREQUISITE_FAILURE = "prerequisite_failure"
    ORDERING = "ordering"


class BehavioralCoverageStatus(str, Enum):
    COVERED = "covered"
    PARTIAL = "partial"
    MISSING = "missing"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class BehavioralCoverageReferenceScope(str, Enum):
    COMPLETE = "complete"
    AVAILABLE_SCENARIOS_ONLY = "available_scenarios_only"


class BehavioralCoverageRequirement(ContractModel):
    """One explicit dimension for one stable tool or tool relationship."""

    subject: CoverageIdentifier
    status: BehavioralCoverageStatus
    reason_code: str = Field(min_length=1, max_length=100)
    evidence: tuple[CoverageIdentifier, ...] = Field(
        default=(), max_length=MAX_COVERAGE_DETAILS
    )
    omitted_evidence: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def normalize_evidence(self) -> "BehavioralCoverageRequirement":
        object.__setattr__(self, "subject", redact_log_text(self.subject))
        object.__setattr__(self, "reason_code", redact_log_text(self.reason_code))
        distinct = set(self.evidence)
        safe_evidence = {redact_log_text(identifier) for identifier in distinct}
        visible = tuple(sorted(safe_evidence))
        object.__setattr__(self, "evidence", visible)
        object.__setattr__(
            self,
            "omitted_evidence",
            self.omitted_evidence + len(distinct) - len(visible),
        )
        return self


class BehavioralCoverageFamily(ContractModel):
    """Counts and bounded detail for one behavioral dimension."""

    dimension: BehavioralDimension
    covered: int = Field(default=0, ge=0)
    partial: int = Field(default=0, ge=0)
    missing: int = Field(default=0, ge=0)
    unknown: int = Field(default=0, ge=0)
    unsupported: int = Field(default=0, ge=0)
    requirements: tuple[BehavioralCoverageRequirement, ...] = Field(
        default=(), max_length=MAX_COVERAGE_DETAILS
    )
    omitted: int = Field(default=0, ge=0)

    @property
    def applicable(self) -> int:
        """The honest denominator: unknown and unsupported are not applicable."""

        return self.covered + self.partial + self.missing

    @model_validator(mode="after")
    def validate_counts_and_order(self) -> "BehavioralCoverageFamily":
        ordered = tuple(sorted(self.requirements, key=lambda item: item.subject))
        subjects = [item.subject for item in ordered]
        if len(subjects) != len(set(subjects)):
            raise ValueError("behavioral coverage requirement subjects must be unique")
        object.__setattr__(self, "requirements", ordered)

        counts = {
            BehavioralCoverageStatus.COVERED: self.covered,
            BehavioralCoverageStatus.PARTIAL: self.partial,
            BehavioralCoverageStatus.MISSING: self.missing,
            BehavioralCoverageStatus.UNKNOWN: self.unknown,
            BehavioralCoverageStatus.UNSUPPORTED: self.unsupported,
        }
        visible: dict[BehavioralCoverageStatus, int] = {
            status: 0 for status in BehavioralCoverageStatus
        }
        for requirement in ordered:
            visible[requirement.status] += 1
        if any(visible[status] > count for status, count in counts.items()):
            raise ValueError("behavioral coverage counts cannot be below visible detail")
        if sum(counts.values()) != len(ordered) + self.omitted:
            raise ValueError(
                "behavioral coverage counts must equal visible plus omitted requirements"
            )
        return self


class BehavioralCoverage(ContractModel):
    """Behavioral requirements bound to one inspected spec and scenario set."""

    contract_version: Literal[
        "agentcheck.behavioral_coverage.v1"
    ] = BEHAVIORAL_COVERAGE_CONTRACT_VERSION
    spec_id: str = Field(min_length=1, max_length=200)
    spec_digest: str = Field(min_length=1, max_length=200)
    scenario_count: int = Field(ge=0)
    scenario_digest: str = Field(min_length=1, max_length=200)
    reference_scenario_count: int = Field(ge=0)
    reference_scenario_digest: str = Field(min_length=1, max_length=200)
    reference_scope: BehavioralCoverageReferenceScope = (
        BehavioralCoverageReferenceScope.COMPLETE
    )
    suite_fingerprint: str | None = Field(default=None, min_length=1, max_length=200)
    families: tuple[BehavioralCoverageFamily, ...] = Field(
        default=(), max_length=MAX_COVERAGE_DETAILS
    )
    fingerprint: str = Field(default="", max_length=200)

    def expected_fingerprint(self) -> str:
        """Return a checksum for detecting stale or corrupted coverage content.

        This is an unkeyed content checksum, not authentication. Source trust
        comes from independently recomputing only those source bindings whose
        documents are available; a stored selected run cannot reconstruct
        fingerprints for scenarios excluded before execution.
        """

        payload = self.model_dump(mode="json", exclude={"fingerprint"})
        # ArtifactStore always redacts before writing. Hash that exact safe
        # representation so mandatory redaction cannot invalidate integrity.
        return canonical_hash(redact_artifact(payload))

    @model_validator(mode="after")
    def normalize_families(self) -> "BehavioralCoverage":
        order = {dimension: index for index, dimension in enumerate(BehavioralDimension)}
        object.__setattr__(self, "spec_id", redact_log_text(self.spec_id))
        if self.suite_fingerprint is not None:
            object.__setattr__(
                self, "suite_fingerprint", redact_log_text(self.suite_fingerprint)
            )
        families = tuple(sorted(self.families, key=lambda item: order[item.dimension]))
        dimensions = [family.dimension for family in families]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("behavioral coverage families must be unique")
        if self.reference_scenario_count < self.scenario_count:
            raise ValueError(
                "reference scenario count cannot be below actual scenario count"
            )
        if (
            self.reference_scenario_count == self.scenario_count
            and self.reference_scenario_digest != self.scenario_digest
        ):
            raise ValueError(
                "equal actual and reference scenario counts require equal digests"
            )
        if (
            self.reference_scope
            is BehavioralCoverageReferenceScope.AVAILABLE_SCENARIOS_ONLY
            and self.reference_scenario_count != self.scenario_count
        ):
            raise ValueError(
                "available-scenarios-only reference count must equal actual count"
            )
        object.__setattr__(self, "families", families)
        expected = self.expected_fingerprint()
        if self.fingerprint and not hmac.compare_digest(self.fingerprint, expected):
            raise ValueError("behavioral coverage fingerprint does not match its contents")
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", expected)

        return self


__all__ = [
    "BEHAVIORAL_COVERAGE_CONTRACT_VERSION",
    "MAX_COVERAGE_DETAILS",
    "BehavioralCoverage",
    "BehavioralCoverageFamily",
    "BehavioralCoverageReferenceScope",
    "BehavioralCoverageRequirement",
    "BehavioralCoverageStatus",
    "BehavioralDimension",
]
