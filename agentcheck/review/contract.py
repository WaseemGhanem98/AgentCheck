"""Versioned human-review records for AgentCheck findings.

A review annotates an existing automated finding. It never changes
``PASS`` / ``FAIL`` / ``INCONCLUSIVE`` / ``INFRA_ERROR``.
"""

from __future__ import annotations

import hmac
from typing import Literal

from pydantic import Field, model_validator

from agentcheck.domain import ContractModel, Finding, UtcDatetime, canonical_hash


HUMAN_REVIEW_CONTRACT_VERSION: Literal["agentcheck.human_review.v1"] = (
    "agentcheck.human_review.v1"
)
REVIEW_SUBDIRECTORY = "reviews"
MAX_NOTE_CHARS = 2_000
MAX_REVIEWER_CHARS = 80
MAX_REVIEW_BYTES = 64 * 1024
HUMAN_DECISIONS = ("accepted", "rejected", "needs_followup")
HumanDecision = Literal["accepted", "rejected", "needs_followup"]


class ReviewSourceBinding(ContractModel):
    """Binding to the authoritative findings artifact, not the SQLite index."""

    findings_path: str = Field(min_length=1, max_length=500)
    findings_digest: str = Field(min_length=8, max_length=80)

    @model_validator(mode="after")
    def validate_digest(self) -> "ReviewSourceBinding":
        if not self.findings_digest.startswith("sha256:"):
            raise ValueError("findings_digest must be a sha256 digest")
        return self


class HumanReview(ContractModel):
    """Append-only human decision bound to one finding identity."""

    schema_version: Literal["agentcheck.human_review.v1"] = (
        HUMAN_REVIEW_CONTRACT_VERSION
    )
    review_id: str = Field(default="", max_length=200)
    run_id: str = Field(min_length=1, max_length=120)
    finding_id: str = Field(min_length=1, max_length=200)
    finding_fingerprint: str = Field(min_length=1, max_length=200)
    failure_signature: str = Field(min_length=1, max_length=500)
    automated_verdict: Literal["FAIL"]
    decision: HumanDecision
    note: str = Field(default="", max_length=MAX_NOTE_CHARS)
    reviewer: str | None = Field(default=None, max_length=MAX_REVIEWER_CHARS)
    recorded_at: UtcDatetime
    source: ReviewSourceBinding
    agentcheck_version: str = Field(min_length=1, max_length=100)
    fingerprint: str = ""

    def expected_fingerprint(self) -> str:
        return canonical_hash(
            self.model_dump(mode="json", exclude={"fingerprint", "review_id"})
        )

    def expected_review_id(self) -> str:
        return f"review-{self.expected_fingerprint().split(':', 1)[1][:24]}"

    @model_validator(mode="after")
    def validate_identity(self) -> "HumanReview":
        if self.reviewer is not None and not self.reviewer.strip():
            raise ValueError("reviewer must not be blank when supplied")
        expected = self.expected_fingerprint()
        if self.fingerprint and not _digest_equal(self.fingerprint, expected):
            raise ValueError("human review fingerprint does not match its contents")
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", expected)
        expected_id = self.expected_review_id()
        if self.review_id and not _digest_equal(self.review_id, expected_id):
            raise ValueError("human review ID does not match its contents")
        if not self.review_id:
            object.__setattr__(self, "review_id", expected_id)
        return self


def finding_fingerprint(finding: Finding) -> str:
    """Canonical identity of an authoritative finding document."""

    return finding.content_hash()


def bound_reviews_for_finding(
    finding: Finding,
    reviews: tuple["HumanReview", ...],
) -> tuple["HumanReview", ...]:
    """Return reviews whose fingerprint still matches this finding."""

    expected = finding_fingerprint(finding)
    return tuple(
        item
        for item in reviews
        if item.finding_id == finding.finding_id
        and item.finding_fingerprint == expected
        and item.failure_signature == finding.failure_signature
    )


def _digest_equal(left: str, right: str) -> bool:
    left_bytes = left.encode("utf-8")
    right_bytes = right.encode("utf-8")
    if len(left_bytes) != len(right_bytes):
        hmac.compare_digest(left_bytes, left_bytes)
        return False
    return hmac.compare_digest(left_bytes, right_bytes)


__all__ = [
    "HUMAN_DECISIONS",
    "HUMAN_REVIEW_CONTRACT_VERSION",
    "MAX_NOTE_CHARS",
    "MAX_REVIEW_BYTES",
    "MAX_REVIEWER_CHARS",
    "REVIEW_SUBDIRECTORY",
    "HumanDecision",
    "HumanReview",
    "ReviewSourceBinding",
    "bound_reviews_for_finding",
    "finding_fingerprint",
]
