"""Deterministic failure identity for shrink acceptance.

A candidate reproduces the original counterexample only when it is still a
required high-confidence FAIL whose failed assertion IDs and oracle IDs match.
Verdict equality alone is not enough. Evidence IDs, run IDs, and human-facing
prose are excluded because they are not stable across executions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from agentcheck.domain import CaseEvaluation, ContractModel, Verdict, canonical_hash


FAILURE_SIGNATURE_CONTRACT_VERSION: Literal["agentcheck.failure_signature.v1"] = (
    "agentcheck.failure_signature.v1"
)
_UNSTABLE_ASSERTION_PREFIXES = ("schema:", "budget:", "infra:")


class FailedAssertion(ContractModel):
    """One required high-confidence FAIL, identified without prose or evidence."""

    assertion_id: str = Field(min_length=1, max_length=200)
    oracle_ids: tuple[str, ...] = Field(min_length=1)


class FailureSignature(ContractModel):
    """Canonical identity of an AgentCheck counterexample."""

    schema_version: Literal["agentcheck.failure_signature.v1"] = (
        FAILURE_SIGNATURE_CONTRACT_VERSION
    )
    verdict: Literal["fail"] = "fail"
    failed_assertions: tuple[FailedAssertion, ...] = Field(min_length=1)
    fingerprint: str = ""

    def expected_fingerprint(self) -> str:
        return canonical_hash(
            self.model_dump(mode="json", exclude={"fingerprint"})
        )

    @model_validator(mode="after")
    def validate_identity(self) -> "FailureSignature":
        assertion_ids = [item.assertion_id for item in self.failed_assertions]
        if tuple(sorted(assertion_ids)) != tuple(assertion_ids):
            raise ValueError("failed assertions must be sorted by assertion_id")
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("failed assertion IDs must be unique")
        for item in self.failed_assertions:
            if tuple(sorted(item.oracle_ids)) != item.oracle_ids:
                raise ValueError("oracle IDs on a failed assertion must be sorted")
        expected = self.expected_fingerprint()
        if self.fingerprint and self.fingerprint != expected:
            raise ValueError("failure signature fingerprint does not match its contents")
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", expected)
        return self


def unsupported_failure_reason(evaluation: CaseEvaluation) -> str | None:
    """Return why this evaluation cannot be shrunk, or None when it can."""

    if evaluation.verdict == Verdict.INFRA_ERROR:
        return "infrastructure errors are not agent counterexamples"
    if evaluation.verdict == Verdict.PASS:
        return "passing cases are not counterexamples"
    if evaluation.verdict == Verdict.INCONCLUSIVE:
        return "inconclusive cases are not counterexamples"
    if evaluation.verdict != Verdict.FAIL:
        return f"unsupported verdict {evaluation.verdict.value}"
    failures = _hard_failures(evaluation)
    if not failures:
        return "FAIL without a required high-confidence assertion cannot be shrunk"
    for assertion_id, _oracle_ids in failures:
        if assertion_id.startswith(_UNSTABLE_ASSERTION_PREFIXES):
            return (
                f"assertion {assertion_id!r} is execution-scoped and is not a "
                "stable shrink identity"
            )
    return None


def extract_failure_signature(evaluation: CaseEvaluation) -> FailureSignature:
    """Build a shrink identity. Raises ValueError when shrinking must refuse."""

    reason = unsupported_failure_reason(evaluation)
    if reason is not None:
        raise ValueError(reason)
    items = tuple(
        FailedAssertion(assertion_id=assertion_id, oracle_ids=oracle_ids)
        for assertion_id, oracle_ids in _hard_failures(evaluation)
    )
    return FailureSignature(failed_assertions=items)


def signatures_match(left: FailureSignature, right: FailureSignature) -> bool:
    return left.fingerprint == right.fingerprint


def protected_criterion_ids(signature: FailureSignature) -> frozenset[str]:
    """Criterion IDs whose removal would change the failure identity."""

    protected: set[str] = set()
    suffix = ":confirmation"
    for item in signature.failed_assertions:
        protected.add(item.assertion_id)
        if item.assertion_id.endswith(suffix):
            protected.add(item.assertion_id[: -len(suffix)])
    return frozenset(protected)


def _hard_failures(evaluation: CaseEvaluation) -> tuple[tuple[str, tuple[str, ...]], ...]:
    items: list[tuple[str, tuple[str, ...]]] = []
    for assertion in evaluation.assertions:
        if not assertion.required:
            continue
        if assertion.result != Verdict.FAIL or assertion.confidence < 0.8:
            continue
        oracle_ids = tuple(sorted(set(assertion.oracle_ids)))
        items.append((assertion.assertion_id, oracle_ids))
    items.sort(key=lambda item: item[0])
    return tuple(items)


__all__ = [
    "FAILURE_SIGNATURE_CONTRACT_VERSION",
    "FailedAssertion",
    "FailureSignature",
    "extract_failure_signature",
    "protected_criterion_ids",
    "signatures_match",
    "unsupported_failure_reason",
]
