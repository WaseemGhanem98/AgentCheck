"""Strict, independently versioned run-to-run behavioral comparison contracts.

A comparison describes two stored runs. It never re-executes a target, never
re-derives a verdict, and never asserts that a repeated verdict is
deterministic: AgentCheck does not replay model output, so an unchanged pair is
evidence about those two executions and nothing more.
"""

from __future__ import annotations

import hmac
from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from agentcheck.baseline.contract import (
    BLOCKING_CATEGORIES,
    CERTIFICATION_FAILURE_CATEGORIES,
    ComparisonCategory,
)
from agentcheck.domain import ContractModel, canonical_hash
from agentcheck.privacy import redact_artifact, redact_log_text


RUN_COMPARISON_CONTRACT_VERSION: Literal["agentcheck.run_comparison.v1"] = (
    "agentcheck.run_comparison.v1"
)

# One bound below privacy's 100-item artifact limit, so the redacted view this
# document hashes is lossless: a checksum must not be blind to a row the
# document itself carries. Items are ordered with every blocking and
# uncertifiable row first, so a larger suite omits only quiet detail, and the
# counts are always computed over every item rather than the visible ones.
MAX_COMPARISON_ITEMS = 100

# The baseline vocabulary, plus the one distinction a run-to-run comparison can
# make and a curated baseline cannot: a scenario absent because this run's
# selection excluded it was never evaluated, so it is not a removed scenario.
RunComparisonCategory = (
    ComparisonCategory | Literal["deselected_scenario"]
)

# Re-exported, not restated: one repository must not be able to block on two
# different definitions of "regression" or of "could not certify".
UNCERTIFIABLE_CATEGORIES: tuple[str, ...] = CERTIFICATION_FAILURE_CATEGORIES


class Comparability(str, Enum):
    COMPARABLE = "comparable"
    INCOMPARABLE = "incomparable"


class IncomparableReason(str, Enum):
    """Why no scenario-level classification was produced."""

    SAME_RUN = "same_run"
    NO_SHARED_SCENARIOS = "no_shared_scenarios"


class ComparabilityCaveat(str, Enum):
    """A binding that moved between the two runs, disclosed but not fatal."""

    SPEC_CHANGED = "spec_changed"
    SCENARIO_SET_CHANGED = "scenario_set_changed"
    SUITE_FINGERPRINT_CHANGED = "suite_fingerprint_changed"
    SEED_CHANGED = "seed_changed"
    SELECTION_ACTIVE = "selection_active"
    SOURCE_REVISION_UNRECORDED = "source_revision_unrecorded"
    SOURCE_REVISION_UNCHANGED = "source_revision_unchanged"
    SCENARIO_IDENTITY_CHANGED = "scenario_identity_changed"


class RunComparisonItem(ContractModel):
    """One scenario compared across two runs."""

    category: RunComparisonCategory
    scenario_id: str = Field(min_length=1, max_length=200)
    fingerprint: str = Field(min_length=1, max_length=200)
    base_verdict: str | None = Field(default=None, max_length=20)
    head_verdict: str | None = Field(default=None, max_length=20)
    base_failure_fingerprint: str | None = Field(default=None, max_length=200)
    head_failure_fingerprint: str | None = Field(default=None, max_length=200)
    blocking: bool = False

    @model_validator(mode="after")
    def normalize_identifiers(self) -> "RunComparisonItem":
        object.__setattr__(self, "scenario_id", redact_log_text(self.scenario_id))
        object.__setattr__(self, "fingerprint", redact_log_text(self.fingerprint))
        return self


class RunComparison(ContractModel):
    """Behavioral differences between two stored runs of one target."""

    schema_version: Literal["agentcheck.run_comparison.v1"] = (
        RUN_COMPARISON_CONTRACT_VERSION
    )
    base_run_id: str = Field(min_length=1, max_length=120)
    head_run_id: str = Field(min_length=1, max_length=120)
    comparability: Comparability
    incomparable_reason: IncomparableReason | None = None
    caveats: tuple[ComparabilityCaveat, ...] = ()
    base_scenario_count: int = Field(ge=0)
    head_scenario_count: int = Field(ge=0)
    shared_scenario_count: int = Field(ge=0)
    base_failure_count: int = Field(ge=0)
    head_failure_count: int = Field(ge=0)
    new_regression_count: int = Field(ge=0)
    changed_failure_count: int = Field(ge=0)
    resolved_count: int = Field(ge=0)
    unchanged_failure_count: int = Field(ge=0)
    uncertifiable_count: int = Field(ge=0)
    items: tuple[RunComparisonItem, ...] = Field(
        default=(), max_length=MAX_COMPARISON_ITEMS
    )
    omitted_items: int = Field(default=0, ge=0)
    fingerprint: str = Field(default="", max_length=200)

    def expected_fingerprint(self) -> str:
        """Return an unkeyed checksum over the redacted comparison content.

        This detects stale or corrupted content. It is not authentication:
        trust in a comparison comes from re-deriving it from the two stored
        runs, which is exactly what producing it does.
        """

        payload = self.model_dump(mode="json", exclude={"fingerprint"})
        return canonical_hash(redact_artifact(payload))

    @model_validator(mode="after")
    def validate_identity(self) -> "RunComparison":
        object.__setattr__(self, "base_run_id", redact_log_text(self.base_run_id))
        object.__setattr__(self, "head_run_id", redact_log_text(self.head_run_id))
        caveats = tuple(dict.fromkeys(self.caveats))
        order = {caveat: index for index, caveat in enumerate(ComparabilityCaveat)}
        object.__setattr__(
            self, "caveats", tuple(sorted(caveats, key=lambda item: order[item]))
        )
        if self.comparability is Comparability.INCOMPARABLE:
            if self.incomparable_reason is None:
                raise ValueError("an incomparable result requires an explicit reason")
            if self.items or self.omitted_items:
                raise ValueError(
                    "an incomparable result must not classify any scenario"
                )
        elif self.incomparable_reason is not None:
            raise ValueError("a comparable result must not carry an incomparable reason")
        blocking = [item for item in self.items if item.blocking]
        if any(item.category not in BLOCKING_CATEGORIES for item in blocking):
            raise ValueError(
                "only new_regression and changed_failure may be marked blocking"
            )
        expected = self.expected_fingerprint()
        if self.fingerprint and not hmac.compare_digest(self.fingerprint, expected):
            raise ValueError("run comparison fingerprint does not match its contents")
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", expected)
        return self


__all__ = [
    "BLOCKING_CATEGORIES",
    "MAX_COMPARISON_ITEMS",
    "RUN_COMPARISON_CONTRACT_VERSION",
    "UNCERTIFIABLE_CATEGORIES",
    "Comparability",
    "ComparabilityCaveat",
    "IncomparableReason",
    "RunComparison",
    "RunComparisonCategory",
    "RunComparisonItem",
]
