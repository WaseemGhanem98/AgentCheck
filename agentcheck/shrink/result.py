"""Versioned shrink-result contract. Not a replay manifest and not disclosure."""

from __future__ import annotations

import hmac
import json
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from agentcheck.artifacts import replace_private_file
from agentcheck.config import AgentCheckConfig, contained_path
from agentcheck.domain import ContractModel, canonical_hash
from agentcheck.errors import ConfigurationError

from .complexity import ScenarioComplexity
from .signature import FailureSignature


SHRINK_RESULT_CONTRACT_VERSION: Literal["agentcheck.shrink_result.v1"] = (
    "agentcheck.shrink_result.v1"
)
SHRINK_SUBDIRECTORY = "shrink"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,119}$")
MAX_SHRINK_RESULT_BYTES = 1 * 1024 * 1024

DEFAULT_MAX_CANDIDATES = 32
MAX_MAX_CANDIDATES = 128
DEFAULT_MAX_ROUNDS = 12
MAX_MAX_ROUNDS = 24
MAX_SOURCE_SCANS = 16


class ShrinkBudget(ContractModel):
    max_candidates: int = Field(ge=1, le=MAX_MAX_CANDIDATES)
    max_rounds: int = Field(ge=1, le=MAX_MAX_ROUNDS)


class RejectedCount(ContractModel):
    reason: str = Field(min_length=1, max_length=100)
    count: int = Field(ge=1)


class ShrinkResult(ContractModel):
    """Machine-readable minimization record. Human explanation stays on the CLI."""

    schema_version: Literal["agentcheck.shrink_result.v1"] = (
        SHRINK_RESULT_CONTRACT_VERSION
    )
    result_id: str = Field(default="", max_length=200)
    source_manifest_id: str = Field(min_length=1, max_length=200)
    source_manifest_fingerprint: str = Field(min_length=1, max_length=200)
    source_scenario_id: str = Field(min_length=1, max_length=200)
    source_scenario_fingerprint: str = Field(min_length=1, max_length=200)
    failure_signature: FailureSignature
    original_complexity: ScenarioComplexity
    minimized_complexity: ScenarioComplexity
    minimized_scenario_fingerprint: str = Field(min_length=1, max_length=200)
    minimized_manifest_id: str = Field(min_length=1, max_length=200)
    minimized_manifest_path: str = Field(min_length=1, max_length=500)
    candidate_executions: int = Field(ge=0)
    accepted_reductions: int = Field(ge=0)
    rejected_candidates: int = Field(ge=0)
    skipped_invalid: int = Field(ge=0)
    rejected_by_reason: tuple[RejectedCount, ...] = ()
    budget: ShrinkBudget
    budget_exhausted: bool
    minimality: Literal["locally_minimal", "budget_exhausted"]
    requires_human_review: Literal[True] = True
    agentcheck_version: str = Field(min_length=1, max_length=100)
    fingerprint: str = ""

    def expected_fingerprint(self) -> str:
        return canonical_hash(
            self.model_dump(
                mode="json",
                exclude={
                    "fingerprint",
                    "result_id",
                    "minimized_manifest_id",
                    "minimized_manifest_path",
                },
            )
        )

    def expected_result_id(self) -> str:
        return f"shrink-{self.expected_fingerprint().split(':', 1)[1][:24]}"

    @model_validator(mode="after")
    def validate_identity(self) -> "ShrinkResult":
        if self.minimality == "budget_exhausted" and not self.budget_exhausted:
            raise ValueError("budget_exhausted must be true when minimality is budget_exhausted")
        if self.minimality == "locally_minimal" and self.budget_exhausted:
            raise ValueError("locally_minimal results cannot also be budget exhausted")
        expected = self.expected_fingerprint()
        if self.fingerprint and not _digest_equal(self.fingerprint, expected):
            raise ValueError("shrink result fingerprint does not match its contents")
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", expected)
        expected_id = self.expected_result_id()
        if self.result_id and not _digest_equal(self.result_id, expected_id):
            raise ValueError("shrink result ID does not match its contents")
        if not self.result_id:
            object.__setattr__(self, "result_id", expected_id)
        return self


def _digest_equal(left: str, right: str) -> bool:
    left_bytes = left.encode("utf-8")
    right_bytes = right.encode("utf-8")
    if len(left_bytes) != len(right_bytes):
        hmac.compare_digest(left_bytes, left_bytes)
        return False
    return hmac.compare_digest(left_bytes, right_bytes)


def encode_shrink_result(result: ShrinkResult) -> bytes:
    payload = (
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > MAX_SHRINK_RESULT_BYTES:
        raise ConfigurationError(
            f"shrink result exceeds the {MAX_SHRINK_RESULT_BYTES} byte size bound"
        )
    return payload


def shrink_result_relative_path(config: AgentCheckConfig, run_id: str) -> str:
    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ConfigurationError(
            "run ID must contain only letters, digits, underscores, or hyphens"
        )
    return f"{config.artifacts_directory}/{SHRINK_SUBDIRECTORY}/{run_id}.json"


def write_shrink_result(
    root: Path,
    config: AgentCheckConfig,
    result: ShrinkResult,
    run_id: str,
) -> Path:
    relative = shrink_result_relative_path(config, run_id)
    destination = contained_path(root, relative)
    payload = encode_shrink_result(result)
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(destination.parent, 0o700)
        replace_private_file(destination, payload)
    except OSError as exc:
        raise ConfigurationError(f"unable to write shrink result: {exc}") from exc
    return destination


__all__ = [
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MAX_ROUNDS",
    "MAX_MAX_CANDIDATES",
    "MAX_MAX_ROUNDS",
    "MAX_SOURCE_SCANS",
    "MAX_SHRINK_RESULT_BYTES",
    "SHRINK_RESULT_CONTRACT_VERSION",
    "SHRINK_SUBDIRECTORY",
    "RejectedCount",
    "ShrinkBudget",
    "ShrinkResult",
    "encode_shrink_result",
    "shrink_result_relative_path",
    "write_shrink_result",
]
