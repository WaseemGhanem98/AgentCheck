"""Versioned policy-pack contracts. Packs are data, not executable plugins."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from agentcheck.domain import ContractModel, JsonObject


POLICY_PACK_CONTRACT_VERSION: Literal["agentcheck.policy_pack.v1"] = (
    "agentcheck.policy_pack.v1"
)


class PolicyRuleKind(str, Enum):
    """Maps onto existing deterministic evaluators; there is no second engine."""

    CONFIRMATION_BEFORE_TOOL = "confirmation_before_tool"
    NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT = "no_retry_after_ambiguous_timeout"
    NO_FABRICATED_SUCCESS = "no_fabricated_success"
    NO_DUPLICATE_SIDE_EFFECT = "no_duplicate_side_effect"


class PolicyRule(ContractModel):
    """One declared expectation. Unknown fields are rejected."""

    schema_version: Literal["agentcheck.policy_pack.v1"] = POLICY_PACK_CONTRACT_VERSION
    rule_id: str = Field(min_length=1, max_length=200)
    kind: PolicyRuleKind
    description: str = Field(min_length=1, max_length=4_000)
    tool_name: str | None = Field(default=None, max_length=200)
    parameters: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_declared_tool_when_needed(self) -> "PolicyRule":
        if self.kind is not PolicyRuleKind.NO_FABRICATED_SUCCESS and not self.tool_name:
            raise ValueError(
                f"policy rule {self.rule_id!r} of kind {self.kind.value!r} "
                "requires a declared tool_name"
            )
        return self


class PolicyPack(ContractModel):
    """A reusable, versioned bundle of deterministic evaluation rules."""

    schema_version: Literal["agentcheck.policy_pack.v1"] = POLICY_PACK_CONTRACT_VERSION
    pack_id: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4_000)
    rules: tuple[PolicyRule, ...] = Field(min_length=1)


__all__ = [
    "POLICY_PACK_CONTRACT_VERSION",
    "PolicyPack",
    "PolicyRule",
    "PolicyRuleKind",
]
