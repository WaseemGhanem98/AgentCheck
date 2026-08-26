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
    """Maps onto existing deterministic evaluators; there is no second engine.

    Every member here names a trajectory or output evaluator that already
    exists. The four added for policies v1 were evaluable long before they were
    authorable: the evaluator understood an ordering relation or a retry ceiling,
    but no declared rule could reach it, so the only way to assert one was to
    hand-write a scenario.
    """

    CONFIRMATION_BEFORE_TOOL = "confirmation_before_tool"
    NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT = "no_retry_after_ambiguous_timeout"
    NO_FABRICATED_SUCCESS = "no_fabricated_success"
    NO_DUPLICATE_SIDE_EFFECT = "no_duplicate_side_effect"
    NO_SAME_STAGE_DUPLICATE_ACTION = "no_same_stage_duplicate_action"
    ORDERING = "ordering"
    MAX_RETRIES = "max_retries"
    MAX_TOOL_CALLS = "max_tool_calls"
    MAX_MODEL_TURNS = "max_model_turns"
    REQUIRED_HANDOFF = "required_handoff"
    FORBIDDEN_HANDOFF = "forbidden_handoff"
    MAX_HANDOFFS = "max_handoffs"
    NO_HANDOFF_LOOP = "no_handoff_loop"
    HANDOFF_BEFORE_TOOL = "handoff_before_tool"


# What each kind must be told before its evaluator can decide anything. A rule
# missing these does not become a lenient rule -- the evaluator would return
# INCONCLUSIVE for every run, which reads like a policy that is being enforced
# and is not. Rejecting it at parse time is the honest failure.
_REQUIRED_PARAMETERS: dict[PolicyRuleKind, tuple[str, ...]] = {
    PolicyRuleKind.ORDERING: ("required_before",),
    PolicyRuleKind.MAX_RETRIES: ("max_retries",),
    PolicyRuleKind.MAX_TOOL_CALLS: ("maximum",),
    PolicyRuleKind.MAX_MODEL_TURNS: ("maximum",),
    PolicyRuleKind.MAX_HANDOFFS: ("maximum",),
}

# Budgets are a property of the run, not of one tool, so they do not need a
# tool_name. A handoff is the same: it moves the whole conversation between
# agents rather than deciding one call. `handoff_before_tool` is the exception
# and stays out of this set -- it asserts that a handoff preceded a *specific*
# call, so without the tool it names nothing.
_TOOL_FREE_KINDS = frozenset(
    {
        PolicyRuleKind.NO_FABRICATED_SUCCESS,
        PolicyRuleKind.MAX_TOOL_CALLS,
        PolicyRuleKind.MAX_MODEL_TURNS,
        PolicyRuleKind.REQUIRED_HANDOFF,
        PolicyRuleKind.FORBIDDEN_HANDOFF,
        PolicyRuleKind.MAX_HANDOFFS,
        PolicyRuleKind.NO_HANDOFF_LOOP,
    }
)

_NON_NEGATIVE_INTEGER_PARAMETERS = frozenset(
    {"max_retries", "maximum", "minimum", "max_edge_repeats"}
)

# An agent filter narrows which handoffs a rule is about. Empty is not a
# narrower filter, it is no filter: the evaluator treats a missing name as
# "any agent", so `""` would silently widen a rule the author meant to scope.
_AGENT_NAME_PARAMETERS = ("from_agent", "to_agent")


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
        if self.kind not in _TOOL_FREE_KINDS and not self.tool_name:
            raise ValueError(
                f"policy rule {self.rule_id!r} of kind {self.kind.value!r} "
                "requires a declared tool_name"
            )
        for name in _REQUIRED_PARAMETERS.get(self.kind, ()):
            if name not in self.parameters:
                raise ValueError(
                    f"policy rule {self.rule_id!r} of kind {self.kind.value!r} "
                    f"requires the {name!r} parameter"
                )
        for name in _NON_NEGATIVE_INTEGER_PARAMETERS:
            if name not in self.parameters:
                continue
            value = self.parameters[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"policy rule {self.rule_id!r} needs {name!r} to be a "
                    f"non-negative integer, not {value!r}"
                )
        if "required_before" in self.parameters:
            required_before = self.parameters["required_before"]
            if not isinstance(required_before, str) or not required_before:
                raise ValueError(
                    f"policy rule {self.rule_id!r} needs 'required_before' to name "
                    "the tool that must be observed first"
                )
            if required_before == self.tool_name:
                raise ValueError(
                    f"policy rule {self.rule_id!r} orders {self.tool_name!r} "
                    "before itself"
                )
        for name in _AGENT_NAME_PARAMETERS:
            if name not in self.parameters:
                continue
            value = self.parameters[name]
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"policy rule {self.rule_id!r} needs {name!r} to name an "
                    f"agent, not {value!r}"
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
