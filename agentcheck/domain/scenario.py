"""Strongly typed, replayable deterministic test scenarios."""

from __future__ import annotations

import hmac
from enum import Enum
from typing import Any, Literal

from pydantic import Field, field_validator, model_serializer, model_validator

from .base import ContractModel, JsonObject, JsonValue, canonical_hash


SCENARIO_CONTRACT_VERSION: Literal["agentcheck.scenario.v1"] = (
    "agentcheck.scenario.v1"
)

# Positive-path cases are the only ones whose pass depends on the agent having
# actually called the tool, so their IDs carry a prefix that both the generator
# and every reader of a finished run can recognise. Kept here, next to the
# scenario contract, so the two never drift apart.
ACTION_SCENARIO_PREFIX = "action-"

# Each follow-up opens another agent execution stage inside one scenario, so the
# count is bounded for the same reason every other collection here is: a case
# must not be able to ask for unbounded work.
MAX_FOLLOWUP_TURNS = 8


class ConversationRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ConversationTurn(ContractModel):
    turn_id: str = Field(min_length=1, max_length=200)
    role: ConversationRole
    content: str = Field(min_length=1, max_length=100_000)
    metadata: JsonObject = Field(default_factory=dict)


class SimulatedToolStatus(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    ERROR = "error"
    TIMEOUT = "timeout"
    MALFORMED = "malformed"
    PARTIAL = "partial"
    STALE = "stale"


class WorldStateEffect(ContractModel):
    path: str = Field(min_length=1, max_length=1_000)
    before: JsonValue = None
    after: JsonValue = None


class SimulatedToolOutcome(ContractModel):
    status: SimulatedToolStatus
    result: JsonValue = None
    error_code: str | None = Field(default=None, max_length=200)
    error_message: str | None = Field(default=None, max_length=4_000)
    latency_ms: float = Field(default=0.0, ge=0.0)
    state_effects: tuple[WorldStateEffect, ...] = ()

    @model_validator(mode="after")
    def validate_error_fixture(self) -> "SimulatedToolOutcome":
        if self.status in {SimulatedToolStatus.ERROR, SimulatedToolStatus.TIMEOUT} and not (
            self.error_code or self.error_message
        ):
            raise ValueError("error and timeout fixtures require an error code or message")
        return self


class ToolFixture(ContractModel):
    fixture_id: str = Field(min_length=1, max_length=200)
    tool_name: str = Field(min_length=1, max_length=200)
    arguments_match: JsonObject = Field(default_factory=dict)
    invocation_index: int | None = Field(default=None, ge=1)
    priority: int = Field(default=0, ge=-100, le=100)
    outcome: SimulatedToolOutcome


class FaultType(str, Enum):
    ERROR = "error"
    TIMEOUT = "timeout"
    MALFORMED_RESPONSE = "malformed_response"
    EMPTY_RESPONSE = "empty_response"
    PARTIAL_RESPONSE = "partial_response"
    STALE_RESPONSE = "stale_response"


class InjectedFault(ContractModel):
    fault_id: str = Field(min_length=1, max_length=200)
    tool_name: str = Field(min_length=1, max_length=200)
    fault_type: FaultType
    invocation_index: int = Field(default=1, ge=1)
    payload: JsonValue = None
    message: str | None = Field(default=None, max_length=4_000)
    latency_ms: float | None = Field(default=None, ge=0.0)


class OracleStrength(str, Enum):
    EXECUTABLE_WORLD_STATE = "executable_world_state"
    TOOL_CONTRACT = "tool_contract"
    VERSIONED_POLICY = "versioned_policy"
    EXPLICIT_INSTRUCTION = "explicit_instruction"
    CONTROLLED_WORLD_FACT = "controlled_world_fact"
    METAMORPHIC_RELATION = "metamorphic_relation"
    LLM_INFERENCE = "llm_inference"


class OracleProvenance(ContractModel):
    oracle_id: str = Field(min_length=1, max_length=200)
    strength: OracleStrength
    source: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    supports_hard_failure: bool = False

    @model_validator(mode="after")
    def protect_hard_failures(self) -> "OracleProvenance":
        if self.supports_hard_failure and self.strength == OracleStrength.LLM_INFERENCE:
            raise ValueError("LLM-inferred oracles cannot directly support hard failures")
        if self.supports_hard_failure and self.confidence < 0.8:
            raise ValueError("low-confidence oracles cannot directly support hard failures")
        return self


class PostconditionOperator(str, Enum):
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    UNCHANGED = "unchanged"


class StatePostcondition(ContractModel):
    criterion_id: str = Field(min_length=1, max_length=200)
    path: str = Field(min_length=1, max_length=1_000)
    operator: PostconditionOperator
    expected: JsonValue = None
    required: bool = True
    oracle_ids: tuple[str, ...] = Field(min_length=1)


class ToolBehaviorConstraint(ContractModel):
    criterion_id: str = Field(min_length=1, max_length=200)
    tool_name: str = Field(min_length=1, max_length=200)
    arguments_match: JsonObject = Field(default_factory=dict)
    min_calls: int = Field(default=0, ge=0)
    max_calls: int | None = Field(default=None, ge=0)
    confirmation_required_before_call: bool = False
    oracle_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_call_range(self) -> "ToolBehaviorConstraint":
        if self.max_calls is not None and self.max_calls < self.min_calls:
            raise ValueError("max_calls cannot be below min_calls")
        return self


class TrajectoryConstraintKind(str, Enum):
    CONFIRMATION_BEFORE_TOOL = "confirmation_before_tool"
    NO_DUPLICATE_SIDE_EFFECT = "no_duplicate_side_effect"
    MAX_RETRIES = "max_retries"
    ORDERING = "ordering"
    NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT = "no_retry_after_ambiguous_timeout"
    MAX_MODEL_TURNS = "max_model_turns"
    MAX_TOOL_CALLS = "max_tool_calls"
    # Handoff kinds evaluate recorded HANDOFF canonical events.  New enum
    # members leave every existing scenario dump, and therefore every existing
    # fingerprint, byte-identical.
    REQUIRED_HANDOFF = "required_handoff"
    FORBIDDEN_HANDOFF = "forbidden_handoff"
    MAX_HANDOFFS = "max_handoffs"
    NO_HANDOFF_LOOP = "no_handoff_loop"
    HANDOFF_BEFORE_TOOL = "handoff_before_tool"
    OTHER = "other"


class TrajectoryConstraint(ContractModel):
    criterion_id: str = Field(min_length=1, max_length=200)
    kind: TrajectoryConstraintKind
    description: str = Field(min_length=1, max_length=4_000)
    parameters: JsonObject = Field(default_factory=dict)
    required: bool = True
    oracle_ids: tuple[str, ...] = Field(min_length=1)


class OutputCriterionKind(str, Enum):
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    JSON_SCHEMA = "json_schema"
    NO_FABRICATED_SUCCESS = "no_fabricated_success"
    ACKNOWLEDGES_TOOL_ERROR = "acknowledges_tool_error"
    GROUNDED_IN_CONTROLLED_EVIDENCE = "grounded_in_controlled_evidence"
    OTHER = "other"


class OutputCriterion(ContractModel):
    criterion_id: str = Field(min_length=1, max_length=200)
    kind: OutputCriterionKind
    description: str = Field(min_length=1, max_length=4_000)
    parameters: JsonObject = Field(default_factory=dict)
    required: bool = True
    oracle_ids: tuple[str, ...] = Field(min_length=1)


class ResourceBudgets(ContractModel):
    wall_clock_seconds: float = Field(default=30.0, gt=0.0)
    max_model_turns: int = Field(default=8, ge=1)
    max_tool_calls: int = Field(default=20, ge=1)
    token_budget: int | None = Field(default=None, ge=1)
    cost_budget_usd: float | None = Field(default=None, ge=0.0)


class Scenario(ContractModel):
    """A deterministic case with constraints, not a single golden trajectory."""

    contract_version: Literal["agentcheck.scenario.v1"] = SCENARIO_CONTRACT_VERSION
    scenario_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    description: str | None = Field(default=None, max_length=8_000)
    conversation_turns: tuple[ConversationTurn, ...] = Field(min_length=1)
    # Scripted replies the runtime withholds until the agent has answered, one
    # per completed execution stage. ``conversation_turns`` keeps its meaning
    # exactly: everything the agent may see before it has said anything. A
    # policy that discloses consequences and only then asks for confirmation
    # cannot be expressed by seeding that confirmation up front -- a correct
    # agent ignores it and asks again, and the case ends before the action it
    # was written to observe.
    followup_turns: tuple[ConversationTurn, ...] = Field(
        default=(), max_length=MAX_FOLLOWUP_TURNS
    )
    initial_world_state: JsonObject = Field(default_factory=dict)
    tool_fixtures: tuple[ToolFixture, ...] = ()
    injected_faults: tuple[InjectedFault, ...] = ()
    expected_postconditions: tuple[StatePostcondition, ...] = ()
    required_tool_behavior: tuple[ToolBehaviorConstraint, ...] = ()
    allowed_tool_behavior: tuple[ToolBehaviorConstraint, ...] = ()
    forbidden_tool_behavior: tuple[ToolBehaviorConstraint, ...] = ()
    trajectory_constraints: tuple[TrajectoryConstraint, ...] = ()
    output_criteria: tuple[OutputCriterion, ...] = ()
    resource_budgets: ResourceBudgets = Field(default_factory=ResourceBudgets)
    dimension_tags: tuple[str, ...] = Field(min_length=1)
    oracle_provenance: tuple[OracleProvenance, ...] = Field(min_length=1)
    generation_seed: int = Field(ge=0)
    fingerprint: str = ""

    @model_serializer(mode="wrap")
    def omit_idle_followup_turns(self, serializer: Any) -> dict[str, Any]:
        """Keep a scenario that declares no follow-up byte-identical to a v1 dump.

        The fingerprint is a hash of this document, so a field that always
        appeared would move every existing scenario, suite, and manifest for a
        feature they do not use.
        """

        data = serializer(self)
        if not data.get("followup_turns"):
            data.pop("followup_turns", None)
        return data

    @field_validator("dimension_tags")
    @classmethod
    def normalize_dimension_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not tag for tag in value):
            raise ValueError("dimension tags must be non-empty")
        return tuple(sorted(set(value)))

    def expected_fingerprint(self) -> str:
        # Display identity is deliberately excluded so structural duplicates with
        # different generated IDs or titles receive the same fingerprint.
        payload = self.model_dump(
            mode="json",
            exclude={"fingerprint", "scenario_id", "title", "description"},
        )
        return canonical_hash(payload)

    @model_validator(mode="after")
    def validate_scenario_links_and_fingerprint(self) -> "Scenario":
        oracle_ids = [oracle.oracle_id for oracle in self.oracle_provenance]
        if len(oracle_ids) != len(set(oracle_ids)):
            raise ValueError("oracle IDs must be unique")

        referenced_oracles: set[str] = set()
        for postcondition in self.expected_postconditions:
            referenced_oracles.update(postcondition.oracle_ids)
        for required_behavior in self.required_tool_behavior:
            referenced_oracles.update(required_behavior.oracle_ids)
        for allowed_behavior in self.allowed_tool_behavior:
            referenced_oracles.update(allowed_behavior.oracle_ids)
        for forbidden_behavior in self.forbidden_tool_behavior:
            referenced_oracles.update(forbidden_behavior.oracle_ids)
        for trajectory_constraint in self.trajectory_constraints:
            referenced_oracles.update(trajectory_constraint.oracle_ids)
        for output_criterion in self.output_criteria:
            referenced_oracles.update(output_criterion.oracle_ids)
        unknown = referenced_oracles.difference(oracle_ids)
        if unknown:
            raise ValueError(f"scenario criteria reference unknown oracles: {sorted(unknown)}")

        if self.followup_turns:
            non_user = sorted(
                {
                    turn.role.value
                    for turn in self.followup_turns
                    if turn.role != ConversationRole.USER
                }
            )
            if non_user:
                # An assistant turn injected mid-run would be AgentCheck writing
                # the agent's side of the transcript the oracle then scores.
                raise ValueError(
                    "follow-up turns must be user turns; found "
                    f"{', '.join(non_user)}"
                )
            turn_ids = [
                turn.turn_id
                for turn in (*self.conversation_turns, *self.followup_turns)
            ]
            if len(turn_ids) != len(set(turn_ids)):
                raise ValueError("conversation and follow-up turn IDs must be unique")

        for constraint in self.required_tool_behavior:
            if constraint.min_calls < 1:
                raise ValueError("required tool behavior must require at least one call")
        for constraint in self.forbidden_tool_behavior:
            if constraint.min_calls != 0 or constraint.max_calls != 0:
                raise ValueError("forbidden tool behavior must set min_calls=max_calls=0")

        expected = self.expected_fingerprint()
        if self.fingerprint and not hmac.compare_digest(self.fingerprint, expected):
            raise ValueError("scenario fingerprint does not match its behavioral contents")
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", expected)
        return self
