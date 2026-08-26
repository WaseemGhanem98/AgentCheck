"""Versioned, evidence-carrying description of an agent under test."""

from __future__ import annotations

from enum import Enum
from typing import Generic, Literal, TypeVar

from pydantic import Field, model_validator

from .base import ContractModel, JsonObject, UtcDatetime


AGENT_SPEC_CONTRACT_VERSION: Literal["agentcheck.agent_spec.v1"] = (
    "agentcheck.agent_spec.v1"
)
PropertyT = TypeVar("PropertyT")


class SourceKind(str, Enum):
    FRAMEWORK_METADATA = "framework_metadata"
    RUNTIME_INTROSPECTION = "runtime_introspection"
    STATIC_ANALYSIS = "static_analysis"
    SYSTEM_INSTRUCTION = "system_instruction"
    TOOL_SCHEMA = "tool_schema"
    DECLARED_POLICY = "declared_policy"
    DEVELOPER_CONFIG = "developer_config"
    STATE_CONTRACT = "state_contract"
    HUMAN = "human"
    LLM_INFERENCE = "llm_inference"
    UNKNOWN = "unknown"


class SourceReference(ContractModel):
    kind: SourceKind
    locator: str = Field(min_length=1, max_length=1_000)
    description: str | None = Field(default=None, max_length=2_000)


class SpecEvidence(ContractModel):
    evidence_id: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=4_000)
    locator: str | None = Field(default=None, max_length=1_000)
    excerpt: str | None = Field(default=None, max_length=8_000)


class AgentProperty(ContractModel, Generic[PropertyT]):
    """A value that never loses how, and how confidently, it was obtained."""

    value: PropertyT
    source: SourceReference
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: tuple[SpecEvidence, ...] = Field(min_length=1)
    inferred: bool = False
    authoritative: bool = False

    @model_validator(mode="after")
    def protect_hard_rules_from_weak_inference(self) -> "AgentProperty[PropertyT]":
        if self.source.kind == SourceKind.LLM_INFERENCE and not self.inferred:
            raise ValueError("LLM-derived properties must be marked inferred")
        if self.authoritative and self.source.kind in {
            SourceKind.LLM_INFERENCE,
            SourceKind.UNKNOWN,
        }:
            raise ValueError("LLM-inferred or unknown properties cannot be authoritative")
        if self.authoritative and self.confidence < 0.8:
            raise ValueError("low-confidence properties cannot be authoritative")
        return self


class ActionKind(str, Enum):
    LOOKUP = "lookup"
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    SCHEDULE = "schedule"
    SUMMARIZE = "summarize"
    RETRIEVE = "retrieve"
    SEND = "send"
    OTHER = "other"


class Capability(ContractModel):
    capability_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4_000)
    action_kind: ActionKind
    state_changing: bool = False
    destructive: bool = False


class ToolDefinition(ContractModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=8_000)
    input_schema: JsonObject = Field(default_factory=dict)
    output_schema: JsonObject | None = None
    state_changing: bool = False
    destructive: bool = False
    replaceable: bool = False


class RiskAuthority(str, Enum):
    """Where a tool's ``state_changing``/``destructive`` value came from.

    Ordered by precedence, strongest first. A resolver must never let a lower
    tier silently overrule a higher one, and ``UNKNOWN`` must stay reachable:
    the absence of evidence is not itself evidence of safety.
    """

    DEVELOPER_DECLARED = "developer_declared"
    FRAMEWORK_AUTHORITATIVE = "framework_authoritative"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class RiskAxis(ContractModel):
    """One resolved risk property (``state_changing`` or ``destructive``).

    ``value`` is always a concrete bool because every existing consumer
    (coverage, fault generation, the derived policy pack) already treats an
    absent classification as ``False``. ``authority`` is what keeps that
    ``False`` from being misread as "confirmed safe" when it really means
    "unknown, resolved conservatively" -- callers that care must read
    ``authority`` rather than trust the bool alone.
    """

    value: bool
    authority: RiskAuthority
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def protect_authority_from_weak_evidence(self) -> "RiskAxis":
        if self.authority is RiskAuthority.UNKNOWN and self.value:
            raise ValueError("an unknown axis cannot assert a true value")
        if self.authority is RiskAuthority.INFERRED and self.confidence >= 1.0:
            # A heuristic is never entitled to full certainty; only a developer
            # declaration or authoritative framework metadata may claim one.
            raise ValueError("inferred risk cannot claim full confidence")
        return self


class ToolRiskAssertion(ContractModel):
    """Full provenance for one tool's resolved risk, independent per axis.

    Kept out of :class:`ToolDefinition` deliberately: ``ToolDefinition`` is
    hashed whole into ``spec_id`` (see ``agentcheck/adapters/base.py``), so
    adding provenance fields there would re-identify every existing target the
    moment this shipped, whether or not its resolved risk actually changed.
    This model lives in a sibling spec section instead, so identity only moves
    when the resolved ``state_changing``/``destructive`` bool itself changes.
    """

    tool_name: str = Field(min_length=1, max_length=200)
    state_changing: RiskAxis
    destructive: RiskAxis
    evidence: tuple[SpecEvidence, ...] = Field(min_length=1)
    conflicts: tuple[str, ...] = ()


class ToolPolicy(ContractModel):
    policy_id: str = Field(min_length=1, max_length=200)
    tool_name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4_000)
    confirmation_required: bool | None = None
    idempotent: bool | None = None
    max_retries: int | None = Field(default=None, ge=0)


class Guardrail(ContractModel):
    guardrail_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    stage: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=4_000)


class Workflow(ContractModel):
    workflow_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=4_000)
    steps: tuple[str, ...] = ()


class Policy(ContractModel):
    policy_id: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=8_000)
    version: str | None = Field(default=None, max_length=100)


class IdentitySpec(ContractModel):
    name: AgentProperty[str]
    framework: AgentProperty[str]
    framework_version: AgentProperty[str | None]
    provider: AgentProperty[str | None]
    model: AgentProperty[str | None]


class InterfaceSpec(ContractModel):
    entrypoint: AgentProperty[str]
    input_modalities: AgentProperty[tuple[str, ...]]
    output_modalities: AgentProperty[tuple[str, ...]]
    input_schema: AgentProperty[JsonObject | None]
    output_schema: AgentProperty[JsonObject | None]
    interactive: AgentProperty[bool]


class InstructionsSpec(ContractModel):
    system: AgentProperty[str | None]
    developer: AgentProperty[str | None]


class CapabilitiesSpec(ContractModel):
    items: tuple[AgentProperty[Capability], ...] = ()


class ToolsSpec(ContractModel):
    items: tuple[AgentProperty[ToolDefinition], ...] = ()


class ToolRiskSpec(ContractModel):
    """Per-tool risk provenance, one entry per tool named in ``tools``.

    A sibling of :class:`ToolsSpec` rather than a field on ``ToolDefinition``
    -- see :class:`ToolRiskAssertion` for why identity requires that split.
    """

    items: tuple[ToolRiskAssertion, ...] = ()

    @model_validator(mode="after")
    def unique_tool_names(self) -> "ToolRiskSpec":
        names = [item.tool_name for item in self.items]
        if len(names) != len(set(names)):
            raise ValueError("tool risk assertions must name each tool at most once")
        return self


class ToolPoliciesSpec(ContractModel):
    items: tuple[AgentProperty[ToolPolicy], ...] = ()


class GuardrailsSpec(ContractModel):
    items: tuple[AgentProperty[Guardrail], ...] = ()


class WorkflowsSpec(ContractModel):
    items: tuple[AgentProperty[Workflow], ...] = ()


class PoliciesSpec(ContractModel):
    items: tuple[AgentProperty[Policy], ...] = ()


class RuntimeSpec(ContractModel):
    max_model_turns: AgentProperty[int | None]
    max_tool_calls: AgentProperty[int | None]
    timeout_seconds: AgentProperty[float | None]
    token_budget: AgentProperty[int | None]
    cost_budget_usd: AgentProperty[float | None]


class ObservabilitySpec(ContractModel):
    supported_event_types: AgentProperty[tuple[str, ...]]
    usage_metrics: AgentProperty[tuple[str, ...]]
    provider_request_ids: AgentProperty[bool]
    source_event_links: AgentProperty[bool]


class UnknownProperty(ContractModel):
    path: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=4_000)
    source: SourceReference
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: tuple[SpecEvidence, ...] = Field(min_length=1)


class InspectionProvenance(ContractModel):
    inspector: str = Field(min_length=1, max_length=200)
    inspector_version: str = Field(min_length=1, max_length=100)
    inspected_at: UtcDatetime
    target: str = Field(min_length=1, max_length=2_000)
    git_revision: str | None = Field(default=None, max_length=200)
    sources: tuple[SourceReference, ...] = Field(min_length=1)


class AgentSpec(ContractModel):
    """Provider-neutral inspection result consumed by the evaluation engine."""

    contract_version: Literal["agentcheck.agent_spec.v1"] = AGENT_SPEC_CONTRACT_VERSION
    spec_id: str = Field(min_length=1, max_length=200)
    # The identity this same inspection would have produced under the
    # pre-portable contract, where the absolute entrypoint path was hashed into
    # spec_id. It exists only so an artifact created before portable identity
    # can still be recognized at the location that produced it. It is never a
    # new binding, and it is absent when no distinct legacy identity exists.
    legacy_spec_id: str | None = Field(default=None, max_length=200)
    identity: IdentitySpec
    interface: InterfaceSpec
    instructions: InstructionsSpec
    capabilities: CapabilitiesSpec = Field(default_factory=CapabilitiesSpec)
    tools: ToolsSpec = Field(default_factory=ToolsSpec)
    tool_risk: ToolRiskSpec = Field(default_factory=ToolRiskSpec)
    tool_policies: ToolPoliciesSpec = Field(default_factory=ToolPoliciesSpec)
    guardrails: GuardrailsSpec = Field(default_factory=GuardrailsSpec)
    workflows: WorkflowsSpec = Field(default_factory=WorkflowsSpec)
    policies: PoliciesSpec = Field(default_factory=PoliciesSpec)
    runtime: RuntimeSpec
    observability: ObservabilitySpec
    unknowns: tuple[UnknownProperty, ...] = ()
    provenance: InspectionProvenance
