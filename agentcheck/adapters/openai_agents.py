"""Fail-closed OpenAI Agents SDK adapter for the Phase 1 local MVP.

Only ordinary ``FunctionTool`` instances are accepted.  Each accepted tool is
reconstructed with an AgentCheck-owned invoker; the original callable and every
advanced callback are deliberately left behind on the source agent.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import functools
import hashlib
import inspect
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

from jsonschema import SchemaError  # type: ignore[import-untyped]
from pydantic import BaseModel, TypeAdapter

from agentcheck.domain import (
    AgentProperty,
    AgentSpec,
    CanonicalEvent,
    CanonicalEventType,
    CanonicalRun,
    CapabilitiesSpec,
    ConversationRole,
    ConversationTurn,
    Guardrail,
    GuardrailsSpec,
    IdentitySpec,
    InspectionProvenance,
    InstructionsSpec,
    InterfaceSpec,
    ObservabilitySpec,
    PoliciesSpec,
    RunTermination,
    RuntimeSpec,
    SourceKind,
    SourceReference,
    SpecEvidence,
    StateTransition,
    ToolAttempt,
    ToolDefinition,
    ToolError,
    ToolOutcome,
    ToolOutcomeStatus,
    ToolPoliciesSpec,
    ToolRiskAssertion,
    ToolRiskSpec,
    ToolsSpec,
    UnknownProperty,
    UsageMetrics,
    WorkflowsSpec,
    canonical_hash,
    utc_now,
)
from agentcheck.inspect.capabilities import extract_capabilities
from agentcheck.inspect.risk_authority import declared_risk_for, resolve_tool_risk

if TYPE_CHECKING:
    from agentcheck.config import ToolRiskDeclaration
from agentcheck.schema_safety import UnsafeSchemaReference, offline_validator
from agentcheck.runner.budgets import BudgetExceeded
from agentcheck.runner.launch_barrier import LaunchBarrier
from agentcheck.runner.network_guard import denied_destinations
from agentcheck.runner.tool_gateway import CallReservation

from .base import (
    portable_identity,
    AdapterDependencyError,
    AdapterRuntimeError,
    EventSinkProtocol,
    FrameworkAdapter,
    PreflightReport,
    PreparedTarget,
    SupportIssue,
    ToolGatewayProtocol,
    missing_extra_message,
)
from .openai_handoff_effects import (
    AgentCheckRunContext,
    ContextAssignment,
    analyze_on_handoff_callback,
    apply_context_assignments,
    encode_context_assignments,
)

_SDK_IMPORT_ERROR: ImportError | None = None
try:
    from agents import Agent, FunctionTool, Runner
    from agents.agent_output import AgentOutputSchemaBase
    from agents.exceptions import AgentsException, MaxTurnsExceeded, ModelBehaviorError
    from agents.handoffs import Handoff
    from agents.items import ItemHelpers, ModelResponse
    from agents.lifecycle import RunHooksBase
    from agents.model_settings import ModelSettings
    from agents.run import RunConfig
    from agents.run_config import ToolExecutionConfig
    from agents.tool import set_function_tool_failure_error_function
    from agents.tool_context import ToolContext
    from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage
except ImportError as exc:  # pragma: no cover - exercised in minimal installs
    _SDK_IMPORT_ERROR = exc
    # These names are classes/functions in the installed-extra branch.  The
    # assignments keep the core package importable without that optional extra.
    Agent = cast(Any, None)  # type: ignore[misc]
    FunctionTool = cast(Any, None)  # type: ignore[misc]
    Runner = cast(Any, None)  # type: ignore[misc]
    Handoff = cast(Any, None)  # type: ignore[misc]
    AgentOutputSchemaBase = cast(Any, None)  # type: ignore[misc]
    AgentsException = cast(Any, Exception)  # type: ignore[misc]
    MaxTurnsExceeded = cast(Any, Exception)  # type: ignore[misc]
    ModelBehaviorError = cast(Any, Exception)  # type: ignore[misc]
    ItemHelpers = cast(Any, None)  # type: ignore[misc]
    ModelResponse = cast(Any, None)  # type: ignore[misc]
    RunHooksBase = cast(Any, object)  # type: ignore[misc]
    ModelSettings = cast(Any, None)  # type: ignore[misc]
    RunConfig = cast(Any, None)  # type: ignore[misc]
    ToolExecutionConfig = cast(Any, None)  # type: ignore[misc]
    set_function_tool_failure_error_function = cast(Any, None)
    ToolContext = cast(Any, None)  # type: ignore[misc]
    ResponseFunctionToolCall = cast(Any, None)  # type: ignore[misc]
    ResponseOutputMessage = cast(Any, None)  # type: ignore[misc]


FRAMEWORK_NAME = "openai_agents"
INSPECTOR_VERSION = "0.1.0"
SUPPORTED_SDK_MINOR = (0, 20)


def _require_sdk() -> None:
    if _SDK_IMPORT_ERROR is not None:
        raise AdapterDependencyError(
            missing_extra_message("OpenAI Agents SDK support", "openai-agents")
        ) from _SDK_IMPORT_ERROR


def _sdk_version() -> str | None:
    try:
        return importlib_metadata.version("openai-agents")
    except importlib_metadata.PackageNotFoundError:
        return None


def _supported_sdk_version(version: str | None) -> bool:
    if version is None:
        return False
    numeric = version.split("+", 1)[0].split("-", 1)[0].split(".")
    try:
        return tuple(int(part) for part in numeric[:2]) == SUPPORTED_SDK_MINOR
    except ValueError:
        return False


# The SDK dispatches function-tool invokers as concurrent asyncio tasks under
# this cap (`asyncio.create_task` + `asyncio.wait`), never real OS threads --
# confirmed by reading agents/run_internal/tool_execution.py for the pinned
# SDK version. A model response rarely emits more than a handful of tool
# calls at once, and ToolGateway.commit is forced back into decision order by
# LaunchBarrier regardless of this cap, so raising it further would not
# change gateway behaviour -- only how many invokers may have pre-commit work
# (argument parsing, attempt recording) in flight at once. 8 is a small,
# explicit bound rather than the unbounded default (`None`), chosen to
# exercise genuine multi-task dispatch without inviting an unbounded number
# of concurrently pending tasks for a single model response.
_MAX_FUNCTION_TOOL_CONCURRENCY = 8
_MAX_REACHABLE_AGENTS = 16
_HANDOFF_FACTORY_MODULE = "agents.handoffs"
_HANDOFF_FACTORY_WRAPPER = "_invoke_handoff_with_redaction"
_HANDOFF_FACTORY_INNER = "handoff.<locals>._invoke_handoff_impl"
_HANDOFF_CLOSURE_KEYS = frozenset({"agent", "input_type", "on_handoff", "type_adapter"})


def _empty_handoff_payload_schema() -> dict[str, Any]:
    """The strict payload schema the SDK factory derives for payload-free handoffs."""

    return {
        "additionalProperties": False,
        "type": "object",
        "properties": {},
        "required": [],
    }


@dataclass(frozen=True)
class _HandoffEdge:
    """One statically analyzed handoff entry on a reachable agent."""

    location: str
    from_agent: Any
    destination: Any | None
    tool_name: str | None
    tool_description: str | None
    input_json_schema: dict[str, Any] | None
    strict_json_schema: bool
    issues: tuple[SupportIssue, ...]
    context_assignments: tuple[ContextAssignment, ...] = ()


@dataclass(frozen=True)
class _AgentGraph:
    """Reachable agents in deterministic BFS order plus analyzed handoff edges."""

    agents: tuple[Any, ...]
    locations: tuple[str, ...]
    edges: tuple[_HandoffEdge, ...]
    issues: tuple[SupportIssue, ...]

    @property
    def has_handoffs(self) -> bool:
        return bool(self.edges)


def _unrecognized_handoff(location: str, reason: str) -> SupportIssue:
    return SupportIssue(
        code="unrecognized_handoff",
        message=(
            f"{reason} Only plain agents.Agent entries and unmodified openai-agents "
            "0.20 handoff() factory products can be proven safe."
        ),
        location=location,
    )


def _resolve_factory_destination(
    entry: Any, location: str
) -> tuple[Any | None, list[SupportIssue], tuple[ContextAssignment, ...]]:
    """Prove a factory-built handoff's static destination without executing it.

    The closure is introspected, never called: ``inspect.getclosurevars`` reads
    cell contents only, and the ``_agent_ref`` weakref dereference runs no target
    code.  Any shape mismatch fails closed as ``unrecognized_handoff``.
    ``on_handoff`` is analyzed statically; the original callable is never invoked.
    """

    invoker = entry.on_invoke_handoff
    if (
        not isinstance(invoker, functools.partial)
        or invoker.keywords
        or len(invoker.args) != 1
    ):
        return None, [
            _unrecognized_handoff(
                location, "The handoff invocation closure is not the SDK factory shape."
            )
        ], ()
    wrapper = invoker.func
    if (
        getattr(wrapper, "__module__", None) != _HANDOFF_FACTORY_MODULE
        or getattr(wrapper, "__qualname__", None) != _HANDOFF_FACTORY_WRAPPER
    ):
        return None, [
            _unrecognized_handoff(
                location, "The handoff invocation wrapper is not the SDK factory wrapper."
            )
        ], ()
    inner = invoker.args[0]
    if (
        getattr(inner, "__module__", None) != _HANDOFF_FACTORY_MODULE
        or getattr(inner, "__qualname__", None) != _HANDOFF_FACTORY_INNER
        or not inspect.iscoroutinefunction(inner)
    ):
        return None, [
            _unrecognized_handoff(
                location, "The handoff routing closure is not the SDK factory closure."
            )
        ], ()
    try:
        nonlocals = inspect.getclosurevars(inner).nonlocals
    except (TypeError, ValueError):
        return None, [
            _unrecognized_handoff(location, "The handoff routing closure cannot be inspected.")
        ], ()
    if not _HANDOFF_CLOSURE_KEYS.issubset(nonlocals):
        return None, [
            _unrecognized_handoff(
                location, "The handoff routing closure is missing expected factory state."
            )
        ], ()

    issues: list[SupportIssue] = []
    assignments: tuple[ContextAssignment, ...] = ()
    if nonlocals["on_handoff"] is not None:
        analysis = analyze_on_handoff_callback(
            nonlocals["on_handoff"], location=f"{location}.on_handoff"
        )
        if analysis.issue is not None:
            issues.append(analysis.issue)
        else:
            assignments = analysis.assignments
    if nonlocals["input_type"] is not None or nonlocals["type_adapter"] is not None:
        issues.append(
            SupportIssue(
                code="handoff_input_schema",
                message=(
                    "The handoff validates a typed tool-call payload; Phase 5A "
                    "supports payload-free routing handoffs only."
                ),
                location=f"{location}.input_type",
            )
        )

    destination = nonlocals["agent"]
    if type(destination) is not Agent:
        return None, [
            *issues,
            SupportIssue(
                code="unsupported_agent_type",
                message=(
                    "Handoff destinations must be exact agents.Agent instances; "
                    f"found {type(destination).__name__}."
                ),
                location=location,
            ),
        ], ()
    reference = entry._agent_ref
    referenced = reference() if reference is not None else None
    if referenced is not destination:
        return None, [
            *issues,
            _unrecognized_handoff(
                location, "The handoff agent reference does not match its routing closure."
            ),
        ], ()
    if isinstance(entry.agent_name, str) and entry.agent_name != destination.name:
        return None, [
            *issues,
            _unrecognized_handoff(
                location, "The handoff agent_name does not match its destination agent."
            ),
        ], ()
    return destination, issues, assignments


def _analyze_handoff_entry(entry: Any, from_agent: Any, location: str) -> _HandoffEdge:
    if type(entry) is Agent:
        return _HandoffEdge(
            location=location,
            from_agent=from_agent,
            destination=entry,
            tool_name=Handoff.default_tool_name(entry),
            tool_description=Handoff.default_tool_description(entry),
            input_json_schema=None,
            strict_json_schema=True,
            issues=(),
        )
    if isinstance(entry, Agent):
        return _HandoffEdge(
            location=location,
            from_agent=from_agent,
            destination=None,
            tool_name=None,
            tool_description=None,
            input_json_schema=None,
            strict_json_schema=True,
            issues=(
                SupportIssue(
                    code="unsupported_agent_type",
                    message=(
                        "Handoff destinations must be exact agents.Agent instances; "
                        f"found {type(entry).__name__}."
                    ),
                    location=location,
                ),
            ),
        )
    if not isinstance(entry, Handoff):
        return _HandoffEdge(
            location=location,
            from_agent=from_agent,
            destination=None,
            tool_name=None,
            tool_description=None,
            input_json_schema=None,
            strict_json_schema=True,
            issues=(
                _unrecognized_handoff(
                    location,
                    f"Handoff entry is {type(entry).__name__}, not an Agent or SDK Handoff.",
                ),
            ),
        )

    issues: list[SupportIssue] = []
    if entry.input_filter is not None:
        issues.append(
            SupportIssue(
                code="handoff_input_filter",
                message=(
                    "The handoff rewrites conversation history through an input "
                    "filter callback, which AgentCheck never executes."
                ),
                location=f"{location}.input_filter",
            )
        )
    if entry.is_enabled is not True:
        issues.append(
            SupportIssue(
                code="dynamic_handoff_enablement",
                message=(
                    "The handoff uses non-default enablement; callable or disabled "
                    "is_enabled values are unsupported in Phase 5A."
                ),
                location=f"{location}.is_enabled",
            )
        )
    if entry.nest_handoff_history not in (None, False):
        issues.append(
            SupportIssue(
                code="handoff_history_override",
                message=(
                    "The handoff overrides the default conversation-history "
                    "behavior, which is unsupported in Phase 5A."
                ),
                location=f"{location}.nest_handoff_history",
            )
        )
    schema = entry.input_json_schema
    schema_declares_payload = not isinstance(schema, Mapping) or bool(
        schema.get("properties") or schema.get("required")
    )

    destination, closure_issues, context_assignments = _resolve_factory_destination(
        entry, location
    )
    issues.extend(closure_issues)
    if schema_declares_payload and not any(
        issue.code == "handoff_input_schema" for issue in issues
    ):
        issues.append(
            SupportIssue(
                code="handoff_input_schema",
                message=(
                    "The handoff exposes a tool-call payload schema to the model; "
                    "Phase 5A supports payload-free routing handoffs only."
                ),
                location=f"{location}.input_json_schema",
            )
        )

    tool_name = entry.tool_name if isinstance(entry.tool_name, str) and entry.tool_name else None
    if tool_name is None:
        issues.append(
            _unrecognized_handoff(location, "The handoff has no usable tool name.")
        )
    return _HandoffEdge(
        location=location,
        from_agent=from_agent,
        destination=destination,
        tool_name=tool_name,
        tool_description=entry.tool_description if isinstance(entry.tool_description, str) else None,
        input_json_schema=dict(schema) if isinstance(schema, Mapping) else None,
        strict_json_schema=bool(entry.strict_json_schema),
        issues=tuple(issues),
        context_assignments=context_assignments,
    )


def _reachable_graph(target: Any) -> _AgentGraph:
    """Collect the bounded, cycle-safe agent graph without executing target code."""

    agents: list[Any] = [target]
    locations: list[str] = ["agent"]
    visited: set[int] = {id(target)}
    edges: list[_HandoffEdge] = []
    issues: list[SupportIssue] = []
    index = 0
    truncated = False
    while index < len(agents):
        current = agents[index]
        current_location = locations[index]
        index += 1
        entries = getattr(current, "handoffs", None) or ()
        for position, entry in enumerate(entries):
            location = f"{current_location}.handoffs[{position}]"
            edge = _analyze_handoff_entry(entry, current, location)
            edges.append(edge)
            destination = edge.destination
            if destination is None or id(destination) in visited:
                continue
            if len(agents) >= _MAX_REACHABLE_AGENTS:
                if not truncated:
                    truncated = True
                    issues.append(
                        SupportIssue(
                            code="handoff_graph_too_large",
                            message=(
                                "The reachable agent graph exceeds the Phase 5A "
                                f"bound of {_MAX_REACHABLE_AGENTS} agents."
                            ),
                            location=location,
                        )
                    )
                continue
            visited.add(id(destination))
            agents.append(destination)
            locations.append(f"{location}.agent")
    return _AgentGraph(
        agents=tuple(agents),
        locations=tuple(locations),
        edges=tuple(edges),
        issues=tuple(issues),
    )


def _edges_by_agent(graph: _AgentGraph) -> dict[int, list[_HandoffEdge]]:
    grouped: dict[int, list[_HandoffEdge]] = {}
    for edge in graph.edges:
        grouped.setdefault(id(edge.from_agent), []).append(edge)
    return grouped


def _json_value(value: Any) -> Any:
    """Detach a value into strict JSON without leaking Python object reprs."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    elif isinstance(value, Enum):
        value = value.value
    elif isinstance(value, datetime):
        value = value.isoformat()
    elif isinstance(value, Mapping):
        value = {str(key): _json_value(item) for key, item in value.items()}
    elif isinstance(value, tuple | list | set | frozenset):
        value = [_json_value(item) for item in value]

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        return {"type": type(value).__name__, "unavailable": True}


def _json_object(value: Any) -> dict[str, Any]:
    detached = _json_value(value)
    return detached if isinstance(detached, dict) else {}


def _evidence_id(locator: str) -> str:
    digest = hashlib.sha256(locator.encode("utf-8")).hexdigest()[:20]
    return f"evidence-{digest}"


def _property(
    value: Any,
    *,
    locator: str,
    summary: str,
    kind: SourceKind = SourceKind.RUNTIME_INTROSPECTION,
    confidence: float = 1.0,
    inferred: bool = False,
    authoritative: bool = True,
) -> AgentProperty[Any]:
    source = SourceReference(kind=kind, locator=locator)
    evidence = SpecEvidence(
        evidence_id=_evidence_id(locator),
        summary=summary,
        locator=locator,
    )
    return AgentProperty(
        value=value,
        source=source,
        confidence=confidence,
        evidence=(evidence,),
        inferred=inferred,
        authoritative=authoritative,
    )


def _unknown_property(value: Any, *, locator: str, summary: str) -> AgentProperty[Any]:
    return _property(
        value,
        locator=locator,
        summary=summary,
        kind=SourceKind.UNKNOWN,
        confidence=0.0,
        authoritative=False,
    )


def _output_schema(agent: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Derive a JSON Schema for ``agent.output_type`` by pure type introspection.

    A custom ``AgentOutputSchemaBase`` instance is deliberately never asked for
    its schema: ``json_schema()``/``validate_json()`` are free-form target-defined
    methods, not declarative type annotations, so calling them would execute
    arbitrary target code merely to discover an output schema. Every other
    ``output_type`` (a plain class, dataclass, TypedDict, or generic alias) is
    resolved with ``pydantic.TypeAdapter``, the same declarative mechanism the
    SDK's own default ``AgentOutputSchema`` uses internally.
    """

    output_type = agent.output_type
    if output_type is None:
        return None, None
    if isinstance(output_type, AgentOutputSchemaBase):
        return None, (
            "output_type is a custom AgentOutputSchemaBase instance; its "
            "json_schema()/validate_json() methods are target-defined code, not "
            "a statically provable schema"
        )
    try:
        schema = TypeAdapter(output_type).json_schema()
        return _json_object(schema), None
    except Exception as exc:
        return None, f"output schema could not be extracted: {type(exc).__name__}"


def _model_identity(
    agent: Any,
) -> tuple[str | None, str | None, bool, bool]:
    model = agent.model
    if isinstance(model, str):
        return "openai", model, False, False
    if model is None:
        return "openai", None, False, True
    model_type = type(model)
    module_root = model_type.__module__.split(".", 1)[0]
    if module_root in {"agents", "openai"}:
        return "openai", model_type.__name__, False, False
    # A local Model implementation may proxy any provider or none at all. Its
    # Python module name is not provider evidence.
    return None, model_type.__name__, True, False


def _explain_failure(exc: BaseException) -> str:
    """Attribute a run failure to AgentCheck's network denial when that caused it.

    HTTP clients wrap a refused connection in their own transport error, so a
    blocked call surfaces as a bare "Connection error." and the developer cannot
    tell an unreachable provider from AgentCheck deliberately denying egress.
    When the guard actually refused something in this worker, say so and name the
    setting that changes it.
    """

    reason = str(exc)[:4_000]
    destinations = denied_destinations()
    if not destinations:
        return reason
    listed = ", ".join(destinations[:3])
    return (
        f"{reason} AgentCheck denied network access to {listed} during this run. "
        "Evaluation runs with network access disabled so a target cannot cause "
        "external side effects. If this endpoint is meant to be reached, set "
        '"allow_network": true in agentcheck.json.'
    )[:4_000]


def _tool_contract_identity(tool: Any) -> str:
    """Canonical digest of the tool contract AgentCheck actually evaluates.

    The gateway intercepts by name and validates arguments against the input
    schema; the original handler is never executed. Two agents sharing one
    imported tool therefore present one unambiguous contract, even though they
    are distinct entries in two ``tools`` lists. Only the fields that decide
    interception and scenario meaning are included, so a cosmetic difference
    elsewhere on the SDK object cannot manufacture a false conflict, and a real
    schema or description conflict cannot be hidden.
    """

    return canonical_hash(
        {
            "name": tool.name,
            "description": tool.description or None,
            "input_schema": _json_object(copy.deepcopy(tool.params_json_schema)),
            "output_schema": (
                _json_object(copy.deepcopy(tool.output_json_schema))
                if tool.output_json_schema is not None
                else None
            ),
        }
    )


def _tool_definition(
    tool: Any,
    *,
    declared_tool_risk: "Mapping[str, ToolRiskDeclaration] | None" = None,
) -> tuple[ToolDefinition, ToolRiskAssertion]:
    declared = declared_risk_for(tool.name, declared_tool_risk)
    state_changing, destructive, assertion = resolve_tool_risk(
        tool.name, tool.description or None, declared=declared
    )
    definition = ToolDefinition(
        name=tool.name,
        description=tool.description or None,
        input_schema=_json_object(copy.deepcopy(tool.params_json_schema)),
        output_schema=(
            _json_object(copy.deepcopy(tool.output_json_schema))
            if tool.output_json_schema is not None
            else None
        ),
        state_changing=state_changing,
        destructive=destructive,
        replaceable=True,
    )
    return definition, assertion


def _guardrail_properties(agent: Any, source: str) -> tuple[AgentProperty[Any], ...]:
    values: list[AgentProperty[Any]] = []
    for stage, configured in (
        ("input", agent.input_guardrails),
        ("output", agent.output_guardrails),
    ):
        for index, guardrail in enumerate(configured):
            locator = f"{source}.agent.{stage}_guardrails[{index}]"
            name = type(guardrail).__name__
            values.append(
                _property(
                    Guardrail(
                        guardrail_id=f"{stage}-{index}-{name}",
                        name=name,
                        stage=stage,
                        description=f"Configured OpenAI Agents SDK {stage} guardrail.",
                    ),
                    locator=locator,
                    summary=f"Runtime Agent declares an {stage} guardrail.",
                    authoritative=True,
                )
            )
    return tuple(values)


@dataclass
class _Capture:
    run_id: str
    sink: EventSinkProtocol | None
    events: list[CanonicalEvent] = dataclasses.field(default_factory=list)
    attempts: list[ToolAttempt] = dataclasses.field(default_factory=list)
    outcomes: list[ToolOutcome] = dataclasses.field(default_factory=list)
    responses: list[Any] = dataclasses.field(default_factory=list)
    response_agents: list[str | None] = dataclasses.field(default_factory=list)
    response_event_ids: list[str] = dataclasses.field(default_factory=list)
    request_ids: list[str] = dataclasses.field(default_factory=list)
    transition_links: dict[str, tuple[str, str]] = dataclasses.field(
        default_factory=dict
    )
    agent_by_attempt: dict[str, str | None] = dataclasses.field(default_factory=dict)
    # Populated once per model response, before any of its tool calls are
    # dispatched: which reservation each call_id owns, and the barrier that
    # keeps their commits in decision order under concurrent dispatch. See
    # `_CapturingHooks.on_llm_end` and `_make_safe_invoker`.
    pending_reservations: dict[str, "CallReservation"] = dataclasses.field(
        default_factory=dict
    )
    pending_barrier: "LaunchBarrier | None" = None

    async def event(
        self,
        event_type: CanonicalEventType,
        payload: Mapping[str, Any] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        source_event_ids: Sequence[str] = (),
    ) -> CanonicalEvent:
        sequence = len(self.events)
        event = CanonicalEvent(
            event_id=f"{self.run_id}:event:{sequence:04d}",
            run_id=self.run_id,
            sequence=sequence,
            event_type=event_type,
            timestamp=utc_now(),
            payload=_json_object(dict(payload or {})),
            metadata=_json_object(dict(metadata or {})),
            source_event_ids=tuple(source_event_ids),
        )
        self.events.append(event)
        if self.sink is not None:
            emitted = self.sink.emit(event)
            if inspect.isawaitable(emitted):
                await emitted
        return event

    async def tool_attempt(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        raw_arguments: str,
        call_id: str,
        validation_errors: Sequence[str],
        state_changing: bool,
        destructive: bool,
        agent_name: str | None = None,
    ) -> ToolAttempt:
        source_event_ids: tuple[str, ...] = ()
        for source_event in reversed(self.events):
            if source_event.event_type == CanonicalEventType.MODEL_RESPONSE:
                source_event_ids = (source_event.event_id,)
                break
        event = await self.event(
            CanonicalEventType.TOOL_ATTEMPT,
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "call_id": call_id,
                "agent_name": agent_name,
                "validation_errors": list(validation_errors),
                "raw_arguments_sha256": hashlib.sha256(
                    raw_arguments.encode("utf-8")
                ).hexdigest(),
            },
            source_event_ids=source_event_ids,
        )
        attempt = ToolAttempt(
            attempt_id=f"{self.run_id}:attempt:{len(self.attempts):04d}",
            event_id=event.event_id,
            tool_name=tool_name,
            arguments=_json_object(arguments),
            sequence=len(self.attempts),
            timestamp=event.timestamp,
            state_changing=state_changing,
            destructive=destructive,
        )
        self.attempts.append(attempt)
        self.agent_by_attempt[attempt.attempt_id] = agent_name
        return attempt

    async def handoff_event(
        self,
        *,
        tool_name: str | None,
        from_agent: str | None,
        to_agent: str | None,
        arguments: dict[str, Any],
        ignored: bool,
        call_id: str | None = None,
        source_event_id: str | None = None,
        context_assignments: Sequence[ContextAssignment] = (),
    ) -> CanonicalEvent:
        source_event_ids: tuple[str, ...] = ()
        if source_event_id is not None:
            source_event_ids = (source_event_id,)
        else:
            for source_event in reversed(self.events):
                if source_event.event_type == CanonicalEventType.MODEL_RESPONSE:
                    source_event_ids = (source_event.event_id,)
                    break
        if call_id is None and self.responses and tool_name is not None:
            matches = [
                output
                for output in getattr(self.responses[-1], "output", ())
                if isinstance(output, ResponseFunctionToolCall)
                and output.name == tool_name
            ]
            if matches:
                call_id = matches[0].call_id
        payload: dict[str, Any] = {
            "handoff_tool_name": tool_name,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "arguments": arguments,
            "call_id": call_id,
            "ignored": ignored,
        }
        if context_assignments:
            payload["callback_effect"] = "context_assignment"
            payload["context_assignments"] = encode_context_assignments(
                context_assignments
            )
        return await self.event(
            CanonicalEventType.HANDOFF,
            payload,
            source_event_ids=source_event_ids,
        )

    async def tool_result(
        self,
        *,
        attempt: ToolAttempt,
        status: ToolOutcomeStatus,
        result: Any,
        error: ToolError | None,
        started_at: datetime,
        gateway_outcome_id: str | None = None,
        state_transition_ids: Sequence[str] = (),
        gateway_metadata: Mapping[str, Any] | None = None,
    ) -> ToolOutcome:
        canonical_transition_ids: list[str] = []
        for gateway_transition_id in state_transition_ids:
            link = self.transition_links.get(gateway_transition_id)
            if link is None:
                canonical_transition_id = (
                    f"{self.run_id}:transition:{len(self.transition_links):04d}"
                )
                link = (canonical_transition_id, attempt.attempt_id)
                self.transition_links[gateway_transition_id] = link
            canonical_transition_ids.append(link[0])

        ended_at = utc_now()
        event = await self.event(
            CanonicalEventType.TOOL_RESULT,
            {
                "tool_name": attempt.tool_name,
                "attempt_id": attempt.attempt_id,
                "agent_name": self.agent_by_attempt.get(attempt.attempt_id),
                "status": status.value,
                "result": _json_value(result),
                "error": _json_value(error) if error is not None else None,
            },
            source_event_ids=(attempt.event_id,),
        )
        outcome = ToolOutcome(
            outcome_id=f"{self.run_id}:outcome:{len(self.outcomes):04d}",
            attempt_id=attempt.attempt_id,
            event_id=event.event_id,
            tool_name=attempt.tool_name,
            status=status,
            result=cast(Any, _json_value(result)),
            error=error,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=max(0.0, (ended_at - started_at).total_seconds() * 1_000),
            state_transition_ids=tuple(canonical_transition_ids),
            metadata={
                **_json_object(dict(gateway_metadata or {})),
                **(
                    {"gateway_outcome_id": gateway_outcome_id}
                    if gateway_outcome_id is not None
                    else {}
                ),
            },
        )
        self.outcomes.append(outcome)
        return outcome


# Class bases are evaluated immediately. Without the SDK extra, RunHooksBase is
# `object`, and `object[Any, Any]` raises TypeError on import.
_CapturingHooksBase: Any = RunHooksBase if _SDK_IMPORT_ERROR is None else object


class _CapturingHooks(_CapturingHooksBase):
    def __init__(
        self,
        capture: _Capture,
        budget_tracker: Any = None,
        gateway: "ToolGatewayProtocol | None" = None,
    ):
        self.capture = capture
        self.budget_tracker = budget_tracker
        self.gateway = gateway

    async def on_llm_start(
        self,
        context: Any,
        agent: Any,
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
        del context
        await self.capture.event(
            CanonicalEventType.MODEL_REQUEST,
            {
                "agent_name": agent.name,
                "input_item_count": len(input_items),
                "system_instructions_present": system_prompt is not None,
                "system_instructions_sha256": (
                    hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
                    if system_prompt is not None
                    else None
                ),
            },
        )
        consume_model_turn = getattr(
            self.budget_tracker, "consume_model_turn", None
        )
        if callable(consume_model_turn):
            consume_model_turn()

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        del context
        self.capture.responses.append(response)
        agent_name = getattr(agent, "name", None)
        self.capture.response_agents.append(
            agent_name if isinstance(agent_name, str) else None
        )
        if response.request_id and response.request_id not in self.capture.request_ids:
            self.capture.request_ids.append(response.request_id)
        response_event = await self.capture.event(
            CanonicalEventType.MODEL_RESPONSE,
            {
                "agent_name": agent.name,
                "response_id": response.response_id,
                "request_id": response.request_id,
                "output_item_count": len(response.output),
                "usage_known": response.raw_usage is not None,
            },
        )
        self.capture.response_event_ids.append(response_event.event_id)
        self._reserve_tool_calls(agent, response)
        for output in response.output:
            if isinstance(output, ResponseOutputMessage):
                text = ItemHelpers.extract_text(output)
                if text is not None:
                    await self.capture.event(
                        CanonicalEventType.ASSISTANT_OUTPUT,
                        {"text": text, "agent_name": agent.name},
                        source_event_ids=(response_event.event_id,),
                    )
        raw_usage = response.raw_usage
        if isinstance(raw_usage, Mapping):
            total_tokens = raw_usage.get("total_tokens")
            if total_tokens is None:
                input_tokens = raw_usage.get(
                    "input_tokens", raw_usage.get("prompt_tokens")
                )
                output_tokens = raw_usage.get(
                    "output_tokens", raw_usage.get("completion_tokens")
                )
                if (
                    isinstance(input_tokens, int)
                    and not isinstance(input_tokens, bool)
                    and isinstance(output_tokens, int)
                    and not isinstance(output_tokens, bool)
                ):
                    total_tokens = input_tokens + output_tokens
            add_tokens = getattr(self.budget_tracker, "add_tokens", None)
            if callable(add_tokens):
                add_tokens(
                    total_tokens
                    if isinstance(total_tokens, int)
                    and not isinstance(total_tokens, bool)
                    else None
                )
            add_cost = getattr(self.budget_tracker, "add_cost", None)
            if callable(add_cost):
                raw_cost = raw_usage.get("cost_usd")
                add_cost(
                    raw_cost
                    if isinstance(raw_cost, int | float)
                    and not isinstance(raw_cost, bool)
                    else None
                )

    def _reserve_tool_calls(self, agent: Any, response: Any) -> None:
        """Deterministically plan this response's tool calls before dispatch.

        Called synchronously from ``on_llm_end``, before the SDK creates any
        task to run an individual tool invoker -- so which reservation each
        call gets can never depend on how the SDK schedules those tasks
        afterwards, however many it runs concurrently. A gateway that does
        not support batch planning (anything implementing only the minimal
        ``ToolGatewayProtocol``) is left alone: invokers fall back to calling
        it directly, exactly as they always have.
        """

        gateway = self.gateway
        if not hasattr(gateway, "plan_batch"):
            return
        function_calls = [
            output
            for output in response.output
            if isinstance(output, ResponseFunctionToolCall)
        ]
        if not function_calls:
            return
        schema_by_tool: dict[str, dict[str, Any]] = {
            tool.name: tool.params_json_schema
            for tool in getattr(agent, "tools", ())
            if isinstance(tool, FunctionTool)
        }
        plannable: list[tuple[str, str, dict[str, Any]]] = []
        for call in function_calls:
            call_id = str(getattr(call, "call_id", "") or "")
            schema = schema_by_tool.get(call.name)
            if not call_id or schema is None:
                # No schema means this call is not one of the reconstructed
                # tools this adapter knows about (or the agent's tool list
                # could not be read here); leave it to the invoker's own
                # fallback path rather than guessing.
                continue
            arguments, validation_errors = _parse_arguments(call.arguments, schema)
            if validation_errors:
                # The invoker will take its own malformed-argument path
                # without ever reaching the gateway; planning a reservation
                # for it would consume a fixture/budget slot for a call that
                # never commits, which could steal that slot from a sibling
                # call actually reaching the gateway.
                continue
            plannable.append((call_id, call.name, arguments))
        if not plannable:
            return
        reservations = cast(Any, gateway).plan_batch(
            [(name, arguments) for _, name, arguments in plannable]
        )
        self.capture.pending_reservations = {
            call_id: reservation
            for (call_id, _, _), reservation in zip(plannable, reservations)
        }
        self.capture.pending_barrier = LaunchBarrier(len(reservations))


def _schema_validation_errors(schema: Mapping[str, Any], arguments: Any) -> list[str]:
    try:
        validator = offline_validator(schema)
    except (SchemaError, UnsafeSchemaReference) as exc:
        return [f"invalid or unsafe tool schema: {exc}"]
    errors = sorted(
        validator.iter_errors(arguments), key=lambda error: list(error.path)
    )
    messages: list[str] = []
    for error in errors:
        path = ".".join(str(item) for item in error.absolute_path) or "$"
        messages.append(f"{path}: {error.message}")
    return messages


def _parse_arguments(
    raw_arguments: str, schema: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    try:
        parsed = json.loads(
            raw_arguments,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return {}, [f"invalid JSON: {exc}"]
    if not isinstance(parsed, dict):
        return {}, ["$: tool arguments must be a JSON object"]
    return parsed, _schema_validation_errors(schema, parsed)


def _tool_error_from_gateway(error: Any, *, fallback_code: str) -> ToolError | None:
    if error is None:
        return None
    if isinstance(error, ToolError):
        return error
    if isinstance(error, Mapping):
        return ToolError(
            code=str(error.get("code") or fallback_code),
            message=str(error.get("message") or "Simulated tool failure"),
            retryable=(
                error.get("retryable")
                if isinstance(error.get("retryable"), bool)
                else None
            ),
            details=_json_object(error.get("details", {})),
        )
    return ToolError(code=fallback_code, message=str(error)[:4_000])


def _gateway_result_parts(
    value: Any,
) -> tuple[
    ToolOutcomeStatus,
    Any,
    ToolError | None,
    str | None,
    tuple[str, ...],
    dict[str, Any],
]:
    if isinstance(value, ToolOutcome):
        return (
            value.status,
            value.result,
            value.error,
            value.outcome_id,
            value.state_transition_ids,
            {
                **value.metadata,
                **(
                    {"gateway_latency_ms": value.latency_ms}
                    if value.latency_ms is not None
                    else {}
                ),
            },
        )

    raw_status = getattr(value, "status", None)
    result = getattr(value, "result", value)
    raw_error = getattr(value, "error", None)
    outcome_id = getattr(value, "outcome_id", None)
    gateway_metadata = _json_object(getattr(value, "metadata", {}))
    transition_ids_value = getattr(value, "state_transition_ids", ())
    transition_ids = (
        tuple(item for item in transition_ids_value if isinstance(item, str))
        if isinstance(transition_ids_value, Sequence)
        and not isinstance(transition_ids_value, str | bytes)
        else ()
    )
    if isinstance(raw_status, Enum):
        raw_status = raw_status.value
    if raw_status is None:
        status = ToolOutcomeStatus.SUCCESS
    else:
        try:
            status = ToolOutcomeStatus(str(raw_status).lower())
        except ValueError:
            status = ToolOutcomeStatus.MALFORMED
            raw_error = {
                "code": "invalid_gateway_outcome",
                "message": f"Unsupported gateway status: {raw_status}",
            }
    error = _tool_error_from_gateway(raw_error, fallback_code=f"tool_{status.value}")
    if (
        status
        in {
            ToolOutcomeStatus.ERROR,
            ToolOutcomeStatus.TIMEOUT,
            ToolOutcomeStatus.BLOCKED,
        }
        and error is None
    ):
        error = ToolError(
            code=f"tool_{status.value}",
            message=f"Simulated tool {status.value}",
        )
    return (
        status,
        result,
        error,
        str(outcome_id) if outcome_id is not None else None,
        transition_ids,
        gateway_metadata,
    )


def _model_visible_tool_output(
    status: ToolOutcomeStatus,
    result: Any,
    error: ToolError | None,
) -> str:
    if error is None and status not in {
        ToolOutcomeStatus.ERROR,
        ToolOutcomeStatus.TIMEOUT,
        ToolOutcomeStatus.BLOCKED,
    }:
        if isinstance(result, str):
            return result
        return json.dumps(
            _json_value(result),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    payload = {
        "status": status.value,
        "result": _json_value(result),
        "error": _json_value(error) if error is not None else None,
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _consume_adapter_tool_budget(gateway: ToolGatewayProtocol) -> ToolError | None:
    """Count attempts that are rejected before normal gateway dispatch."""

    tracker = getattr(gateway, "budgets", None)
    consume = getattr(tracker, "consume_tool_call", None)
    if not callable(consume):
        return None
    try:
        consume()
    except BudgetExceeded as exc:
        return ToolError(
            code=f"{exc.resource}_budget_exceeded",
            message=str(exc),
            retryable=False,
            details={"limit": exc.limit, "observed": exc.observed},
        )
    return None


async def _invoke_gateway(
    gateway: ToolGatewayProtocol,
    tool_name: str,
    arguments: dict[str, Any],
    world_state: Any,
) -> Any:
    result = gateway.invoke(tool_name, arguments, world_state)
    if inspect.isawaitable(result):
        return await result
    return result


async def _invoke_reserved_or_direct(
    *,
    gateway: ToolGatewayProtocol,
    capture: "_Capture",
    call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    world_state: Any,
) -> Any:
    """Commit this call's pre-planned reservation, in decision order, if one
    exists; otherwise fall back to invoking the gateway directly.

    The fallback covers a gateway that does not support batch planning, and
    any call `_reserve_tool_calls` deliberately left unplanned (an unknown
    schema, or arguments that will take the invoker's own malformed-argument
    path without reaching the gateway at all). Popping the reservation makes
    each one committable at most once.
    """

    reservation = capture.pending_reservations.pop(call_id, None)
    barrier = capture.pending_barrier
    if reservation is not None and barrier is not None and hasattr(gateway, "commit"):
        return await barrier.run_in_order(
            reservation.rank,
            lambda: cast(Any, gateway).commit(reservation, world_state),
        )
    return await _invoke_gateway(gateway, tool_name, arguments, world_state)


def _make_safe_invoker(
    *,
    tool_name: str,
    schema: dict[str, Any],
    gateway: ToolGatewayProtocol,
    world_state: Any,
    capture_holder: dict[str, _Capture | None],
    state_changing: bool,
    destructive: bool,
) -> Any:
    async def invoke(context: Any, raw_arguments: str) -> str:
        capture = capture_holder.get("capture")
        if capture is None:
            raise RuntimeError(
                "prepared AgentCheck target was invoked outside adapter.run()"
            )

        arguments, validation_errors = _parse_arguments(raw_arguments, schema)
        call_id = str(getattr(context, "tool_call_id", "") or "unknown")
        active_agent = getattr(getattr(context, "agent", None), "name", None)
        attempt = await capture.tool_attempt(
            tool_name=tool_name,
            arguments=arguments,
            raw_arguments=raw_arguments,
            call_id=call_id,
            validation_errors=validation_errors,
            state_changing=state_changing,
            destructive=destructive,
            agent_name=active_agent if isinstance(active_agent, str) else None,
        )
        started_at = utc_now()
        if validation_errors:
            budget_error = _consume_adapter_tool_budget(gateway)
            if budget_error is not None:
                await capture.tool_result(
                    attempt=attempt,
                    status=ToolOutcomeStatus.BLOCKED,
                    result=None,
                    error=budget_error,
                    started_at=started_at,
                )
                return _model_visible_tool_output(
                    ToolOutcomeStatus.BLOCKED, None, budget_error
                )
            validation_error = ToolError(
                code="invalid_tool_arguments",
                message="; ".join(validation_errors)[:4_000],
                retryable=True,
            )
            await capture.tool_result(
                attempt=attempt,
                status=ToolOutcomeStatus.MALFORMED,
                result=None,
                error=validation_error,
                started_at=started_at,
            )
            return _model_visible_tool_output(
                ToolOutcomeStatus.MALFORMED,
                None,
                validation_error,
            )

        try:
            gateway_result = await _invoke_reserved_or_direct(
                gateway=gateway,
                capture=capture,
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
                world_state=world_state,
            )
        except BaseException as exc:
            controlled_outcome = getattr(exc, "outcome", None)
            if isinstance(controlled_outcome, ToolOutcome):
                (
                    status,
                    result,
                    error,
                    gateway_outcome_id,
                    state_transition_ids,
                    gateway_metadata,
                ) = _gateway_result_parts(controlled_outcome)
                await capture.tool_result(
                    attempt=attempt,
                    status=status,
                    result=result,
                    error=error,
                    started_at=started_at,
                    gateway_outcome_id=gateway_outcome_id,
                    state_transition_ids=state_transition_ids,
                    gateway_metadata=gateway_metadata,
                )
                return _model_visible_tool_output(status, result, error)
            await capture.event(
                CanonicalEventType.ERROR,
                {
                    "layer": "tool_gateway",
                    "tool_name": tool_name,
                    "attempt_id": attempt.attempt_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:4_000],
                },
                source_event_ids=(attempt.event_id,),
            )
            raise

        (
            status,
            result,
            error,
            gateway_outcome_id,
            state_transition_ids,
            gateway_metadata,
        ) = _gateway_result_parts(gateway_result)
        await capture.tool_result(
            attempt=attempt,
            status=status,
            result=result,
            error=error,
            started_at=started_at,
            gateway_outcome_id=gateway_outcome_id,
            state_transition_ids=state_transition_ids,
            gateway_metadata=gateway_metadata,
        )
        return _model_visible_tool_output(status, result, error)

    return invoke


def _make_safe_handoff_invoker(
    *,
    tool_name: str,
    from_agent_name: str,
    to_agent_name: str,
    destination: Any,
    capture_holder: dict[str, _Capture | None],
    context_assignments: tuple[ContextAssignment, ...] = (),
) -> Any:
    """Route to the safe destination clone; no original handoff code is reachable."""

    async def invoke(context: Any, input_json: str | None = None) -> Any:
        capture = capture_holder.get("capture")
        if capture is None:
            raise RuntimeError(
                "prepared AgentCheck target was invoked outside adapter.run()"
            )
        if context_assignments:
            apply_context_assignments(
                getattr(context, "context", None), context_assignments
            )
        arguments: dict[str, Any] = {}
        if input_json:
            try:
                decoded = json.loads(input_json)
            except (json.JSONDecodeError, ValueError):
                decoded = None
            if isinstance(decoded, dict):
                arguments = decoded
        await capture.handoff_event(
            tool_name=tool_name,
            from_agent=from_agent_name,
            to_agent=to_agent_name,
            arguments=arguments,
            ignored=False,
            context_assignments=context_assignments,
        )
        return destination

    return invoke


def _sanitized_model_settings(settings: Any) -> Any:
    return dataclasses.replace(
        settings,
        extra_query=None,
        extra_body=None,
        extra_headers=None,
        extra_args=None,
        store=False,
        preserve_raw_usage=True,
    )


def _world_for(prepared: PreparedTarget) -> Any:
    if prepared.world_state is not None:
        return prepared.world_state
    return getattr(prepared.gateway, "world", None)


def _world_snapshot(world: Any) -> dict[str, Any]:
    if world is None:
        return {}
    value = world
    snapshot = getattr(world, "snapshot", None)
    if callable(snapshot):
        value = snapshot()
    elif hasattr(world, "state"):
        value = world.state
    return _json_object(value)


def _state_transitions(
    prepared: PreparedTarget,
    capture: _Capture,
) -> tuple[StateTransition, ...]:
    candidates: Any = getattr(prepared.gateway, "state_transitions", None)
    if candidates is None:
        candidates = getattr(prepared.gateway, "transitions", None)
    world = _world_for(prepared)
    if candidates is None and world is not None:
        candidates = getattr(world, "transitions", None)
    if not isinstance(candidates, Sequence) or isinstance(candidates, str | bytes):
        return ()
    normalized: list[StateTransition] = []
    for item in candidates:
        if not isinstance(item, StateTransition):
            continue
        link = capture.transition_links.get(item.transition_id)
        if link is None:
            continue
        canonical_transition_id, canonical_attempt_id = link
        normalized.append(
            item.model_copy(
                update={
                    "transition_id": canonical_transition_id,
                    "attempt_id": canonical_attempt_id,
                }
            )
        )
    return tuple(normalized)


def _usage_metrics(responses: Sequence[Any]) -> UsageMetrics:
    if not responses or any(response.raw_usage is None for response in responses):
        return UsageMetrics()

    input_total = 0
    output_total = 0
    total_total = 0
    cost_total = 0.0
    cost_known = True
    for response in responses:
        raw = response.raw_usage
        if not isinstance(raw, Mapping):
            return UsageMetrics()
        input_value = raw.get("input_tokens", raw.get("prompt_tokens"))
        output_value = raw.get("output_tokens", raw.get("completion_tokens"))
        total_value = raw.get("total_tokens")
        if not all(
            isinstance(value, int) and value >= 0
            for value in (input_value, output_value)
        ):
            return UsageMetrics()
        if total_value is None:
            total_value = input_value + output_value
        if not isinstance(total_value, int) or total_value < 0:
            return UsageMetrics()
        input_total += input_value
        output_total += output_value
        total_total += total_value
        cost_value = raw.get("cost_usd")
        if (
            isinstance(cost_value, int | float)
            and not isinstance(cost_value, bool)
            and cost_value >= 0
        ):
            cost_total += float(cost_value)
        else:
            cost_known = False
    return UsageMetrics(
        input_tokens=input_total,
        output_tokens=output_total,
        total_tokens=total_total,
        cost_usd=cost_total if cost_known else None,
    )


def _runner_input(
    input_value: str | Sequence[ConversationTurn],
) -> tuple[
    Any,
    tuple[tuple[ConversationRole, str, str | None, dict[str, Any]], ...],
]:
    """Preserve fixture dialogue roles instead of embedding them in one user turn."""

    if isinstance(input_value, str):
        return input_value, ((ConversationRole.USER, input_value, None, {}),)
    turns = tuple(input_value)
    if not turns:
        raise ValueError("an agent run requires at least one conversation turn")
    items: list[dict[str, str]] = []
    normalized: list[
        tuple[ConversationRole, str, str | None, dict[str, Any]]
    ] = []
    for turn in turns:
        if not isinstance(turn, ConversationTurn):
            raise TypeError("conversation input must contain ConversationTurn values")
        if turn.role not in {ConversationRole.USER, ConversationRole.ASSISTANT}:
            raise ValueError(
                f"Phase 1 cannot seed {turn.role.value!r} conversation turns"
            )
        items.append({"role": turn.role.value, "content": turn.content})
        normalized.append((turn.role, turn.content, turn.turn_id, dict(turn.metadata)))
    return items, tuple(normalized)


def _scripted_followups(
    turns: Sequence[ConversationTurn],
) -> tuple[ConversationTurn, ...]:
    """Validate the replies the runtime will inject between execution stages."""

    scripted = tuple(turns)
    for turn in scripted:
        if not isinstance(turn, ConversationTurn):
            raise TypeError("follow-up input must contain ConversationTurn values")
        if turn.role != ConversationRole.USER:
            # Injecting an assistant turn mid-run would put words the target
            # never produced into the trajectory the oracle then scores.
            raise ValueError(
                f"a scripted follow-up must be a user turn, not {turn.role.value!r}"
            )
    return scripted


def _model_turns_used(capture: _Capture) -> int:
    """Model turns already spent by this scenario, across every stage."""

    return sum(
        1
        for event in capture.events
        if event.event_type == CanonicalEventType.MODEL_REQUEST
    )


async def _record_unknown_tool_calls(
    capture: _Capture,
    prepared: PreparedTarget,
    gateway: ToolGatewayProtocol,
    *,
    start_index: int = 0,
) -> int:
    """Preserve model-owned unknown calls as deterministic agent evidence.

    Allowed names are the active agent's own FunctionTools plus that agent's
    declared handoff tool names.  A tool that exists only on a different
    reachable agent is recorded as blocked, not treated as infrastructure.
    """

    tools_by_agent = prepared.metadata.get("tools_by_agent", {})
    handoff_tools_by_agent = prepared.metadata.get("handoff_tools_by_agent", {})
    fallback = {
        *prepared.tool_names,
        *prepared.metadata.get("handoff_tool_names", ()),
    }
    recorded = 0
    for index, response in enumerate(capture.responses):
        if index < start_index:
            continue
        agent_name = (
            capture.response_agents[index]
            if index < len(capture.response_agents)
            else None
        )
        if isinstance(agent_name, str) and (
            agent_name in tools_by_agent or agent_name in handoff_tools_by_agent
        ):
            allowed = {
                *tools_by_agent.get(agent_name, ()),
                *handoff_tools_by_agent.get(agent_name, ()),
            }
        else:
            allowed = set(fallback)
        for output in getattr(response, "output", ()):
            if not isinstance(output, ResponseFunctionToolCall):
                continue
            tool_name = output.name
            if tool_name in allowed:
                continue
            raw_arguments = output.arguments
            try:
                decoded = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError, ValueError):
                decoded = {}
            arguments = decoded if isinstance(decoded, dict) else {}
            attempt = await capture.tool_attempt(
                tool_name=tool_name,
                arguments=arguments,
                raw_arguments=raw_arguments,
                call_id=output.call_id,
                validation_errors=(
                    "tool is not declared by the active agent"
                    if isinstance(agent_name, str)
                    else "tool is not declared by the inspected agent"
                ),
                state_changing=False,
                destructive=False,
                agent_name=agent_name if isinstance(agent_name, str) else None,
            )
            budget_error = _consume_adapter_tool_budget(gateway)
            error = budget_error or ToolError(
                code="unknown_tool",
                message=(
                    f"Tool {tool_name!r} is not declared by agent {agent_name!r} "
                    "and was blocked."
                    if isinstance(agent_name, str)
                    else f"Tool {tool_name!r} is not declared and was blocked."
                ),
                retryable=False,
                details={
                    "supported_tools": sorted(allowed),
                    **({"agent_name": agent_name} if isinstance(agent_name, str) else {}),
                },
            )
            await capture.tool_result(
                attempt=attempt,
                status=ToolOutcomeStatus.BLOCKED,
                result=None,
                error=error,
                started_at=utc_now(),
            )
            recorded += 1
    return recorded


def _agent_preflight_issues(
    agent: Any,
    location: str,
    *,
    handoff_tool_names: tuple[str, ...] = (),
) -> list[SupportIssue]:
    """Apply the full single-agent support checks to one reachable agent."""

    issues: list[SupportIssue] = []
    if not isinstance(agent.instructions, str) and agent.instructions is not None:
        issues.append(
            SupportIssue(
                code="dynamic_instructions",
                message="Dynamic instruction callbacks are unsupported in Phase 1.",
                location=f"{location}.instructions",
            )
        )
    if agent.prompt is not None:
        issues.append(
            SupportIssue(
                code="stored_prompt",
                message="Stored prompts can add opaque runtime behavior and are unsupported.",
                location=f"{location}.prompt",
            )
        )
    if agent.output_type is not None:
        _, output_schema_error = _output_schema(agent)
        if output_schema_error is not None:
            issues.append(
                SupportIssue(
                    code="structured_output",
                    message=(
                        "Structured output is supported only when its JSON schema "
                        "can be statically derived without executing target code: "
                        f"{output_schema_error}"
                    ),
                    location=f"{location}.output_type",
                )
            )
    if agent.mcp_servers:
        issues.append(
            SupportIssue(
                code="mcp_servers",
                message="MCP tools cannot be intercepted by the Phase 1 gateway.",
                location=f"{location}.mcp_servers",
            )
        )
    if agent.hooks is not None:
        issues.append(
            SupportIssue(
                code="agent_hooks",
                message="Agent-owned lifecycle hooks are unsupported in isolated evaluation runs.",
                location=f"{location}.hooks",
            )
        )
    if agent.input_guardrails or agent.output_guardrails:
        issues.append(
            SupportIssue(
                code="agent_guardrails",
                message="Executable agent guardrails are inspected but not run in Phase 1.",
                location=f"{location}.guardrails",
            )
        )
    if agent.tool_use_behavior != "run_llm_again":
        issues.append(
            SupportIssue(
                code="tool_use_behavior",
                message="Only the default run_llm_again tool behavior is supported.",
                location=f"{location}.tool_use_behavior",
            )
        )

    extras = {
        "extra_query": agent.model_settings.extra_query,
        "extra_body": agent.model_settings.extra_body,
        "extra_headers": agent.model_settings.extra_headers,
        "extra_args": agent.model_settings.extra_args,
    }
    for name, value in extras.items():
        if value is not None:
            issues.append(
                SupportIssue(
                    code="opaque_model_settings",
                    message=f"Model setting {name} is opaque and unsupported in Phase 1.",
                    location=f"{location}.model_settings.{name}",
                )
            )

    seen_names: set[str] = set()
    for index, tool in enumerate(agent.tools):
        tool_location = f"{location}.tools[{index}]"
        if type(tool) is not FunctionTool:
            issues.append(
                SupportIssue(
                    code="unsupported_tool_type",
                    message=(
                        "Only exact FunctionTool instances can be replaced safely; "
                        f"found {type(tool).__name__}."
                    ),
                    location=tool_location,
                )
            )
            continue
        if tool.name in seen_names:
            issues.append(
                SupportIssue(
                    code="duplicate_tool_name",
                    message=f"Duplicate tool name {tool.name!r} is ambiguous.",
                    location=tool_location,
                )
            )
        seen_names.add(tool.name)
        if not isinstance(tool.is_enabled, bool):
            issues.append(
                SupportIssue(
                    code="dynamic_tool_enablement",
                    message=f"Tool {tool.name!r} has an executable is_enabled callback.",
                    location=f"{tool_location}.is_enabled",
                )
            )
        if tool.tool_input_guardrails or tool.tool_output_guardrails:
            issues.append(
                SupportIssue(
                    code="tool_guardrails",
                    message=f"Tool {tool.name!r} has executable tool guardrails.",
                    location=f"{tool_location}.guardrails",
                )
            )
        if tool.needs_approval is not False:
            issues.append(
                SupportIssue(
                    code="tool_approval_callback",
                    message=f"Tool {tool.name!r} has approval behavior unsupported by Phase 1.",
                    location=f"{tool_location}.needs_approval",
                )
            )
        if (
            tool.timeout_seconds is not None
            or tool.timeout_error_function is not None
        ):
            issues.append(
                SupportIssue(
                    code="tool_timeout_callback",
                    message=f"Tool {tool.name!r} has SDK-owned timeout behavior.",
                    location=f"{tool_location}.timeout",
                )
            )
        if tool.defer_loading or tool.allowed_callers is not None:
            issues.append(
                SupportIssue(
                    code="advanced_tool_dispatch",
                    message=f"Tool {tool.name!r} uses deferred or programmatic dispatch.",
                    location=tool_location,
                )
            )
        if tool.custom_data_extractor is not None:
            issues.append(
                SupportIssue(
                    code="tool_custom_data_callback",
                    message=f"Tool {tool.name!r} has a custom data callback.",
                    location=f"{tool_location}.custom_data_extractor",
                )
            )
        if (
            tool._is_agent_tool
            or tool._agent_instance is not None
            or tool._is_codex_tool
            or tool._tool_namespace is not None
            or tool._tool_namespace_description is not None
            or tool._tool_origin is not None
        ):
            issues.append(
                SupportIssue(
                    code="advanced_function_tool",
                    message=f"Tool {tool.name!r} is not a plain local function tool.",
                    location=tool_location,
                )
            )
        try:
            offline_validator(tool.params_json_schema)
        except (SchemaError, UnsafeSchemaReference) as exc:
            issues.append(
                SupportIssue(
                    code="invalid_tool_schema",
                    message=f"Tool {tool.name!r} has invalid or unsafe JSON Schema: {exc}",
                    location=f"{tool_location}.params_json_schema",
                )
            )

    tool_choice = agent.model_settings.tool_choice
    allowed_choices = {None, "auto", "required", "none", *seen_names, *handoff_tool_names}
    if (
        not isinstance(tool_choice, str | type(None))
        or tool_choice not in allowed_choices
    ):
        issues.append(
            SupportIssue(
                code="unsupported_tool_choice",
                message="Model tool_choice targets an unsupported tool surface.",
                location=f"{location}.model_settings.tool_choice",
            )
        )
    return issues


async def _record_ignored_handoffs(capture: _Capture, prepared: PreparedTarget) -> None:
    """Record redundant same-turn handoff calls the SDK discarded.

    The SDK executes only the first handoff tool call in one model turn and
    answers every additional one with a fixed "Multiple handoffs detected"
    message.  Those discarded calls are deterministic model behavior worth
    asserting on, so they become ``ignored`` HANDOFF events.
    """

    handoff_names = set(prepared.metadata.get("handoff_tool_names", ()))
    if not handoff_names:
        return
    handoff_targets = cast(
        dict[tuple[str, str], str], prepared.metadata.get("handoff_targets", {})
    )
    for index, response in enumerate(capture.responses):
        calls = [
            output
            for output in getattr(response, "output", ())
            if isinstance(output, ResponseFunctionToolCall)
            and output.name in handoff_names
        ]
        for extra in calls[1:]:
            try:
                decoded = json.loads(extra.arguments)
            except (json.JSONDecodeError, TypeError, ValueError):
                decoded = None
            from_agent = (
                capture.response_agents[index]
                if index < len(capture.response_agents)
                else None
            )
            await capture.handoff_event(
                tool_name=extra.name,
                from_agent=from_agent,
                to_agent=(
                    handoff_targets.get((from_agent, extra.name))
                    if from_agent is not None
                    else None
                ),
                arguments=decoded if isinstance(decoded, dict) else {},
                ignored=True,
                call_id=extra.call_id,
                source_event_id=(
                    capture.response_event_ids[index]
                    if index < len(capture.response_event_ids)
                    else None
                ),
            )


class OpenAIAgentsAdapter(FrameworkAdapter):
    """Adapter for one trusted local OpenAI Agents SDK ``Agent``."""

    framework = FRAMEWORK_NAME

    def inspect(
        self,
        target: Any,
        *,
        source: str | None = None,
        identity_locator: str | None = None,
        declared_tool_risk: "Mapping[str, ToolRiskDeclaration] | None" = None,
    ) -> AgentSpec:
        _require_sdk()
        source = source or "runtime:agent"
        if type(target) is not Agent:
            raise TypeError("OpenAIAgentsAdapter.inspect expects an exact agents.Agent")

        framework_version = _sdk_version()
        provider, model_name, provider_inferred, model_inferred = _model_identity(target)
        output_schema, output_schema_error = _output_schema(target)
        unknowns: list[UnknownProperty] = []

        if isinstance(target.instructions, str) or target.instructions is None:
            system_instructions = _property(
                target.instructions,
                locator=f"{source}.agent.instructions",
                summary="Static system instructions declared on the runtime Agent.",
                kind=SourceKind.SYSTEM_INSTRUCTION,
            )
        else:
            system_instructions = _unknown_property(
                None,
                locator=f"{source}.agent.instructions",
                summary="Dynamic instruction callbacks are not executed during inspection.",
            )
            unknowns.append(
                UnknownProperty(
                    path="instructions.system",
                    reason="The agent uses a dynamic instruction callback.",
                    source=system_instructions.source,
                    confidence=0.0,
                    evidence=system_instructions.evidence,
                )
            )
        if output_schema_error is not None:
            locator = f"{source}.agent.output_type"
            unknowns.append(
                UnknownProperty(
                    path="interface.output_schema",
                    reason=output_schema_error,
                    source=SourceReference(
                        kind=SourceKind.RUNTIME_INTROSPECTION, locator=locator
                    ),
                    confidence=0.0,
                    evidence=(
                        SpecEvidence(
                            evidence_id=_evidence_id(locator),
                            summary="The configured output type could not be converted to JSON Schema.",
                            locator=locator,
                        ),
                    ),
                )
            )

        graph = _reachable_graph(target)
        tool_properties: list[AgentProperty[Any]] = []
        definitions: list[ToolDefinition] = []
        tool_risk_assertions: dict[str, ToolRiskAssertion] = {}
        fingerprint_tools: list[Any] = []
        for index, tool in enumerate(target.tools):
            locator = f"{source}.agent.tools[{index}]"
            if isinstance(tool, FunctionTool):
                definition, risk_assertion = _tool_definition(
                    tool, declared_tool_risk=declared_tool_risk
                )
                definitions.append(definition)
                tool_risk_assertions.setdefault(tool.name, risk_assertion)
                fingerprint_tools.append(definition.model_dump(mode="json"))
                tool_properties.append(
                    _property(
                        definition,
                        locator=locator,
                        summary="Function tool name, description, and schemas read from the SDK object.",
                        kind=SourceKind.TOOL_SCHEMA,
                        confidence=0.9,
                        inferred=True,
                        authoritative=False,
                    )
                )
            else:
                tool_type = type(tool).__name__
                fingerprint_tools.append({"unsupported_type": tool_type})
                locator = f"{locator}.type"
                unknowns.append(
                    UnknownProperty(
                        path=f"tools.items[{index}]",
                        reason=f"Unsupported SDK tool type: {tool_type}",
                        source=SourceReference(
                            kind=SourceKind.RUNTIME_INTROSPECTION,
                            locator=locator,
                        ),
                        confidence=1.0,
                        evidence=(
                            SpecEvidence(
                                evidence_id=_evidence_id(locator),
                                summary=f"Runtime object is {tool_type}, not FunctionTool.",
                                locator=locator,
                            ),
                        ),
                    )
                )

        # Tools declared by reachable handoff destinations join the merged tool
        # surface: the scenario gateway must be able to allowlist every tool a
        # downstream agent can attempt, and preflight separately rejects
        # cross-agent name collisions before any run.
        subagent_fingerprints: list[dict[str, Any]] = []
        for agent, agent_location in zip(graph.agents[1:], graph.locations[1:]):
            subagent_tools: list[Any] = []
            for index, tool in enumerate(agent.tools):
                locator = f"{source}.{agent_location}.tools[{index}]"
                if isinstance(tool, FunctionTool):
                    definition, risk_assertion = _tool_definition(
                        tool, declared_tool_risk=declared_tool_risk
                    )
                    definitions.append(definition)
                    tool_risk_assertions.setdefault(tool.name, risk_assertion)
                    subagent_tools.append(definition.model_dump(mode="json"))
                    tool_properties.append(
                        _property(
                            definition,
                            locator=locator,
                            summary=(
                                f"Function tool declared by reachable handoff agent {agent.name!r}."
                            ),
                            kind=SourceKind.TOOL_SCHEMA,
                            confidence=0.9,
                            inferred=True,
                            authoritative=False,
                        )
                    )
                else:
                    tool_type = type(tool).__name__
                    subagent_tools.append({"unsupported_type": tool_type})
                    unknowns.append(
                        UnknownProperty(
                            path=f"{agent_location}.tools[{index}]",
                            reason=f"Unsupported SDK tool type: {tool_type}",
                            source=SourceReference(
                                kind=SourceKind.RUNTIME_INTROSPECTION,
                                locator=f"{locator}.type",
                            ),
                            confidence=1.0,
                            evidence=(
                                SpecEvidence(
                                    evidence_id=_evidence_id(f"{locator}.type"),
                                    summary=f"Runtime object is {tool_type}, not FunctionTool.",
                                    locator=f"{locator}.type",
                                ),
                            ),
                        )
                    )
            if not isinstance(agent.instructions, str) and agent.instructions is not None:
                locator = f"{source}.{agent_location}.instructions"
                unknowns.append(
                    UnknownProperty(
                        path=f"{agent_location}.instructions",
                        reason="The reachable agent uses a dynamic instruction callback.",
                        source=SourceReference(
                            kind=SourceKind.RUNTIME_INTROSPECTION, locator=locator
                        ),
                        confidence=0.0,
                        evidence=(
                            SpecEvidence(
                                evidence_id=_evidence_id(locator),
                                summary="Dynamic instruction callbacks are not executed during inspection.",
                                locator=locator,
                            ),
                        ),
                    )
                )
            subagent_fingerprints.append(
                {
                    "location": agent_location,
                    "name": agent.name,
                    "model": _model_identity(agent)[1],
                    "instructions_hash": (
                        canonical_hash(agent.instructions)
                        if isinstance(agent.instructions, str)
                        else None
                    ),
                    "tools": subagent_tools,
                }
            )
        for edge in graph.edges:
            if edge.destination is not None:
                continue
            locator = f"{source}.{edge.location}"
            unknowns.append(
                UnknownProperty(
                    path=edge.location,
                    reason="The handoff entry could not be resolved to a static destination agent.",
                    source=SourceReference(
                        kind=SourceKind.RUNTIME_INTROSPECTION, locator=locator
                    ),
                    confidence=0.0,
                    evidence=(
                        SpecEvidence(
                            evidence_id=_evidence_id(locator),
                            summary="Handoff destinations are resolved statically and never executed.",
                            locator=locator,
                        ),
                    ),
                )
            )

        capability_properties: list[AgentProperty[Any]] = []
        for extracted in extract_capabilities(definitions):
            capability_properties.append(
                AgentProperty(
                    value=extracted.capability,
                    source=SourceReference(
                        kind=SourceKind.TOOL_SCHEMA,
                        locator=f"tool:{extracted.tool_name}",
                        description=(
                            "Argument surface read from the declared schema; action "
                            "kind and side-effect risk are inferred and not authoritative."
                        ),
                    ),
                    confidence=extracted.confidence,
                    evidence=extracted.evidence,
                    inferred=True,
                    authoritative=False,
                )
            )
            unknowns.extend(extracted.unknowns)

        fingerprint: dict[str, Any] = {
            "source": source,
            "name": target.name,
            "framework_version": framework_version,
            "model": model_name,
            "instructions_hash": (
                canonical_hash(target.instructions)
                if isinstance(target.instructions, str)
                else None
            ),
            "tools": fingerprint_tools,
        }
        if graph.has_handoffs:
            # Included only for handoff targets so every existing single-agent
            # spec_id (and the replay/baseline bindings derived from it) stays
            # byte-stable.
            fingerprint["handoff_graph"] = {
                "agents": subagent_fingerprints,
                "edges": [
                    {
                        "location": edge.location,
                        "from": edge.from_agent.name,
                        "tool_name": edge.tool_name,
                        "to": (
                            edge.destination.name
                            if edge.destination is not None
                            else None
                        ),
                    }
                    for edge in graph.edges
                ],
            }
        spec_id, legacy_spec_id = portable_identity(
            fingerprint,
            location_locator=source,
            identity_locator=identity_locator,
        )
        runtime_locator = f"{source}.agent.runtime"

        return AgentSpec(
            spec_id=spec_id,
            legacy_spec_id=legacy_spec_id,
            identity=IdentitySpec(
                name=_property(
                    target.name,
                    locator=f"{source}.agent.name",
                    summary="Agent name read from the runtime object.",
                ),
                framework=_property(
                    FRAMEWORK_NAME,
                    locator="python-package:openai-agents",
                    summary="Runtime object is an OpenAI Agents SDK Agent.",
                    kind=SourceKind.FRAMEWORK_METADATA,
                ),
                framework_version=_property(
                    framework_version,
                    locator="python-package:openai-agents.version",
                    summary="Installed distribution version.",
                    kind=SourceKind.FRAMEWORK_METADATA,
                ),
                provider=_property(
                    provider,
                    locator=f"{source}.agent.model",
                    summary="Provider derived from the explicit model configuration.",
                    confidence=0.5 if provider_inferred else 1.0,
                    inferred=provider_inferred,
                    authoritative=not provider_inferred,
                ),
                model=_property(
                    model_name,
                    locator=f"{source}.agent.model",
                    summary="Model configuration read without invoking the provider.",
                    confidence=0.5 if model_inferred else 1.0,
                    inferred=model_inferred,
                    authoritative=not model_inferred,
                ),
            ),
            interface=InterfaceSpec(
                entrypoint=_property(
                    source,
                    locator=source,
                    summary="Explicit configured Python entrypoint.",
                    kind=SourceKind.DEVELOPER_CONFIG,
                ),
                input_modalities=_property(
                    ("text",),
                    locator=f"{source}.interface.input",
                    summary="Phase 1 adapter accepts text input.",
                    kind=SourceKind.FRAMEWORK_METADATA,
                ),
                output_modalities=_property(
                    ("text",),
                    locator=f"{source}.interface.output",
                    summary="Phase 1 adapter captures text output.",
                    kind=SourceKind.FRAMEWORK_METADATA,
                ),
                input_schema=_property(
                    {"type": "string"},
                    locator=f"{source}.interface.input_schema",
                    summary="Runner input is a text string.",
                    kind=SourceKind.FRAMEWORK_METADATA,
                ),
                output_schema=_property(
                    output_schema,
                    locator=f"{source}.agent.output_type",
                    summary="Output schema extracted from the configured output type.",
                    confidence=1.0 if output_schema_error is None else 0.0,
                    authoritative=output_schema_error is None,
                ),
                interactive=_property(
                    True,
                    locator=f"{source}.interface.interactive",
                    summary="Agents SDK runner supports iterative model/tool turns.",
                    kind=SourceKind.FRAMEWORK_METADATA,
                ),
            ),
            instructions=InstructionsSpec(
                system=system_instructions,
                developer=_unknown_property(
                    None,
                    locator=f"{source}.instructions.developer",
                    summary="The SDK Agent has no separate developer-instruction field.",
                ),
            ),
            capabilities=CapabilitiesSpec(items=tuple(capability_properties)),
            tools=ToolsSpec(items=tuple(tool_properties)),
            tool_risk=ToolRiskSpec(
                items=tuple(
                    tool_risk_assertions[name] for name in sorted(tool_risk_assertions)
                )
            ),
            tool_policies=ToolPoliciesSpec(),
            guardrails=GuardrailsSpec(items=_guardrail_properties(target, source)),
            workflows=WorkflowsSpec(),
            policies=PoliciesSpec(),
            runtime=RuntimeSpec(
                max_model_turns=_unknown_property(
                    None,
                    locator=f"{runtime_locator}.max_model_turns",
                    summary="The maximum is supplied by each AgentCheck scenario run.",
                ),
                max_tool_calls=_unknown_property(
                    None,
                    locator=f"{runtime_locator}.max_tool_calls",
                    summary="No tool-call limit is declared on the Agent object.",
                ),
                timeout_seconds=_unknown_property(
                    None,
                    locator=f"{runtime_locator}.timeout_seconds",
                    summary="Wall-clock timeout is enforced by the AgentCheck worker.",
                ),
                token_budget=_unknown_property(
                    None,
                    locator=f"{runtime_locator}.token_budget",
                    summary="No token budget is declared on the Agent object.",
                ),
                cost_budget_usd=_unknown_property(
                    None,
                    locator=f"{runtime_locator}.cost_budget_usd",
                    summary="No cost budget is declared on the Agent object.",
                ),
            ),
            observability=ObservabilitySpec(
                supported_event_types=_property(
                    (
                        CanonicalEventType.USER_TURN.value,
                        CanonicalEventType.ASSISTANT_OUTPUT.value,
                        CanonicalEventType.MODEL_REQUEST.value,
                        CanonicalEventType.MODEL_RESPONSE.value,
                        CanonicalEventType.TOOL_ATTEMPT.value,
                        CanonicalEventType.TOOL_RESULT.value,
                        CanonicalEventType.HANDOFF.value,
                        CanonicalEventType.ERROR.value,
                        CanonicalEventType.FINAL_OUTPUT.value,
                    ),
                    locator="agentcheck.adapters.openai_agents.events",
                    summary="Observable SDK callbacks and run items normalized by the adapter.",
                    kind=SourceKind.FRAMEWORK_METADATA,
                ),
                usage_metrics=_property(
                    ("input_tokens", "output_tokens", "total_tokens"),
                    locator="agents.items.ModelResponse.raw_usage",
                    summary="Usage is known only when a raw provider usage payload is present.",
                    kind=SourceKind.FRAMEWORK_METADATA,
                ),
                provider_request_ids=_property(
                    True,
                    locator="agents.items.ModelResponse.request_id",
                    summary="The SDK exposes provider request IDs when available.",
                    kind=SourceKind.FRAMEWORK_METADATA,
                ),
                source_event_links=_property(
                    True,
                    locator="agents.tool_context.ToolContext.tool_call_id",
                    summary="Tool call IDs allow source event correlation.",
                    kind=SourceKind.FRAMEWORK_METADATA,
                ),
            ),
            unknowns=tuple(unknowns),
            provenance=InspectionProvenance(
                inspector="agentcheck.adapters.openai_agents",
                inspector_version=INSPECTOR_VERSION,
                inspected_at=utc_now(),
                target=source,
                git_revision=None,
                sources=(
                    SourceReference(
                        kind=SourceKind.RUNTIME_INTROSPECTION, locator=source
                    ),
                    SourceReference(
                        kind=SourceKind.FRAMEWORK_METADATA,
                        locator="python-package:openai-agents",
                    ),
                ),
            ),
        )

    def preflight(self, target: Any) -> PreflightReport:
        _require_sdk()
        issues: list[SupportIssue] = []
        version = _sdk_version()
        if not _supported_sdk_version(version):
            issues.append(
                SupportIssue(
                    code="unsupported_sdk_version",
                    message=f"Expected openai-agents 0.20.x, found {version or 'not installed'}.",
                    location="python-package:openai-agents",
                )
            )
        if type(target) is not Agent:
            issues.append(
                SupportIssue(
                    code="unsupported_agent_type",
                    message="Phase 1 requires an exact agents.Agent instance.",
                    location=type(target).__name__,
                )
            )
            return PreflightReport(framework=FRAMEWORK_NAME, issues=tuple(issues))

        graph = _reachable_graph(target)
        issues.extend(graph.issues)
        for edge in graph.edges:
            issues.extend(edge.issues)

        edges_by_agent = _edges_by_agent(graph)
        seen_agent_names: dict[str, str] = {}
        tool_owner_location: dict[str, str] = {}
        tool_contract_identity: dict[str, str] = {}
        all_function_tool_names: set[str] = set()
        for agent, location in zip(graph.agents, graph.locations):
            agent_edges = edges_by_agent.get(id(agent), [])
            handoff_names = [
                edge.tool_name for edge in agent_edges if edge.tool_name is not None
            ]
            issues.extend(
                _agent_preflight_issues(
                    agent, location, handoff_tool_names=tuple(handoff_names)
                )
            )
            if agent.name in seen_agent_names:
                issues.append(
                    SupportIssue(
                        code="duplicate_agent_name",
                        message=(
                            f"Two reachable agents share the name {agent.name!r}; "
                            "handoff and event attribution would be ambiguous."
                        ),
                        location=location,
                    )
                )
            else:
                seen_agent_names[agent.name] = location
            local_names: set[str] = set()
            for index, tool in enumerate(agent.tools):
                if type(tool) is not FunctionTool or tool.name in local_names:
                    continue
                local_names.add(tool.name)
                all_function_tool_names.add(tool.name)
                owner = tool_owner_location.get(tool.name)
                contract = _tool_contract_identity(tool)
                if owner is None:
                    tool_owner_location[tool.name] = location
                    tool_contract_identity[tool.name] = contract
                elif owner != location and tool_contract_identity[tool.name] != contract:
                    # Sharing one imported tool across several agents is ordinary
                    # SDK usage and is not ambiguous: the gateway intercepts by
                    # name and validates against the schema, so an identical
                    # contract means the same call means the same thing wherever
                    # it is reached. Only a genuine contract conflict -- same name,
                    # different schema or description -- leaves scenario tool
                    # identity undecidable, so only that still fails closed.
                    issues.append(
                        SupportIssue(
                            code="duplicate_tool_name",
                            message=(
                                f"Tool name {tool.name!r} is declared by multiple "
                                "reachable agents with different schemas or "
                                "descriptions; scenario tool identity would be "
                                "ambiguous."
                            ),
                            location=f"{location}.tools[{index}]",
                        )
                    )
            seen_handoff_names: set[str] = set()
            for edge in agent_edges:
                if edge.tool_name is None:
                    continue
                if edge.tool_name in seen_handoff_names:
                    issues.append(
                        SupportIssue(
                            code="handoff_tool_name_collision",
                            message=(
                                f"Handoff tool name {edge.tool_name!r} is declared "
                                "twice on one agent."
                            ),
                            location=edge.location,
                        )
                    )
                seen_handoff_names.add(edge.tool_name)
        for edge in graph.edges:
            if edge.tool_name is not None and edge.tool_name in all_function_tool_names:
                issues.append(
                    SupportIssue(
                        code="handoff_tool_name_collision",
                        message=(
                            f"Handoff tool name {edge.tool_name!r} collides with a "
                            "declared function tool name."
                        ),
                        location=edge.location,
                    )
                )
        return PreflightReport(framework=FRAMEWORK_NAME, issues=tuple(issues))

    def describe_topology(
        self, target: Any, *, source: str | None = None
    ) -> dict[str, Any] | None:
        """Additive inspect diagnostic describing the reachable handoff graph.

        Returns ``None`` for single-agent targets so existing inspect behavior
        and payloads stay byte-identical.  This is diagnostic data, not part of
        ``agentcheck.agent_spec.v1``.
        """

        _require_sdk()
        del source
        if type(target) is not Agent:
            return None
        graph = _reachable_graph(target)
        if not graph.has_handoffs:
            return None
        edges_by_agent = _edges_by_agent(graph)
        agents_payload: list[dict[str, Any]] = []
        for agent, location in zip(graph.agents, graph.locations):
            handoffs_payload: list[dict[str, Any]] = []
            for edge in edges_by_agent.get(id(agent), []):
                edge_payload: dict[str, Any] = {
                    "tool_name": edge.tool_name,
                    "target_agent": (
                        edge.destination.name
                        if edge.destination is not None
                        else None
                    ),
                    "location": edge.location,
                    "issue_codes": sorted({issue.code for issue in edge.issues}),
                }
                if edge.context_assignments:
                    edge_payload["context_assignments"] = encode_context_assignments(
                        edge.context_assignments
                    )
                handoffs_payload.append(edge_payload)
            agents_payload.append(
                {
                    "name": agent.name,
                    "location": location,
                    "model": _model_identity(agent)[1],
                    "instructions_static": (
                        isinstance(agent.instructions, str) or agent.instructions is None
                    ),
                    "tool_names": [
                        tool.name for tool in agent.tools if isinstance(tool, FunctionTool)
                    ],
                    "handoffs": handoffs_payload,
                }
            )
        return _json_object({"framework": FRAMEWORK_NAME, "agents": agents_payload})

    def _safe_tools_for(
        self,
        agent: Any,
        *,
        gateway: ToolGatewayProtocol,
        world_state: Any,
        capture_holder: dict[str, _Capture | None],
        tool_risks: dict[str, tuple[bool, bool]],
        declared_tool_risk: "Mapping[str, ToolRiskDeclaration] | None" = None,
    ) -> list[Any]:
        safe_tools: list[Any] = []
        for original in agent.tools:
            definition, _ = _tool_definition(
                original, declared_tool_risk=declared_tool_risk
            )
            schema = copy.deepcopy(original.params_json_schema)
            invoker = _make_safe_invoker(
                tool_name=original.name,
                schema=schema,
                gateway=gateway,
                world_state=world_state,
                capture_holder=capture_holder,
                state_changing=definition.state_changing,
                destructive=definition.destructive,
            )
            safe = FunctionTool(
                name=original.name,
                description=original.description,
                params_json_schema=schema,
                on_invoke_tool=invoker,
                strict_json_schema=original.strict_json_schema,
                is_enabled=original.is_enabled,
                output_json_schema=(
                    copy.deepcopy(original.output_json_schema)
                    if original.output_json_schema is not None
                    else None
                ),
            )
            # Unexpected gateway exceptions are platform failures.  The SDK's
            # default formatter would otherwise turn them into model-visible text.
            set_function_tool_failure_error_function(safe, None)
            safe_tools.append(safe)
            tool_risks[original.name] = (
                definition.state_changing,
                definition.destructive,
            )
        return safe_tools

    def prepare(
        self,
        target: Any,
        gateway: ToolGatewayProtocol,
        *,
        world_state: Any = None,
        event_sink: EventSinkProtocol | None = None,
        source: str | None = None,
        identity_locator: str | None = None,
        controlled_model: bool = False,
        declared_tool_risk: "Mapping[str, ToolRiskDeclaration] | None" = None,
    ) -> PreparedTarget:
        report = self.preflight(target)
        report.require_supported()
        spec = self.inspect(
            target,
            source=source,
            identity_locator=identity_locator,
            declared_tool_risk=declared_tool_risk,
        )
        graph = _reachable_graph(target)
        capture_holder: dict[str, _Capture | None] = {"capture": None}
        tool_risks: dict[str, tuple[bool, bool]] = {}
        tool_names: list[str] = []
        tools_by_agent: dict[str, tuple[str, ...]] = {}
        clones: dict[int, Any] = {}

        # Pass one: clone every reachable agent with reconstructed tools and no
        # handoffs. Each clone keeps its own model object so a target that
        # already supplies a local scripted model keeps driving offline runs;
        # controlled_model instead substitutes a deterministic offline model on
        # every reachable agent, so a target whose provider is unreachable can
        # still be evaluated.
        # Imported here, not at module scope: controlled_model depends on the
        # framework SDK, and agentcheck.cli must stay importable when the
        # optional [agentcheck] extra is absent. _require_sdk() in preflight
        # has already run by this point.
        controlled = None
        if controlled_model:
            from .controlled_model import ControlledModel

            controlled = ControlledModel(spec.interface.output_schema.value)
        for agent in graph.agents:
            safe_tools = self._safe_tools_for(
                agent,
                gateway=gateway,
                world_state=world_state,
                capture_holder=capture_holder,
                tool_risks=tool_risks,
                declared_tool_risk=declared_tool_risk,
            )
            agent_tool_names = tuple(tool.name for tool in safe_tools)
            tools_by_agent[agent.name] = agent_tool_names
            tool_names.extend(agent_tool_names)
            clone_overrides: dict[str, Any] = {
                "tools": safe_tools,
                "mcp_servers": [],
                "handoffs": [],
                "prompt": None,
                "hooks": None,
                "input_guardrails": [],
                "output_guardrails": [],
                "model_settings": _sanitized_model_settings(agent.model_settings),
            }
            if controlled is not None:
                clone_overrides["model"] = controlled
            clones[id(agent)] = agent.clone(**clone_overrides)

        # Pass two: wire AgentCheck-owned handoffs between clones.  Original
        # Handoff objects, their routing closures, and any user callbacks are
        # left behind on the source graph and are never invoked.  Proven
        # on_handoff effects become declarative assignments on an
        # AgentCheck-owned context bag.
        handoff_tool_names: list[str] = []
        handoff_tools_by_agent: dict[str, list[str]] = {}
        handoff_targets: dict[tuple[str, str], str] = {}
        run_context: AgentCheckRunContext | None = None
        if any(edge.context_assignments for edge in graph.edges):
            run_context = AgentCheckRunContext()
        for edge in graph.edges:
            if edge.destination is None or edge.tool_name is None:
                raise AdapterRuntimeError(
                    "an unproven handoff edge survived preflight"
                )  # pragma: no cover - preflight fails closed first
            destination_clone = clones[id(edge.destination)]
            safe_handoff = Handoff(
                tool_name=edge.tool_name,
                tool_description=edge.tool_description or "",
                input_json_schema=(
                    copy.deepcopy(edge.input_json_schema)
                    if edge.input_json_schema is not None
                    else _empty_handoff_payload_schema()
                ),
                on_invoke_handoff=_make_safe_handoff_invoker(
                    tool_name=edge.tool_name,
                    from_agent_name=edge.from_agent.name,
                    to_agent_name=edge.destination.name,
                    destination=destination_clone,
                    capture_holder=capture_holder,
                    context_assignments=edge.context_assignments,
                ),
                agent_name=edge.destination.name,
                input_filter=None,
                nest_handoff_history=None,
                strict_json_schema=edge.strict_json_schema,
                is_enabled=True,
            )
            clones[id(edge.from_agent)].handoffs.append(safe_handoff)
            handoff_tool_names.append(edge.tool_name)
            handoff_tools_by_agent.setdefault(edge.from_agent.name, []).append(
                edge.tool_name
            )
            handoff_targets[(edge.from_agent.name, edge.tool_name)] = (
                edge.destination.name
            )

        return PreparedTarget(
            framework=FRAMEWORK_NAME,
            runtime_agent=clones[id(target)],
            spec=spec,
            tool_names=tuple(tool_names),
            gateway=gateway,
            world_state=world_state,
            event_sink=event_sink,
            metadata={
                "capture_holder": capture_holder,
                "tool_risks": tool_risks,
                "tools_by_agent": tools_by_agent,
                "handoff_tool_names": tuple(dict.fromkeys(handoff_tool_names)),
                "handoff_tools_by_agent": {
                    name: tuple(dict.fromkeys(names))
                    for name, names in handoff_tools_by_agent.items()
                },
                "handoff_targets": handoff_targets,
                "run_context": run_context,
                "consumed": False,
            },
        )

    async def run(
        self,
        prepared: PreparedTarget,
        input_text: str | Sequence[ConversationTurn],
        *,
        run_id: str,
        max_turns: int,
        followup_turns: Sequence[ConversationTurn] = (),
        scenario_id: str | None = None,
        target_id: str | None = None,
    ) -> CanonicalRun:
        _require_sdk()
        if prepared.framework != FRAMEWORK_NAME:
            raise TypeError(f"Prepared target belongs to {prepared.framework!r}")
        if prepared.metadata.get("consumed"):
            raise RuntimeError("a prepared target may be run only once")
        if max_turns < 1:
            raise ValueError("max_turns must be at least one")
        scripted = _scripted_followups(followup_turns)
        prepared.metadata["consumed"] = True

        started_at = utc_now()
        initial_world = _world_snapshot(_world_for(prepared))
        capture = _Capture(run_id=run_id, sink=prepared.event_sink)
        holder = cast(dict[str, _Capture | None], prepared.metadata["capture_holder"])
        holder["capture"] = capture
        runner_input, seeded_turns = _runner_input(input_text)
        for role, content, turn_id, turn_metadata in seeded_turns:
            await capture.event(
                (
                    CanonicalEventType.USER_TURN
                    if role == ConversationRole.USER
                    else CanonicalEventType.ASSISTANT_OUTPUT
                ),
                {"text": content, "turn_id": turn_id},
                metadata={**turn_metadata, "scenario_input": True},
            )

        termination = RunTermination.COMPLETED
        termination_reason: str | None = None
        final_output: str | None = None
        stages_executed = 0
        delivered = 0
        stage_response_start = 0
        try:
            run_kwargs: dict[str, Any] = {
                "max_turns": max_turns,
                "hooks": _CapturingHooks(
                    capture, getattr(prepared.gateway, "budgets", None), prepared.gateway
                ),
                "run_config": RunConfig(
                    tracing_disabled=True,
                    trace_include_sensitive_data=False,
                    tool_execution=ToolExecutionConfig(
                        max_function_tool_concurrency=_MAX_FUNCTION_TOOL_CONCURRENCY
                    ),
                    tool_not_found_behavior="raise_error",
                    tool_name_collision_policy="error",
                ),
            }
            run_context = prepared.metadata.get("run_context")
            if run_context is not None:
                run_kwargs["context"] = run_context

            # One prepared target, one gateway, one budget, one capture: the
            # stages below are phases of a single scenario, not separate runs.
            agent: Any = prepared.runtime_agent
            stage_input: Any = runner_input
            completed = False
            while True:
                stage_response_start = len(capture.responses)
                stages_executed += 1
                result = await Runner.run(agent, stage_input, **run_kwargs)
                if delivered >= len(scripted):
                    completed = True
                    break
                remaining = max_turns - _model_turns_used(capture)
                if remaining < 1:
                    termination = RunTermination.MAX_MODEL_TURNS
                    termination_reason = (
                        f"The scenario's {max_turns}-turn model budget was spent "
                        f"before scripted user turn {delivered + 1} of "
                        f"{len(scripted)} could be delivered."
                    )
                    await capture.event(
                        CanonicalEventType.ERROR,
                        {
                            "error_type": "BudgetExceeded",
                            "resource": "model_turns",
                            "message": termination_reason,
                        },
                    )
                    break
                turn = scripted[delivered]
                # Continuation, not replay: the SDK's own input view carries the
                # tool calls and results it already produced, so nothing is
                # re-executed and the gateway is never consulted again for them.
                stage_input = [
                    *result.to_input_list(),
                    {"role": turn.role.value, "content": turn.content},
                ]
                agent = result.last_agent
                await capture.event(
                    CanonicalEventType.USER_TURN,
                    {"text": turn.content, "turn_id": turn.turn_id},
                    metadata={
                        **dict(turn.metadata),
                        "scenario_input": True,
                        "followup_index": delivered,
                    },
                )
                delivered += 1
                run_kwargs["max_turns"] = remaining

            if completed:
                value = result.final_output
                if value is not None:
                    if isinstance(value, str):
                        final_output = value[:100_000]
                    else:
                        encoded = json.dumps(
                            _json_value(value),
                            ensure_ascii=False,
                            allow_nan=False,
                            sort_keys=True,
                        )
                        final_output = encoded[:100_000]
                await capture.event(
                    CanonicalEventType.FINAL_OUTPUT,
                    {"text": final_output},
                )
        except MaxTurnsExceeded as exc:
            termination = RunTermination.MAX_MODEL_TURNS
            termination_reason = str(exc)[:4_000]
            await capture.event(
                CanonicalEventType.ERROR,
                {"error_type": type(exc).__name__, "message": termination_reason},
            )
        except BudgetExceeded as exc:
            termination = {
                "model_turns": RunTermination.MAX_MODEL_TURNS,
                "tool_calls": RunTermination.MAX_TOOL_CALLS,
                "tokens": RunTermination.TOKEN_BUDGET,
                "cost_usd": RunTermination.COST_BUDGET,
                "wall_time": RunTermination.WALL_CLOCK_TIMEOUT,
            }.get(exc.resource, RunTermination.ADAPTER_ERROR)
            termination_reason = str(exc)[:4_000]
            await capture.event(
                CanonicalEventType.ERROR,
                {
                    "error_type": type(exc).__name__,
                    "resource": exc.resource,
                    "limit": exc.limit,
                    "observed": exc.observed,
                    "message": termination_reason,
                },
            )
        except ModelBehaviorError as exc:
            unknown_calls = await _record_unknown_tool_calls(
                capture, prepared, prepared.gateway, start_index=stage_response_start
            )
            if unknown_calls:
                termination = RunTermination.COMPLETED
                termination_reason = (
                    f"Agent attempted {unknown_calls} undeclared tool call(s); "
                    "all were blocked before execution."
                )
            else:
                termination = RunTermination.PROVIDER_ERROR
                termination_reason = str(exc)[:4_000]
            await capture.event(
                CanonicalEventType.ERROR,
                {
                    "error_type": type(exc).__name__,
                    "message": termination_reason,
                    "agent_evaluable": bool(unknown_calls),
                },
            )
        except asyncio.CancelledError:
            termination = RunTermination.CANCELLED
            termination_reason = "Agent run was cancelled."
            await capture.event(
                CanonicalEventType.ERROR,
                {"error_type": "CancelledError", "message": termination_reason},
            )
        except AgentsException as exc:
            termination = RunTermination.ADAPTER_ERROR
            termination_reason = _explain_failure(exc)
            await capture.event(
                CanonicalEventType.ERROR,
                {"error_type": type(exc).__name__, "message": termination_reason},
            )
        except Exception as exc:
            termination = RunTermination.ADAPTER_ERROR
            termination_reason = _explain_failure(exc)
            await capture.event(
                CanonicalEventType.ERROR,
                {"error_type": type(exc).__name__, "message": termination_reason},
            )
        finally:
            holder["capture"] = None

        await _record_ignored_handoffs(capture, prepared)
        ended_at = utc_now()
        return CanonicalRun(
            run_id=run_id,
            scenario_id=scenario_id or run_id,
            target_id=target_id or prepared.spec.spec_id,
            started_at=started_at,
            ended_at=ended_at,
            termination=termination,
            termination_reason=termination_reason,
            events=tuple(capture.events),
            tool_attempts=tuple(capture.attempts),
            tool_outcomes=tuple(capture.outcomes),
            state_transitions=_state_transitions(prepared, capture),
            initial_world_state=initial_world,
            final_world_state=_world_snapshot(_world_for(prepared)),
            final_output=final_output,
            usage=_usage_metrics(capture.responses),
            latency_ms=max(0.0, (ended_at - started_at).total_seconds() * 1_000),
            provider_request_ids=tuple(capture.request_ids),
            metadata={
                "framework": FRAMEWORK_NAME,
                "framework_version": _sdk_version(),
                "usage_unknown": any(
                    response.raw_usage is None for response in capture.responses
                ),
                # Omitted entirely for a scenario with no scripted follow-up, so
                # an ordinary run's metadata is unchanged.
                **(
                    {
                        "stages_executed": stages_executed,
                        "followups_delivered": delivered,
                        "followups_undelivered": len(scripted) - delivered,
                    }
                    if scripted
                    else {}
                ),
            },
        )


__all__ = ["FRAMEWORK_NAME", "OpenAIAgentsAdapter"]
