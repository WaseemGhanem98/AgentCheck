"""Adapter for one trusted local PydanticAI ``Agent``.

The second framework AgentCheck supports, and the reason it was chosen is a
safety property rather than popularity: a PydanticAI tool is a ``Tool`` object
carrying a declared JSON Schema and a separate ``function``, so a replacement
tool can be built from the schema alone and the original callable is never
referenced. Interception happens before any target handler runs, which is the
same guarantee the OpenAI Agents adapter makes and the one AgentCheck cannot
trade away.

Everything outside this module stays framework-neutral: the gateway, worker,
budgets, oracles, and artifacts are the ones the first adapter already used.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect as _inspect
import json
from importlib import metadata as importlib_metadata
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

from agentcheck.domain.agent_spec import (
    AgentProperty,
    AgentSpec,
    CapabilitiesSpec,
    InstructionsSpec,
    IdentitySpec,
    InspectionProvenance,
    InterfaceSpec,
    ObservabilitySpec,
    RuntimeSpec,
    SourceKind,
    SourceReference,
    SpecEvidence,
    ToolDefinition,
    ToolRiskAssertion,
    ToolRiskSpec,
    ToolsSpec,
)
from agentcheck.domain.base import canonical_hash, utc_now
from agentcheck.domain.run import (
    CanonicalEvent,
    CanonicalEventType,
    CanonicalRun,
    RunTermination,
    StateTransition,
    ToolAttempt,
    ToolError,
    ToolOutcome,
    ToolOutcomeStatus,
    UsageMetrics,
)
from agentcheck.domain.scenario import ConversationRole, ConversationTurn
from agentcheck.inspect.capabilities import extract_capabilities
from agentcheck.inspect.risk_authority import declared_risk_for, resolve_tool_risk
from agentcheck.runner.launch_barrier import LaunchBarrier
from agentcheck.runner.tool_gateway import CallReservation
from agentcheck.schema_safety import UnsafeSchemaReference, offline_validator
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from agentcheck.config import ToolRiskDeclaration

from .base import (
    portable_identity,
    AdapterDependencyError,
    EventSinkProtocol,
    FrameworkAdapter,
    PreflightReport,
    PreparedTarget,
    SupportIssue,
    ToolGatewayProtocol,
    missing_extra_message,
)

FRAMEWORK_NAME = "pydantic_ai"

# Exact-equality like the OpenAI adapter's 0.20.x pin, and for the same reason:
# instructions and validators are read from private attributes, so an unverified
# version could silently yield a wrong AgentSpec rather than an honest failure.
SUPPORTED_SDK_MINOR = (2, 32)

# Capabilities the framework installs on every agent. Anything beyond these is
# target-supplied middleware that wraps node execution, so it is rejected
# rather than silently dropped when the sanitized agent is rebuilt. Pinned to
# one minor precisely so this list cannot drift underneath the check.
_DEFAULT_CAPABILITIES = frozenset({"ToolSearch", "PendingMessageDrainCapability"})

_SDK_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - import guard mirrors the OpenAI adapter
    from pydantic_ai import Agent
    from pydantic_ai.messages import (
        TextPart,
        ToolCallPart,
    )
    from pydantic_ai.models import Model
    from pydantic_ai.tools import RunContext, Tool
except Exception as exc:  # pragma: no cover - exercised without the extra
    _SDK_IMPORT_ERROR = exc
    Agent = Any  # type: ignore[assignment,misc]
    Model = object  # type: ignore[assignment,misc]



def _require_sdk() -> None:
    if _SDK_IMPORT_ERROR is not None:
        raise AdapterDependencyError(
            missing_extra_message("PydanticAI support", "pydantic-ai")
        ) from _SDK_IMPORT_ERROR


def _sdk_version() -> str | None:
    for name in ("pydantic-ai-slim", "pydantic-ai"):
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def _supported_sdk_version(version: str | None) -> bool:
    if version is None:
        return False
    numeric = version.split("+", 1)[0].split("-", 1)[0].split(".")
    try:
        return tuple(int(part) for part in numeric[:2]) == SUPPORTED_SDK_MINOR
    except ValueError:
        return False


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
    return AgentProperty(
        value=value,
        source=SourceReference(kind=kind, locator=locator),
        confidence=confidence,
        evidence=(
            SpecEvidence(
                evidence_id=_evidence_id(locator), summary=summary, locator=locator
            ),
        ),
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


def _schema_validation_errors(schema: Mapping[str, Any], arguments: Any) -> list[str]:
    try:
        validator = offline_validator(schema)
    except (SchemaError, UnsafeSchemaReference) as exc:
        return [f"invalid or unsafe tool schema: {exc}"]
    errors = sorted(validator.iter_errors(arguments), key=lambda error: list(error.path))
    return [
        f"{'.'.join(str(item) for item in error.absolute_path) or '$'}: {error.message}"
        for error in errors
    ]


def _agent_toolset(target: Any) -> Any | None:
    """The agent's own function toolset, or None when the shape is unexpected."""

    toolsets = getattr(target, "toolsets", ())
    for toolset in toolsets:
        if isinstance(getattr(toolset, "tools", None), dict):
            return toolset
    return None


def _agent_tools(target: Any) -> dict[str, Any]:
    toolset = _agent_toolset(target)
    if toolset is None:
        return {}
    return dict(toolset.tools)


def _static_instruction_parts(target: Any) -> tuple[list[str], list[str]]:
    """Return (static parts, reasons the instructions are not fully static).

    ``Agent.instructions`` is a decorator, not a value; the configured text lives
    in private attributes. Anything callable there is a dynamic instruction,
    which is target code AgentCheck will not execute during inspection.
    """

    static: list[str] = []
    dynamic: list[str] = []
    for item in getattr(target, "_instructions", ()) or ():
        if isinstance(item, str):
            static.append(item)
        else:
            dynamic.append("agent.instructions")
    for item in getattr(target, "_system_prompts", ()) or ():
        if isinstance(item, str):
            static.append(item)
        else:
            dynamic.append("agent.system_prompt")
    if getattr(target, "_system_prompt_functions", None):
        dynamic.append("agent.system_prompt (function)")
    return static, dynamic


def _model_identity(target: Any) -> tuple[str | None, str | None]:
    """(model name, provider) read from the configured model object."""

    model = getattr(target, "model", None)
    if model is None:
        return None, None
    if isinstance(model, str):
        provider, _, name = model.partition(":")
        return (name or model), (provider or None)
    raw_name = getattr(model, "model_name", None)
    raw_system = getattr(model, "system", None)
    resolved_name: str = raw_name if isinstance(raw_name, str) else type(model).__name__
    resolved_system: str | None = raw_system if isinstance(raw_system, str) else None
    return resolved_name, resolved_system


def _output_schema(target: Any) -> dict[str, Any] | None:
    """The declared structured-output schema, when the agent declares one."""

    output_type = getattr(target, "output_type", None)
    if output_type is None or output_type is str:
        return None
    schema = getattr(output_type, "model_json_schema", None)
    if callable(schema):
        try:
            built = schema()
        except Exception:  # pragma: no cover - defensive
            return None
        return cast("dict[str, Any]", built) if isinstance(built, dict) else None
    return None


def _tool_definition(
    name: str,
    tool: Any,
    *,
    declared_tool_risk: "Mapping[str, ToolRiskDeclaration] | None" = None,
) -> tuple[ToolDefinition, ToolRiskAssertion]:
    tool_def = getattr(tool, "tool_def", None)
    schema = getattr(tool_def, "parameters_json_schema", None)
    description = getattr(tool_def, "description", None)
    resolved = description if isinstance(description, str) and description else None
    # The same shared resolver the OpenAI Agents adapter uses. These two flags
    # were once hardcoded False here, and everything downstream reads them:
    # fault generation skips a tool that is not state-changing, and the
    # ambiguous-timeout case is destructive-only. A PydanticAI target therefore
    # received no fault family for any tool, while `inspect` printed a summary
    # of zero state-changing actions directly above a capability listing that
    # described the same tool as state-changing -- capability extraction
    # classifies independently, so only the adapter disagreed.
    #
    # Inference alone is still never authoritative, and reported as such. A
    # developer declaration in ``declared_tool_risk`` is the only thing here
    # entitled to override it; see `agentcheck.inspect.risk_authority`.
    declared = declared_risk_for(name, declared_tool_risk)
    state_changing, destructive, assertion = resolve_tool_risk(
        name, resolved, declared=declared
    )
    definition = ToolDefinition(
        name=name,
        description=resolved,
        input_schema=dict(schema) if isinstance(schema, Mapping) else {},
        output_schema=None,
        state_changing=state_changing,
        destructive=destructive,
        replaceable=True,
    )
    return definition, assertion


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:  # pragma: no cover - defensive
            return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _json_object(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(item) for key, item in value.items()}


@dataclasses.dataclass
class _Capture:
    """Builds canonical domain objects while a PydanticAI run is in progress.

    Deliberately self-contained rather than shared with the OpenAI adapter: the
    shape looks similar today, but unifying two implementations on the evidence
    of two is how a false abstraction gets frozen in. A third adapter can decide.
    """

    run_id: str
    sink: EventSinkProtocol | None
    events: list[CanonicalEvent] = dataclasses.field(default_factory=list)
    transition_links: dict[str, tuple[str, str]] = dataclasses.field(
        default_factory=dict
    )
    attempts: list[ToolAttempt] = dataclasses.field(default_factory=list)
    outcomes: list[ToolOutcome] = dataclasses.field(default_factory=list)
    request_ids: list[str] = dataclasses.field(default_factory=list)
    usage: list[Any] = dataclasses.field(default_factory=list)
    # Populated once per model response, before PydanticAI's own executor
    # dispatches any of its tool calls -- which, unlike the OpenAI adapter,
    # it does concurrently by default. See `_CapturingModel.request` and
    # `_make_safe_tool`.
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
            if _inspect.isawaitable(emitted):
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
    ) -> ToolAttempt:
        source_event_ids: tuple[str, ...] = ()
        for source_event in reversed(self.events):
            if source_event.event_type == CanonicalEventType.MODEL_RESPONSE:
                source_event_ids = (source_event.event_id,)
                break
        index = len(self.attempts)
        event = await self.event(
            CanonicalEventType.TOOL_ATTEMPT,
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "call_id": call_id,
                "raw_arguments_sha256": hashlib.sha256(
                    raw_arguments.encode("utf-8")
                ).hexdigest(),
                "validation_errors": list(validation_errors),
            },
            source_event_ids=source_event_ids,
        )
        attempt = ToolAttempt(
            attempt_id=f"{self.run_id}:attempt:{index:04d}",
            event_id=event.event_id,
            sequence=index,
            tool_name=tool_name,
            arguments=_json_object(arguments),
            timestamp=event.timestamp,
            state_changing=state_changing,
            destructive=destructive,
        )
        self.attempts.append(attempt)
        return attempt

    async def tool_result(
        self,
        *,
        attempt: ToolAttempt,
        status: ToolOutcomeStatus,
        result: Any,
        error: ToolError | None,
        started_at: Any,
        gateway_outcome_id: str | None = None,
        state_transition_ids: Sequence[str] = (),
        gateway_metadata: Mapping[str, Any] | None = None,
    ) -> ToolOutcome:
        # Remapped here, not left for the caller to do afterwards: a gateway
        # transition ID must become its canonical run-scoped ID before it is
        # ever stored on a ToolOutcome, or two calls whose outcomes are
        # recorded out of the order their `transition_links` entries were
        # populated could end up with an outcome referencing an ID no
        # `StateTransition` in the final run actually carries.
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
        index = len(self.outcomes)
        event = await self.event(
            CanonicalEventType.TOOL_RESULT,
            {
                "tool_name": attempt.tool_name,
                "attempt_id": attempt.attempt_id,
                "status": status.value,
                "result": _json_value(result),
                "error": error.model_dump(mode="json") if error is not None else None,
            },
            source_event_ids=(attempt.event_id,),
        )
        outcome = ToolOutcome(
            outcome_id=f"{self.run_id}:outcome:{index:04d}",
            attempt_id=attempt.attempt_id,
            event_id=event.event_id,
            tool_name=attempt.tool_name,
            status=status,
            result=_json_value(result),
            error=error,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=max(0.0, (ended_at - started_at).total_seconds() * 1_000),
            state_transition_ids=tuple(canonical_transition_ids),
            metadata=_json_object(dict(gateway_metadata or {}))
            | ({"gateway_outcome_id": gateway_outcome_id} if gateway_outcome_id else {}),
        )
        self.outcomes.append(outcome)
        return outcome



def _tool_error_from_gateway(error: Any, *, fallback_code: str) -> ToolError | None:
    if error is None:
        return None
    if isinstance(error, ToolError):
        return error
    if isinstance(error, Mapping):
        retryable = error.get("retryable")
        return ToolError(
            code=str(error.get("code") or fallback_code),
            message=str(error.get("message") or "Simulated tool failure"),
            retryable=retryable if isinstance(retryable, bool) else None,
            details=_json_object(error.get("details", {})),
        )
    return ToolError(code=fallback_code, message=str(error)[:4_000])


def _gateway_result_parts(
    value: Any,
) -> tuple[
    ToolOutcomeStatus, Any, ToolError | None, str | None, tuple[str, ...], dict[str, Any]
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
    raw_ids = getattr(value, "state_transition_ids", ())
    transition_ids = (
        tuple(item for item in raw_ids if isinstance(item, str))
        if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes))
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
            code=f"tool_{status.value}", message=f"Simulated tool {status.value}"
        )
    return (
        status,
        result,
        error,
        str(outcome_id) if outcome_id is not None else None,
        transition_ids,
        gateway_metadata,
    )


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

    Mirrors the OpenAI adapter's helper of the same name. The fallback
    covers a gateway without batch-planning support, and any call
    `_CapturingModel._reserve_tool_calls` deliberately left unplanned.
    """

    reservation = capture.pending_reservations.pop(call_id, None)
    barrier = capture.pending_barrier
    if reservation is not None and barrier is not None and hasattr(gateway, "commit"):
        return await barrier.run_in_order(
            reservation.rank,
            lambda: cast(Any, gateway).commit(reservation, world_state),
        )
    result = gateway.invoke(tool_name, arguments, world_state)
    if _inspect.isawaitable(result):
        return await result
    return result


def _model_visible_tool_output(
    status: ToolOutcomeStatus, result: Any, error: ToolError | None
) -> str:
    """What the model is allowed to see. Gateway internals never leak."""

    if error is None and status not in {
        ToolOutcomeStatus.ERROR,
        ToolOutcomeStatus.TIMEOUT,
        ToolOutcomeStatus.BLOCKED,
    }:
        if isinstance(result, str):
            return result
        return json.dumps(
            _json_value(result), ensure_ascii=False, allow_nan=False, sort_keys=True
        )
    payload = {
        "status": status.value,
        "result": _json_value(result),
        "error": _json_value(error) if error is not None else None,
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _consume_adapter_tool_budget(gateway: ToolGatewayProtocol) -> ToolError | None:
    tracker = getattr(gateway, "budgets", None)
    consume = getattr(tracker, "consume_tool_call", None)
    if not callable(consume):
        return None
    try:
        consume()
    except Exception as exc:  # BudgetExceeded, kept adapter-neutral
        resource = getattr(exc, "resource", None)
        if resource is None:
            raise
        return ToolError(
            code=f"{resource}_budget_exceeded",
            message=str(exc),
            retryable=False,
            details={
                "limit": getattr(exc, "limit", None),
                "observed": getattr(exc, "observed", None),
            },
        )
    return None


def _state_transitions(
    prepared: PreparedTarget, capture: _Capture
) -> tuple[StateTransition, ...]:
    candidates: Any = getattr(prepared.gateway, "state_transitions", None)
    if candidates is None:
        candidates = getattr(prepared.gateway, "transitions", None)
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return ()
    normalized: list[StateTransition] = []
    for item in candidates:
        if not isinstance(item, StateTransition):
            continue
        link = capture.transition_links.get(item.transition_id)
        if link is None:
            continue
        canonical_id, attempt_id = link
        normalized.append(
            item.model_copy(
                update={"transition_id": canonical_id, "attempt_id": attempt_id}
            )
        )
    return tuple(normalized)


def _make_safe_tool(
    *,
    definition: ToolDefinition,
    gateway: ToolGatewayProtocol,
    world_state: Any,
    capture_holder: dict[str, "_Capture | None"],
    state_changing: bool,
    destructive: bool,
) -> Any:
    """A replacement ``Tool`` built from the declared schema alone.

    The original callable is left on the source agent. It is never copied,
    wrapped, or called, so there is no code path from a model decision to a
    target handler.
    """

    name = definition.name

    async def invoke(ctx: "RunContext[Any]", **kwargs: Any) -> str:
        capture = capture_holder.get("capture")
        if capture is None:
            raise RuntimeError(
                "prepared AgentCheck target was invoked outside adapter.run()"
            )
        arguments = {str(key): value for key, value in kwargs.items()}
        raw_arguments = json.dumps(
            _json_value(arguments), ensure_ascii=False, sort_keys=True
        )
        call_id = str(getattr(ctx, "tool_call_id", None) or "") or (
            f"agentcheck-{name}-{len(capture.attempts) + 1}"
        )
        attempt = await capture.tool_attempt(
            tool_name=name,
            arguments=arguments,
            raw_arguments=raw_arguments,
            call_id=call_id,
            validation_errors=(),
            state_changing=state_changing,
            destructive=destructive,
        )
        started_at = utc_now()
        try:
            result = await _invoke_reserved_or_direct(
                gateway=gateway,
                capture=capture,
                call_id=call_id,
                tool_name=name,
                arguments=arguments,
                world_state=world_state,
            )
        except BaseException as exc:
            controlled = getattr(exc, "outcome", None)
            if isinstance(controlled, ToolOutcome):
                (
                    status,
                    value,
                    error,
                    outcome_id,
                    transition_ids,
                    metadata,
                ) = _gateway_result_parts(controlled)
                outcome = await capture.tool_result(
                    attempt=attempt,
                    status=status,
                    result=value,
                    error=error,
                    started_at=started_at,
                    gateway_outcome_id=outcome_id,
                    state_transition_ids=transition_ids,
                    gateway_metadata=metadata,
                )
                del outcome
                return _model_visible_tool_output(status, value, error)
            await capture.event(
                CanonicalEventType.ERROR,
                {
                    "layer": "tool_gateway",
                    "tool_name": name,
                    "attempt_id": attempt.attempt_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:4_000],
                },
                source_event_ids=(attempt.event_id,),
            )
            raise

        (
            status,
            value,
            error,
            outcome_id,
            transition_ids,
            metadata,
        ) = _gateway_result_parts(result)
        await capture.tool_result(
            attempt=attempt,
            status=status,
            result=value,
            error=error,
            started_at=started_at,
            gateway_outcome_id=outcome_id,
            state_transition_ids=transition_ids,
            gateway_metadata=metadata,
        )
        return _model_visible_tool_output(status, value, error)

    invoke.__name__ = name
    return Tool.from_schema(
        invoke,
        name=name,
        description=definition.description or "",
        json_schema=dict(definition.input_schema),
        takes_ctx=True,
    )


class _CapturingModel(Model):
    """Wraps the target's own model to record turns and enforce the scenario budget.

    The model is not replaced or scripted here: this only observes. Budget
    enforcement lives on the gateway's tracker so counts stay cumulative across
    interactive stages, exactly as they do for the first adapter.
    """

    def __init__(
        self,
        inner: Any,
        capture: _Capture,
        budgets: Any,
        *,
        gateway: "ToolGatewayProtocol | None" = None,
        schema_by_tool: Mapping[str, dict[str, Any]] | None = None,
    ) -> None:
        self._inner = inner
        self._capture = capture
        self._budgets = budgets
        self._gateway = gateway
        self._schema_by_tool = dict(schema_by_tool or {})

    @property
    def model_name(self) -> str:
        name = getattr(self._inner, "model_name", None)
        return name if isinstance(name, str) else type(self._inner).__name__

    @property
    def system(self) -> str:
        system = getattr(self._inner, "system", None)
        return system if isinstance(system, str) else FRAMEWORK_NAME

    async def request(
        self,
        messages: list[Any],
        model_settings: Any,
        model_request_parameters: Any,
    ) -> Any:
        instructions = None
        for message in messages:
            candidate = getattr(message, "instructions", None)
            if isinstance(candidate, str):
                instructions = candidate
        await self._capture.event(
            CanonicalEventType.MODEL_REQUEST,
            {
                "agent_name": None,
                "input_item_count": len(messages),
                "system_instructions_present": instructions is not None,
                "system_instructions_sha256": (
                    hashlib.sha256(instructions.encode("utf-8")).hexdigest()
                    if instructions is not None
                    else None
                ),
            },
        )
        consume = getattr(self._budgets, "consume_model_turn", None)
        if callable(consume):
            consume()

        response = await self._inner.request(
            messages, model_settings, model_request_parameters
        )

        usage = getattr(response, "usage", None)
        if usage is not None:
            self._capture.usage.append(usage)
        request_id = getattr(response, "provider_response_id", None)
        if isinstance(request_id, str) and request_id not in self._capture.request_ids:
            self._capture.request_ids.append(request_id)
        parts = list(getattr(response, "parts", ()))
        await self._capture.event(
            CanonicalEventType.MODEL_RESPONSE,
            {
                "agent_name": None,
                "response_id": request_id,
                "request_id": request_id,
                "output_item_count": len(parts),
                "usage_known": usage is not None,
            },
        )
        add_tokens = getattr(self._budgets, "add_tokens", None)
        if callable(add_tokens) and usage is not None:
            total = getattr(usage, "total_tokens", None)
            if not isinstance(total, int):
                inp = getattr(usage, "input_tokens", None)
                out = getattr(usage, "output_tokens", None)
                total = inp + out if isinstance(inp, int) and isinstance(out, int) else None
            add_tokens(total if isinstance(total, int) else None)
        for part in parts:
            text = getattr(part, "content", None)
            if isinstance(part, TextPart) and isinstance(text, str) and text:
                await self._capture.event(
                    CanonicalEventType.ASSISTANT_OUTPUT,
                    {"text": text, "agent_name": None},
                )
        self._reserve_tool_calls(parts)
        return response

    def _reserve_tool_calls(self, parts: Sequence[Any]) -> None:
        """Deterministically plan this response's tool calls before dispatch.

        Unlike the OpenAI adapter, PydanticAI dispatches a model response's
        tool calls concurrently *by default* -- there is no serialized
        baseline to opt out of. This runs synchronously, before returning
        control to PydanticAI's own tool-execution graph, so which
        reservation each call gets can never depend on how that graph
        happens to schedule the resulting tasks.
        """

        gateway = self._gateway
        if gateway is None or not hasattr(gateway, "plan_batch"):
            return
        calls = [part for part in parts if isinstance(part, ToolCallPart)]
        if not calls:
            return
        plannable: list[tuple[str, str, dict[str, Any]]] = []
        for call in calls:
            call_id = str(getattr(call, "tool_call_id", "") or "")
            schema = self._schema_by_tool.get(call.tool_name)
            if not call_id or schema is None:
                continue
            raw_args = call.args
            if isinstance(raw_args, str):
                try:
                    parsed_args: Any = json.loads(raw_args)
                except (TypeError, ValueError):
                    continue
            elif raw_args is None:
                parsed_args = {}
            else:
                parsed_args = raw_args
            if not isinstance(parsed_args, dict):
                continue
            if _schema_validation_errors(schema, parsed_args):
                # PydanticAI will not dispatch this call to our tool function
                # at all -- it answers with its own retry prompt instead.
                # Planning a reservation for it would consume a fixture/budget
                # slot for a call that never commits.
                continue
            plannable.append((call_id, call.tool_name, parsed_args))
        if not plannable:
            return
        reservations = cast(Any, gateway).plan_batch(
            [(name, arguments) for _, name, arguments in plannable]
        )
        self._capture.pending_reservations = {
            call_id: reservation
            for (call_id, _, _), reservation in zip(plannable, reservations)
        }
        self._capture.pending_barrier = LaunchBarrier(len(reservations))


def _usage_metrics(entries: Sequence[Any]) -> UsageMetrics:
    """Unknown stays unknown: a missing count is null, never zero."""

    if not entries:
        return UsageMetrics()
    inputs = outputs = 0
    cost_total = 0.0
    cost_known = True
    for usage in entries:
        value_in = getattr(usage, "input_tokens", None)
        value_out = getattr(usage, "output_tokens", None)
        if not isinstance(value_in, int) or not isinstance(value_out, int):
            return UsageMetrics()
        inputs += value_in
        outputs += value_out
        cost = getattr(usage, "cost", None)
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
            cost_total += float(cost)
        else:
            cost_known = False
    return UsageMetrics(
        input_tokens=inputs,
        output_tokens=outputs,
        total_tokens=inputs + outputs,
        cost_usd=cost_total if cost_known else None,
    )


async def _record_blocked_tool_calls(
    capture: _Capture, prepared: PreparedTarget, messages: Sequence[Any], start: int
) -> int:
    """Turn the framework's unknown-tool retries into recorded blocked attempts.

    PydanticAI answers a call for an undeclared tool with a ``RetryPromptPart``
    instead of raising, so without this the attempt would be invisible. It is
    deterministic model behaviour worth asserting on, and no handler ran.
    """

    allowed = set(prepared.tool_names)
    recorded = 0
    for message in list(messages)[start:]:
        for part in getattr(message, "parts", ()):
            if not isinstance(part, ToolCallPart):
                continue
            if part.tool_name in allowed:
                continue
            args = part.args
            if isinstance(args, str):
                try:
                    decoded = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    decoded = {}
            else:
                decoded = args if isinstance(args, dict) else {}
            attempt = await capture.tool_attempt(
                tool_name=part.tool_name,
                arguments={str(k): v for k, v in decoded.items()},
                raw_arguments=json.dumps(_json_value(decoded), sort_keys=True),
                call_id=str(getattr(part, "tool_call_id", "") or "unknown"),
                validation_errors=("tool is not declared by the inspected agent",),
                state_changing=False,
                destructive=False,
            )
            budget_error = _consume_adapter_tool_budget(prepared.gateway)
            await capture.tool_result(
                attempt=attempt,
                status=ToolOutcomeStatus.BLOCKED,
                result=None,
                error=budget_error
                or ToolError(
                    code="unknown_tool",
                    message=(
                        f"Tool {part.tool_name!r} is not declared and was blocked."
                    ),
                    retryable=False,
                    details={"supported_tools": sorted(allowed)},
                ),
                started_at=utc_now(),
            )
            recorded += 1
    return recorded


class PydanticAIAdapter(FrameworkAdapter):
    """Adapter for one trusted local PydanticAI ``Agent``."""

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
        locator = source or f"{type(target).__module__}.{type(target).__name__}"
        framework_version = _sdk_version()
        model_name, provider = _model_identity(target)
        static, _dynamic = _static_instruction_parts(target)
        instructions = "\n\n".join(static) if static else None
        output_schema = _output_schema(target)

        definitions: list[ToolDefinition] = []
        tool_properties: list[AgentProperty[Any]] = []
        tool_risk_assertions: dict[str, ToolRiskAssertion] = {}
        fingerprint_tools: list[Any] = []
        for index, (name, tool) in enumerate(sorted(_agent_tools(target).items())):
            definition, risk_assertion = _tool_definition(
                name, tool, declared_tool_risk=declared_tool_risk
            )
            definitions.append(definition)
            tool_risk_assertions[name] = risk_assertion
            fingerprint_tools.append(definition.model_dump(mode="json"))
            tool_properties.append(
                _property(
                    definition,
                    locator=f"{locator}.agent.tools[{index}]",
                    summary="Tool name, description, and JSON Schema read from the Tool object.",
                    kind=SourceKind.TOOL_SCHEMA,
                    confidence=0.9,
                    inferred=True,
                    authoritative=False,
                )
            )

        capabilities = [
            _property(
                extracted.capability,
                locator=f"tool:{extracted.tool_name}",
                summary=(
                    "Argument surface read from the declared schema; action kind and "
                    "side-effect risk are inferred and not authoritative."
                ),
                kind=SourceKind.TOOL_SCHEMA,
                confidence=extracted.confidence,
                inferred=True,
                authoritative=False,
            )
            for extracted in extract_capabilities(definitions)
        ]

        fingerprint: dict[str, Any] = {
            "source": locator,
            "name": getattr(target, "name", None),
            "framework_version": framework_version,
            "model": model_name,
            "instructions_hash": (
                canonical_hash(instructions) if instructions is not None else None
            ),
            "tools": fingerprint_tools,
        }
        spec_id, legacy_spec_id = portable_identity(
            fingerprint,
            location_locator=locator,
            identity_locator=identity_locator,
        )

        return AgentSpec(
            spec_id=spec_id,
            legacy_spec_id=legacy_spec_id,
            identity=IdentitySpec(
                name=_property(
                    getattr(target, "name", None) or "agent",
                    locator=f"{locator}.agent.name",
                    summary="Agent name read from the runtime object.",
                ),
                framework=_property(
                    FRAMEWORK_NAME,
                    locator=f"{locator}.agent",
                    summary="Framework identified from the imported Agent type.",
                    kind=SourceKind.FRAMEWORK_METADATA,
                ),
                framework_version=_property(
                    framework_version,
                    locator="python-package:pydantic-ai-slim",
                    summary="Installed PydanticAI distribution version.",
                    kind=SourceKind.FRAMEWORK_METADATA,
                ),
                provider=(
                    _property(
                        provider,
                        locator=f"{locator}.agent.model.system",
                        summary="Provider reported by the configured model object.",
                        kind=SourceKind.FRAMEWORK_METADATA,
                        confidence=0.9,
                        inferred=True,
                        authoritative=False,
                    )
                    if provider is not None
                    else _unknown_property(
                        None,
                        locator=f"{locator}.agent.model",
                        summary="The configured model does not report a provider.",
                    )
                ),
                model=_property(
                    model_name,
                    locator=f"{locator}.agent.model",
                    summary="Model identifier read from the configured model.",
                ),
            ),
            interface=InterfaceSpec(
                entrypoint=_property(
                    locator,
                    locator=f"{locator}.agent",
                    summary="Entrypoint AgentCheck imported.",
                ),
                input_modalities=_property(
                    ("text",),
                    locator=f"{locator}.agent.run",
                    summary="Agent.run accepts a text prompt.",
                ),
                output_modalities=_property(
                    ("text",),
                    locator=f"{locator}.agent.output_type",
                    summary="Agent output is text or a declared structured type.",
                ),
                input_schema=_unknown_property(
                    None,
                    locator=f"{locator}.agent.run",
                    summary="PydanticAI does not declare an input schema.",
                ),
                output_schema=_property(
                    output_schema,
                    locator=f"{locator}.agent.output_type",
                    summary="Structured output schema declared by output_type.",
                ),
                interactive=_property(
                    True,
                    locator=f"{locator}.agent.run",
                    summary="message_history supports multi-stage continuation.",
                ),
            ),
            instructions=InstructionsSpec(
                system=_property(
                    instructions,
                    locator=f"{locator}.agent.instructions",
                    summary="Static instructions and system prompts read from the agent.",
                    kind=SourceKind.SYSTEM_INSTRUCTION,
                ),
                developer=_unknown_property(
                    None,
                    locator=f"{locator}.agent",
                    summary="PydanticAI has no separate developer instruction channel.",
                ),
            ),
            capabilities=CapabilitiesSpec(items=tuple(capabilities)),
            tools=ToolsSpec(items=tuple(tool_properties)),
            tool_risk=ToolRiskSpec(
                items=tuple(
                    tool_risk_assertions[name] for name in sorted(tool_risk_assertions)
                )
            ),
            runtime=RuntimeSpec(
                max_model_turns=_unknown_property(
                    None,
                    locator=f"{locator}.agent",
                    summary="Turn limits come from the scenario, not the agent.",
                ),
                max_tool_calls=_unknown_property(
                    None,
                    locator=f"{locator}.agent",
                    summary="Tool-call limits come from the scenario, not the agent.",
                ),
                timeout_seconds=_unknown_property(
                    None,
                    locator=f"{locator}.agent",
                    summary="Timeouts come from the scenario, not the agent.",
                ),
                token_budget=_unknown_property(
                    None,
                    locator=f"{locator}.agent",
                    summary="Token budgets come from the scenario, not the agent.",
                ),
                cost_budget_usd=_unknown_property(
                    None,
                    locator=f"{locator}.agent",
                    summary="Cost budgets come from the scenario, not the agent.",
                ),
            ),
            observability=ObservabilitySpec(
                supported_event_types=_property(
                    (
                        "user_turn",
                        "assistant_output",
                        "model_request",
                        "model_response",
                        "tool_attempt",
                        "tool_result",
                        "error",
                        "final_output",
                    ),
                    locator=f"{locator}.adapter",
                    summary="Canonical events this adapter emits.",
                ),
                usage_metrics=_property(
                    ("input_tokens", "output_tokens", "total_tokens"),
                    locator=f"{locator}.adapter",
                    summary="Usage metrics PydanticAI reports per request.",
                ),
                provider_request_ids=_property(
                    True,
                    locator=f"{locator}.adapter",
                    summary="provider_response_id is recorded when the model supplies one.",
                ),
                source_event_links=_property(
                    True,
                    locator=f"{locator}.adapter",
                    summary="Tool events link back to the model response that caused them.",
                ),
            ),
            provenance=InspectionProvenance(
                inspector=__name__,
                inspector_version=_agentcheck_version(),
                inspected_at=utc_now(),
                target=locator,
                sources=(
                    SourceReference(
                        kind=SourceKind.RUNTIME_INTROSPECTION, locator=locator
                    ),
                    SourceReference(
                        kind=SourceKind.FRAMEWORK_METADATA,
                        locator="python-package:pydantic-ai-slim",
                    ),
                ),
            ),
        )

    def preflight(self, target: Any) -> PreflightReport:
        """Decide, before any model call, whether every surface can be replaced."""

        if _SDK_IMPORT_ERROR is not None:
            return PreflightReport(
                framework=FRAMEWORK_NAME,
                issues=(
                    SupportIssue(
                        code="framework_unavailable",
                        message="PydanticAI is not installed in the worker environment.",
                        location="python-package:pydantic-ai-slim",
                    ),
                ),
            )
        issues: list[SupportIssue] = []
        version = _sdk_version()
        if not _supported_sdk_version(version):
            issues.append(
                SupportIssue(
                    code="unsupported_sdk_version",
                    message=(
                        f"Expected pydantic-ai {SUPPORTED_SDK_MINOR[0]}."
                        f"{SUPPORTED_SDK_MINOR[1]}.x, found {version or 'not installed'}."
                    ),
                    location="python-package:pydantic-ai-slim",
                )
            )
        if type(target) is not Agent:
            issues.append(
                SupportIssue(
                    code="unsupported_agent_type",
                    message="This adapter requires an exact pydantic_ai.Agent instance.",
                    location="agent",
                )
            )
            return PreflightReport(framework=FRAMEWORK_NAME, issues=tuple(issues))

        _static, dynamic = _static_instruction_parts(target)
        for location in dict.fromkeys(dynamic):
            issues.append(
                SupportIssue(
                    code="dynamic_instructions",
                    message=(
                        "Instructions are computed by target code, which AgentCheck "
                        "will not execute during inspection."
                    ),
                    location=location,
                )
            )
        if getattr(target, "_output_validators", None):
            issues.append(
                SupportIssue(
                    code="output_validator",
                    message=(
                        "Output validators are executable target code and are not "
                        "replaced by this adapter."
                    ),
                    location="agent.output_validator",
                )
            )
        deps_type = getattr(target, "_deps_type", object)
        if deps_type is not object and deps_type is not type(None):
            issues.append(
                SupportIssue(
                    code="dependency_injection_required",
                    message=(
                        "The agent declares deps_type, so a run needs dependencies "
                        "AgentCheck cannot fabricate safely."
                    ),
                    location="agent.deps_type",
                )
            )
        root = getattr(target, "_root_capability", None)
        declared = [
            type(capability).__name__
            for capability in (getattr(root, "capabilities", None) or ())
            if type(capability).__name__ not in _DEFAULT_CAPABILITIES
        ]
        if declared:
            issues.append(
                SupportIssue(
                    code="unsupported_capability",
                    message=(
                        "The agent declares capabilities, which wrap node execution "
                        f"with target code: {', '.join(sorted(set(declared)))}."
                    ),
                    location="agent.capabilities",
                )
            )
        if getattr(target, "_event_stream_handler", None) is not None:
            issues.append(
                SupportIssue(
                    code="unsupported_event_stream_handler",
                    message=(
                        "An event stream handler is target code invoked during a run."
                    ),
                    location="agent.event_stream_handler",
                )
            )
        toolsets = list(getattr(target, "toolsets", ()) or ())
        own = _agent_toolset(target)
        extra = [t for t in toolsets if t is not own]
        if extra:
            issues.append(
                SupportIssue(
                    code="unsupported_toolset",
                    message=(
                        "Only the agent's own function toolset is supported; external "
                        "toolsets are not replaced."
                    ),
                    location="agent.toolsets",
                )
            )
        for name, tool in sorted(_agent_tools(target).items()):
            if getattr(tool, "takes_ctx", False):
                issues.append(
                    SupportIssue(
                        code="tool_requires_run_context",
                        message=(
                            "The tool takes a RunContext, which carries target "
                            "dependencies AgentCheck does not construct."
                        ),
                        location=f"agent.tools[{name}]",
                    )
                )
            schema = getattr(getattr(tool, "tool_def", None), "parameters_json_schema", None)
            if not isinstance(schema, Mapping):
                issues.append(
                    SupportIssue(
                        code="tool_schema_unavailable",
                        message="The tool does not declare a JSON Schema.",
                        location=f"agent.tools[{name}]",
                    )
                )
        return PreflightReport(framework=FRAMEWORK_NAME, issues=tuple(issues))

    def describe_topology(
        self, target: Any, *, source: str | None = None
    ) -> dict[str, Any] | None:
        """No multi-agent topology is supported yet, so none is described."""

        del target, source
        return None

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
        _require_sdk()
        report = self.preflight(target)
        report.require_supported()
        spec = self.inspect(
            target,
            source=source,
            identity_locator=identity_locator,
            declared_tool_risk=declared_tool_risk,
        )

        capture_holder: dict[str, _Capture | None] = {"capture": None}
        tool_risks: dict[str, tuple[bool, bool]] = {}
        safe_tools = []
        for item in spec.tools.items:
            definition = item.value
            # `definition.state_changing`/`.destructive` are already the fully
            # resolved values -- developer declaration, then framework
            # metadata, then inference, then unknown -- so they are used
            # directly. Re-deriving them from `spec.capabilities` here (as this
            # code once did) reruns pure name/description inference and would
            # silently discard a developer declaration or authoritative
            # framework value the moment one existed.
            risk = (definition.state_changing, definition.destructive)
            tool_risks[definition.name] = risk
            safe_tools.append(
                _make_safe_tool(
                    definition=definition,
                    gateway=gateway,
                    world_state=world_state,
                    capture_holder=capture_holder,
                    state_changing=risk[0],
                    destructive=risk[1],
                )
            )

        static, _dynamic = _static_instruction_parts(target)
        model: Any = getattr(target, "model", None)
        if controlled_model:
            from .pydantic_ai_controlled import ControlledPydanticModel

            model = ControlledPydanticModel(spec.interface.output_schema.value)

        # A NEW Agent built from the inspected surface. The source agent is not
        # mutated and its tool functions are never referenced.
        runtime_agent = Agent(
            model,
            instructions="\n\n".join(static) if static else None,
            output_type=getattr(target, "output_type", str),
            tools=safe_tools,
            name=getattr(target, "name", None),
            model_settings=getattr(target, "model_settings", None),
        )
        return PreparedTarget(
            framework=FRAMEWORK_NAME,
            runtime_agent=runtime_agent,
            spec=spec,
            tool_names=tuple(sorted(tool_risks)),
            gateway=gateway,
            world_state=world_state,
            event_sink=event_sink,
            metadata={
                "capture_holder": capture_holder,
                "tool_risks": tool_risks,
                "inner_model": model,
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
        scripted = tuple(followup_turns)
        for turn in scripted:
            if not isinstance(turn, ConversationTurn):
                raise TypeError("follow-up input must contain ConversationTurn values")
            if turn.role != ConversationRole.USER:
                raise ValueError(
                    f"a scripted follow-up must be a user turn, not {turn.role.value!r}"
                )
        prepared.metadata["consumed"] = True

        started_at = utc_now()
        capture = _Capture(run_id=run_id, sink=prepared.event_sink)
        holder = cast(dict[str, "_Capture | None"], prepared.metadata["capture_holder"])
        holder["capture"] = capture

        prompt, seeded = _runner_input(input_text)
        for role, content, turn_id, metadata in seeded:
            await capture.event(
                (
                    CanonicalEventType.USER_TURN
                    if role == ConversationRole.USER
                    else CanonicalEventType.ASSISTANT_OUTPUT
                ),
                {"text": content, "turn_id": turn_id},
                metadata={**metadata, "scenario_input": True},
            )

        budgets = getattr(prepared.gateway, "budgets", None)
        agent = prepared.runtime_agent
        schema_by_tool = {
            item.value.name: dict(item.value.input_schema)
            for item in prepared.spec.tools.items
        }
        agent.model = _CapturingModel(
            prepared.metadata.get("inner_model"),
            capture,
            budgets,
            gateway=prepared.gateway,
            schema_by_tool=schema_by_tool,
        )

        termination = RunTermination.COMPLETED
        termination_reason: str | None = None
        final_output: str | None = None
        stages_executed = 0
        delivered = 0
        history: list[Any] = []
        result: Any = None
        completed = False
        try:
            while True:
                message_start = len(history)
                stages_executed += 1
                result = await agent.run(prompt, message_history=list(history) or None)
                history = list(result.all_messages())
                blocked = await _record_blocked_tool_calls(
                    capture, prepared, history, message_start
                )
                del blocked
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
                # message_history is PydanticAI's own continuation mechanism, so
                # the simulated tool calls and their results travel as history
                # and the gateway is never consulted for them again.
                prompt = turn.content
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

            if completed and result is not None:
                value = result.output
                if value is not None:
                    if isinstance(value, str):
                        final_output = value[:100_000]
                    else:
                        final_output = json.dumps(
                            _json_value(value),
                            ensure_ascii=False,
                            allow_nan=False,
                            sort_keys=True,
                        )[:100_000]
                await capture.event(
                    CanonicalEventType.FINAL_OUTPUT, {"text": final_output}
                )
        except BaseException as exc:  # noqa: BLE001 - mapped below, never swallowed
            mapped, termination_reason = _map_exception(exc)
            if mapped is None:
                holder["capture"] = None
                raise
            termination = mapped
            await capture.event(
                CanonicalEventType.ERROR,
                {"error_type": type(exc).__name__, "message": termination_reason},
            )
        finally:
            holder["capture"] = None

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
            initial_world_state={},
            final_world_state={},
            final_output=final_output,
            usage=_usage_metrics(capture.usage),
            latency_ms=max(0.0, (ended_at - started_at).total_seconds() * 1_000),
            provider_request_ids=tuple(capture.request_ids),
            metadata={
                "framework": FRAMEWORK_NAME,
                "framework_version": _sdk_version(),
                "usage_unknown": not capture.usage,
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


def _runner_input(
    input_value: str | Sequence[ConversationTurn],
) -> tuple[str, tuple[tuple[ConversationRole, str, str | None, dict[str, Any]], ...]]:
    """Seed the opening turns and return the prompt the first stage runs on.

    PydanticAI's ``run`` takes one user prompt plus history, so a seeded
    assistant turn is recorded as an event but cannot be replayed as framework
    history without fabricating a model response. Multi-turn openings are
    therefore rejected rather than approximated.
    """

    if isinstance(input_value, str):
        return input_value, ((ConversationRole.USER, input_value, None, {}),)
    turns = tuple(input_value)
    if not turns:
        raise ValueError("an agent run requires at least one conversation turn")
    if len(turns) != 1 or turns[0].role != ConversationRole.USER:
        raise ValueError(
            "this adapter seeds exactly one opening user turn; use followup_turns "
            "for later user turns"
        )
    turn = turns[0]
    return turn.content, (
        (turn.role, turn.content, turn.turn_id, dict(turn.metadata)),
    )


def _model_turns_used(capture: _Capture) -> int:
    return sum(
        1
        for event in capture.events
        if event.event_type == CanonicalEventType.MODEL_REQUEST
    )


def _map_exception(exc: BaseException) -> tuple[RunTermination | None, str]:
    """Map a framework failure onto a termination, or re-raise if unrecognised."""

    resource = getattr(exc, "resource", None)
    if resource is not None:
        return (
            {
                "model_turns": RunTermination.MAX_MODEL_TURNS,
                "tool_calls": RunTermination.MAX_TOOL_CALLS,
                "tokens": RunTermination.TOKEN_BUDGET,
                "cost_usd": RunTermination.COST_BUDGET,
                "wall_time": RunTermination.WALL_CLOCK_TIMEOUT,
            }.get(str(resource), RunTermination.ADAPTER_ERROR),
            str(exc)[:4_000],
        )
    name = type(exc).__name__
    if name == "CancelledError":
        return RunTermination.CANCELLED, "Agent run was cancelled."
    if name == "UsageLimitExceeded":
        return RunTermination.MAX_MODEL_TURNS, str(exc)[:4_000]
    if name in {"UnexpectedModelBehavior", "ModelHTTPError", "ModelRetry"}:
        return RunTermination.PROVIDER_ERROR, str(exc)[:4_000]
    if isinstance(exc, Exception):
        return RunTermination.ADAPTER_ERROR, str(exc)[:4_000]
    return None, ""


def _agentcheck_version() -> str:
    from agentcheck import __version__

    return __version__


__all__ = ["FRAMEWORK_NAME", "SUPPORTED_SDK_MINOR", "PydanticAIAdapter"]
