"""Fail-closed OpenAI Agents SDK adapter for the Phase 1 local MVP.

Only ordinary ``FunctionTool`` instances are accepted.  Each accepted tool is
reconstructed with an AgentCheck-owned invoker; the original callable and every
advanced callback are deliberately left behind on the source agent.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import inspect
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from importlib import metadata as importlib_metadata
from typing import Any, Mapping, Sequence, cast

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
    ToolsSpec,
    UnknownProperty,
    UsageMetrics,
    WorkflowsSpec,
    canonical_hash,
    utc_now,
)
from agentcheck.inspect.capabilities import classify_tool, extract_capabilities
from agentcheck.schema_safety import UnsafeSchemaReference, offline_validator
from agentcheck.runner.budgets import BudgetExceeded

from .base import (
    AdapterDependencyError,
    EventSinkProtocol,
    FrameworkAdapter,
    PreflightReport,
    PreparedTarget,
    SupportIssue,
    ToolGatewayProtocol,
)

_SDK_IMPORT_ERROR: ImportError | None = None
try:
    from agents import Agent, FunctionTool, Runner
    from agents.agent_output import AgentOutputSchemaBase
    from agents.exceptions import AgentsException, MaxTurnsExceeded, ModelBehaviorError
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
            "OpenAI Agents SDK support requires the `agentcheck` extra "
            "(`pip install 'agentlens[agentcheck]'`)."
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
    output_type = agent.output_type
    if output_type is None:
        return None, None
    try:
        if isinstance(output_type, AgentOutputSchemaBase):
            schema = output_type.json_schema()
        else:
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


def _tool_definition(tool: Any) -> ToolDefinition:
    _, state_changing, destructive = classify_tool(tool.name, tool.description or None)
    return ToolDefinition(
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
    request_ids: list[str] = dataclasses.field(default_factory=list)
    transition_links: dict[str, tuple[str, str]] = dataclasses.field(
        default_factory=dict
    )

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
        return attempt

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
    def __init__(self, capture: _Capture, budget_tracker: Any = None):
        self.capture = capture
        self.budget_tracker = budget_tracker

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
        attempt = await capture.tool_attempt(
            tool_name=tool_name,
            arguments=arguments,
            raw_arguments=raw_arguments,
            call_id=call_id,
            validation_errors=validation_errors,
            state_changing=state_changing,
            destructive=destructive,
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
            gateway_result = await _invoke_gateway(
                gateway, tool_name, arguments, world_state
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


async def _record_unknown_tool_calls(
    capture: _Capture,
    supported_tool_names: Sequence[str],
    gateway: ToolGatewayProtocol,
) -> int:
    """Preserve model-owned unknown calls as deterministic agent evidence."""

    supported = set(supported_tool_names)
    recorded = 0
    for response in capture.responses:
        for output in getattr(response, "output", ()):
            if not isinstance(output, ResponseFunctionToolCall):
                continue
            tool_name = output.name
            if tool_name in supported:
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
                validation_errors=("tool is not declared by the inspected agent",),
                state_changing=False,
                destructive=False,
            )
            budget_error = _consume_adapter_tool_budget(gateway)
            error = budget_error or ToolError(
                code="unknown_tool",
                message=f"Tool {tool_name!r} is not declared and was blocked.",
                retryable=False,
                details={"supported_tools": sorted(supported)},
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


class OpenAIAgentsAdapter(FrameworkAdapter):
    """Adapter for one trusted local OpenAI Agents SDK ``Agent``."""

    framework = FRAMEWORK_NAME

    def inspect(self, target: Any, *, source: str | None = None) -> AgentSpec:
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

        tool_properties: list[AgentProperty[Any]] = []
        definitions: list[ToolDefinition] = []
        fingerprint_tools: list[Any] = []
        for index, tool in enumerate(target.tools):
            locator = f"{source}.agent.tools[{index}]"
            if isinstance(tool, FunctionTool):
                definition = _tool_definition(tool)
                definitions.append(definition)
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

        fingerprint = {
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
        spec_id = f"agentspec-{canonical_hash(fingerprint).split(':', 1)[1][:24]}"
        runtime_locator = f"{source}.agent.runtime"

        return AgentSpec(
            spec_id=spec_id,
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

        if not isinstance(target.instructions, str) and target.instructions is not None:
            issues.append(
                SupportIssue(
                    code="dynamic_instructions",
                    message="Dynamic instruction callbacks are unsupported in Phase 1.",
                    location="agent.instructions",
                )
            )
        if target.prompt is not None:
            issues.append(
                SupportIssue(
                    code="stored_prompt",
                    message="Stored prompts can add opaque runtime behavior and are unsupported.",
                    location="agent.prompt",
                )
            )
        if target.output_type is not None:
            issues.append(
                SupportIssue(
                    code="structured_output",
                    message="Phase 1 supports text output only; structured output types are unsupported.",
                    location="agent.output_type",
                )
            )
        if target.handoffs:
            issues.append(
                SupportIssue(
                    code="handoffs",
                    message="Phase 1 supports one interactive agent and no handoffs.",
                    location="agent.handoffs",
                )
            )
        if target.mcp_servers:
            issues.append(
                SupportIssue(
                    code="mcp_servers",
                    message="MCP tools cannot be intercepted by the Phase 1 gateway.",
                    location="agent.mcp_servers",
                )
            )
        if target.hooks is not None:
            issues.append(
                SupportIssue(
                    code="agent_hooks",
                    message="Agent-owned lifecycle hooks are unsupported in isolated evaluation runs.",
                    location="agent.hooks",
                )
            )
        if target.input_guardrails or target.output_guardrails:
            issues.append(
                SupportIssue(
                    code="agent_guardrails",
                    message="Executable agent guardrails are inspected but not run in Phase 1.",
                    location="agent.guardrails",
                )
            )
        if target.tool_use_behavior != "run_llm_again":
            issues.append(
                SupportIssue(
                    code="tool_use_behavior",
                    message="Only the default run_llm_again tool behavior is supported.",
                    location="agent.tool_use_behavior",
                )
            )

        extras = {
            "extra_query": target.model_settings.extra_query,
            "extra_body": target.model_settings.extra_body,
            "extra_headers": target.model_settings.extra_headers,
            "extra_args": target.model_settings.extra_args,
        }
        for name, value in extras.items():
            if value is not None:
                issues.append(
                    SupportIssue(
                        code="opaque_model_settings",
                        message=f"Model setting {name} is opaque and unsupported in Phase 1.",
                        location=f"agent.model_settings.{name}",
                    )
                )

        seen_names: set[str] = set()
        for index, tool in enumerate(target.tools):
            location = f"agent.tools[{index}]"
            if type(tool) is not FunctionTool:
                issues.append(
                    SupportIssue(
                        code="unsupported_tool_type",
                        message=(
                            "Only exact FunctionTool instances can be replaced safely; "
                            f"found {type(tool).__name__}."
                        ),
                        location=location,
                    )
                )
                continue
            if tool.name in seen_names:
                issues.append(
                    SupportIssue(
                        code="duplicate_tool_name",
                        message=f"Duplicate tool name {tool.name!r} is ambiguous.",
                        location=location,
                    )
                )
            seen_names.add(tool.name)
            if not isinstance(tool.is_enabled, bool):
                issues.append(
                    SupportIssue(
                        code="dynamic_tool_enablement",
                        message=f"Tool {tool.name!r} has an executable is_enabled callback.",
                        location=f"{location}.is_enabled",
                    )
                )
            if tool.tool_input_guardrails or tool.tool_output_guardrails:
                issues.append(
                    SupportIssue(
                        code="tool_guardrails",
                        message=f"Tool {tool.name!r} has executable tool guardrails.",
                        location=f"{location}.guardrails",
                    )
                )
            if tool.needs_approval is not False:
                issues.append(
                    SupportIssue(
                        code="tool_approval_callback",
                        message=f"Tool {tool.name!r} has approval behavior unsupported by Phase 1.",
                        location=f"{location}.needs_approval",
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
                        location=f"{location}.timeout",
                    )
                )
            if tool.defer_loading or tool.allowed_callers is not None:
                issues.append(
                    SupportIssue(
                        code="advanced_tool_dispatch",
                        message=f"Tool {tool.name!r} uses deferred or programmatic dispatch.",
                        location=location,
                    )
                )
            if tool.custom_data_extractor is not None:
                issues.append(
                    SupportIssue(
                        code="tool_custom_data_callback",
                        message=f"Tool {tool.name!r} has a custom data callback.",
                        location=f"{location}.custom_data_extractor",
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
                        location=location,
                    )
                )
            try:
                offline_validator(tool.params_json_schema)
            except (SchemaError, UnsafeSchemaReference) as exc:
                issues.append(
                    SupportIssue(
                        code="invalid_tool_schema",
                        message=f"Tool {tool.name!r} has invalid or unsafe JSON Schema: {exc}",
                        location=f"{location}.params_json_schema",
                    )
                )

        tool_choice = target.model_settings.tool_choice
        allowed_choices = {None, "auto", "required", "none", *seen_names}
        if (
            not isinstance(tool_choice, str | type(None))
            or tool_choice not in allowed_choices
        ):
            issues.append(
                SupportIssue(
                    code="unsupported_tool_choice",
                    message="Model tool_choice targets an unsupported tool surface.",
                    location="agent.model_settings.tool_choice",
                )
            )
        return PreflightReport(framework=FRAMEWORK_NAME, issues=tuple(issues))

    def prepare(
        self,
        target: Any,
        gateway: ToolGatewayProtocol,
        *,
        world_state: Any = None,
        event_sink: EventSinkProtocol | None = None,
        source: str | None = None,
    ) -> PreparedTarget:
        report = self.preflight(target)
        report.require_supported()
        spec = self.inspect(target, source=source)
        capture_holder: dict[str, _Capture | None] = {"capture": None}
        safe_tools: list[Any] = []
        tool_risks: dict[str, tuple[bool, bool]] = {}

        for original in target.tools:
            definition = _tool_definition(original)
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

        safe_agent = target.clone(
            tools=safe_tools,
            mcp_servers=[],
            handoffs=[],
            prompt=None,
            hooks=None,
            input_guardrails=[],
            output_guardrails=[],
            model_settings=_sanitized_model_settings(target.model_settings),
        )
        return PreparedTarget(
            framework=FRAMEWORK_NAME,
            runtime_agent=safe_agent,
            spec=spec,
            tool_names=tuple(tool.name for tool in safe_tools),
            gateway=gateway,
            world_state=world_state,
            event_sink=event_sink,
            metadata={
                "capture_holder": capture_holder,
                "tool_risks": tool_risks,
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
        try:
            result = await Runner.run(
                prepared.runtime_agent,
                runner_input,
                max_turns=max_turns,
                hooks=_CapturingHooks(
                    capture, getattr(prepared.gateway, "budgets", None)
                ),
                run_config=RunConfig(
                    tracing_disabled=True,
                    trace_include_sensitive_data=False,
                    tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1),
                    tool_not_found_behavior="raise_error",
                    tool_name_collision_policy="error",
                ),
            )
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
                capture, prepared.tool_names, prepared.gateway
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
            termination_reason = str(exc)[:4_000]
            await capture.event(
                CanonicalEventType.ERROR,
                {"error_type": type(exc).__name__, "message": termination_reason},
            )
        except Exception as exc:
            termination = RunTermination.ADAPTER_ERROR
            termination_reason = str(exc)[:4_000]
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
            },
        )


__all__ = ["FRAMEWORK_NAME", "OpenAIAgentsAdapter"]
