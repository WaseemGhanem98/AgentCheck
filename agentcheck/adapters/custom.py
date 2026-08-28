"""Adapter for an agent AgentCheck was handed as declarations, not as a framework object.

The other two adapters earn their safety guarantee by *rebuilding* the target:
an ``agents.Agent`` and a ``pydantic_ai.Agent`` both carry their tools as
objects with a declared schema beside a separate callable, so a replacement
tool can be built from the schema alone and the original function is never
referenced. An agent written directly against a model API has no such surface
to rebuild from.

A custom agent gets the same guarantee from the other direction: from what
AgentCheck is *given*. It declares its tools as ``ToolDefinition`` -- name,
schema and risk flags, with no callable anywhere -- and reaches them only
through the ``ToolRuntime`` this module supplies. There is no rebuild step
because there is nothing to strip: AgentCheck never receives ``cancel_order``,
so it cannot call it. See ``agentcheck/custom.py`` for the contracts.

What this adapter therefore does is narrower than the other two. It does not
replace a model, because a custom agent's model calls happen inside its own
loop and AgentCheck never sees them. Two consequences follow, and both are
stated rather than papered over:

* ``controlled_model`` has nothing to substitute. A real provider call from
  inside orchestration is contained by the worker's deny-by-default network
  guard, not by a substituted model.
* No ``model_request``/``model_response`` events are emitted, because none were
  observed. Fabricating one per agent turn would make the evaluator's
  ``MAX_MODEL_TURNS`` oracle *look* satisfied on evidence AgentCheck does not
  have. The turn budget is instead enforced here, against agent turns, and
  refuses to deliver a turn it cannot pay for.

The side-effect boundary is likewise exact. Every declared tool call routes
through ``ToolGateway``, so it is schema-validated, fixture-matched and refused
when the tool was never declared. A side effect written directly into the
orchestration loop -- ``os.remove``, a subprocess, a local database write --
is the agent, so it runs. Process isolation, an empty environment allowlist and
socket-level network denial catch the common escapes; a local filesystem write
inside reasoning code is not preventable, and this docstring says so rather
than implying a sandbox.

Everything outside this module stays framework-neutral: the gateway, worker,
budgets, oracles and artifacts are the ones the first two adapters already use.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect as _inspect
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentcheck.config import ToolRiskDeclaration
    from agentcheck.mcp_manifest import McpManifest

from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]

from agentcheck.custom import TurnResult
from agentcheck.domain.agent_spec import (
    AgentProperty,
    AgentSpec,
    CapabilitiesSpec,
    IdentitySpec,
    InspectionProvenance,
    InstructionsSpec,
    InterfaceSpec,
    ObservabilitySpec,
    RiskAuthority,
    RiskAxis,
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
from agentcheck.inspect.risk_authority import declared_risk_for
from agentcheck.runner.tool_gateway import ToolCallBlockedError
from agentcheck.schema_safety import UnsafeSchemaReference, offline_validator

from .base import (
    portable_identity,
    require_known_tool_risk_names,
    AdapterRuntimeError,
    EventSinkProtocol,
    FrameworkAdapter,
    PreflightReport,
    PreparedTarget,
    SupportIssue,
    ToolGatewayProtocol,
    UnsupportedTargetError,
)

FRAMEWORK_NAME = "custom"

# The declared surface a custom agent must expose. Kept as data so preflight and
# the tests that pin fail-closed behaviour name the same thing.
_REQUIRED_TURN_METHODS = ("start", "resume")

# Non-self parameters each turn method must accept, in order. Names are not
# enforced -- positional dispatch is what the adapter actually relies on -- but
# arity is, because a mismatch is a TypeError raised in the middle of a run
# rather than a refusal before one.
_START_ARITY = 2
_RESUME_ARITY = 3


# ---------------------------------------------------------------------------
# Spec construction helpers
#
# Deliberately duplicated from the other two adapters rather than hoisted into a
# shared module. Unifying them means editing two shipped adapters to serve a
# third, and the shapes have not yet been shown to be the same shape; a fourth
# adapter, or a real divergence, is the evidence that would justify it.
# ---------------------------------------------------------------------------


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
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _json_object(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(item) for key, item in value.items()}


def _text_output(value: Any) -> str | None:
    """Render one turn's output as the canonical run's text, or ``None``.

    ``TurnResult.output`` is deliberately ``Any``: a custom agent may answer
    with a string or with a structured object. Neither is coerced into the
    other -- a string is taken as written, anything else is canonically
    JSON-encoded so two equal outputs render identically.
    """

    if value is None:
        return None
    if isinstance(value, str):
        return value[:100_000]
    return json.dumps(
        _json_value(value), ensure_ascii=False, allow_nan=False, sort_keys=True
    )[:100_000]


# ---------------------------------------------------------------------------
# Declared surface, read without executing anything
# ---------------------------------------------------------------------------


def _raw_tools(target: Any) -> Any:
    return getattr(target, "tools", None)


def _declared_tools(target: Any) -> tuple[ToolDefinition, ...]:
    """The declared tools, or ``()`` when the declaration is unusable.

    Never raises: ``preflight`` is the single place that turns a bad declaration
    into an actionable refusal, and ``inspect`` must be able to run far enough
    to describe what it found.
    """

    raw = _raw_tools(target)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(item for item in raw if isinstance(item, ToolDefinition))


def _spec_tool(
    definition: ToolDefinition,
    *,
    declared_tool_risk: "Mapping[str, ToolRiskDeclaration] | None" = None,
) -> tuple[ToolDefinition, ToolRiskAssertion]:
    """The inspected form of one declared tool, plus its risk provenance.

    ``replaceable`` is set here rather than asked of the author. The flag means
    "AgentCheck can substitute this tool safely", which for the other adapters
    is a claim about being able to build a replacement for a live callable. A
    custom declaration is already inert data -- there is no callable to replace
    -- so the claim holds structurally, and requiring the author to assert it
    would be a tax on a field whose meaning is internal to the adapters.

    A custom agent's ``state_changing``/``destructive`` are already a first-hand
    developer declaration written directly on the ``ToolDefinition`` -- there is
    no name-based inference for this adapter, unlike the SDK adapters, because
    there is nothing to infer from that the author did not already state
    outright. ``declared_tool_risk`` (the ``tool_risk`` block in
    ``agentcheck.json``) is an *additional*, independent developer surface that
    can override either axis without editing the agent's source; a conflict
    between the two is recorded rather than silently resolved one way.
    """

    override = declared_risk_for(definition.name, declared_tool_risk)
    state_changing = definition.state_changing
    destructive = definition.destructive
    conflicts: list[str] = []
    if override is not None:
        if override.state_changing is not None:
            if override.state_changing != definition.state_changing:
                conflicts.append(
                    f"{definition.name}.state_changing: agentcheck.json declares "
                    f"{override.state_changing}, overriding the agent's own "
                    f"declaration of {definition.state_changing}"
                )
            state_changing = override.state_changing
        if override.destructive is not None:
            if override.destructive != definition.destructive:
                conflicts.append(
                    f"{definition.name}.destructive: agentcheck.json declares "
                    f"{override.destructive}, overriding the agent's own "
                    f"declaration of {definition.destructive}"
                )
            destructive = override.destructive

    locator = f"tool:{definition.name}"
    assertion = ToolRiskAssertion(
        tool_name=definition.name,
        state_changing=RiskAxis(
            value=state_changing, authority=RiskAuthority.DEVELOPER_DECLARED, confidence=1.0
        ),
        destructive=RiskAxis(
            value=destructive, authority=RiskAuthority.DEVELOPER_DECLARED, confidence=1.0
        ),
        evidence=(
            SpecEvidence(
                evidence_id=f"tool-risk:{definition.name}:declared",
                summary=(
                    f"state_changing={state_changing}, destructive={destructive}, "
                    "declared directly on the agent's own ToolDefinition"
                    + (" and overridden by agentcheck.json" if override is not None else "")
                    + "."
                ),
                locator=locator,
            ),
        ),
        conflicts=tuple(conflicts),
    )
    updated = definition.model_copy(
        update={
            "replaceable": True,
            "state_changing": state_changing,
            "destructive": destructive,
        }
    )
    return updated, assertion


def _turn_method(target: Any, name: str) -> Any:
    return getattr(target, name, None)


def _non_self_arity(method: Any) -> int | None:
    """Positional parameters a bound turn method accepts, or ``None`` if unknown.

    ``*args`` reports as unknown rather than as a match: a signature AgentCheck
    cannot read is not a signature it should assume is compatible.
    """

    try:
        signature = _inspect.signature(method)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return None
    count = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (
            _inspect.Parameter.VAR_POSITIONAL,
            _inspect.Parameter.VAR_KEYWORD,
        ):
            return None
        if parameter.kind is _inspect.Parameter.KEYWORD_ONLY:
            if parameter.default is _inspect.Parameter.empty:
                return None
            continue
        count += 1
    return count


# ---------------------------------------------------------------------------
# Canonical capture
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Capture:
    """Builds canonical domain objects while a custom agent turn is in progress.

    Synchronous, unlike its counterparts in the other two adapters, because
    ``ToolRuntime.call`` is synchronous by contract and a custom loop should not
    have to become async to be testable. Events are therefore buffered here and
    flushed to an optional live sink by ``run`` between turns; the sink still
    sees every event, in order, just not mid-turn.
    """

    run_id: str
    events: list[CanonicalEvent] = dataclasses.field(default_factory=list)
    transition_links: dict[str, tuple[str, str]] = dataclasses.field(
        default_factory=dict
    )
    attempts: list[ToolAttempt] = dataclasses.field(default_factory=list)
    outcomes: list[ToolOutcome] = dataclasses.field(default_factory=list)
    flushed: int = 0

    def event(
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
        return event

    def tool_attempt(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        raw_arguments: str,
        call_id: str,
        state_changing: bool,
        destructive: bool,
    ) -> ToolAttempt:
        # The causal ancestor of a custom tool call is the user turn that is
        # being answered: there is no observed model response to point at, and
        # inventing one would put a fabricated link in the evidence graph.
        source_event_ids: tuple[str, ...] = ()
        for source_event in reversed(self.events):
            if source_event.event_type == CanonicalEventType.USER_TURN:
                source_event_ids = (source_event.event_id,)
                break
        index = len(self.attempts)
        event = self.event(
            CanonicalEventType.TOOL_ATTEMPT,
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "call_id": call_id,
                "raw_arguments_sha256": hashlib.sha256(
                    raw_arguments.encode("utf-8")
                ).hexdigest(),
                "validation_errors": [],
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

    def tool_result(
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
        ended_at = utc_now()
        index = len(self.outcomes)
        event = self.event(
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
            state_transition_ids=tuple(state_transition_ids),
            metadata=_json_object(dict(gateway_metadata or {}))
            | ({"gateway_outcome_id": gateway_outcome_id} if gateway_outcome_id else {}),
        )
        self.outcomes.append(outcome)
        return outcome

    def link_transitions(
        self, transition_ids: Sequence[str], attempt: ToolAttempt
    ) -> tuple[str, ...]:
        canonical_ids: list[str] = []
        for gateway_id in transition_ids:
            link = self.transition_links.get(gateway_id)
            if link is None:
                link = (
                    f"{self.run_id}:transition:{len(self.transition_links):04d}",
                    attempt.attempt_id,
                )
                self.transition_links[gateway_id] = link
            canonical_ids.append(link[0])
        return tuple(canonical_ids)


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


# ---------------------------------------------------------------------------
# The ToolRuntime -> ToolGateway bridge
# ---------------------------------------------------------------------------


class _BaseGatewayToolRuntime:
    """Shared plumbing for the sync and async ``ToolRuntime`` implementations.

    Holds ``__init__`` and the actual gateway-invoking logic (``_do_call``).
    Neither leaf subclass overrides the other's ``call``, so mypy sees two
    unrelated, individually consistent method signatures rather than a sync
    override of an async one or vice versa.
    """

    def __init__(
        self,
        *,
        gateway: ToolGatewayProtocol,
        world_state: Any,
        capture_holder: dict[str, "_Capture | None"],
        tool_risks: Mapping[str, tuple[bool, bool]],
    ) -> None:
        self._gateway = gateway
        self._world_state = world_state
        self._capture_holder = capture_holder
        self._tool_risks = dict(tool_risks)

    def _do_call(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome:
        capture = self._capture_holder.get("capture")
        if capture is None:
            raise RuntimeError(
                "prepared AgentCheck target was invoked outside adapter.run()"
            )
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        if not isinstance(arguments, Mapping):
            raise TypeError("tool arguments must be a mapping")
        safe_arguments = {str(key): value for key, value in arguments.items()}
        raw_arguments = json.dumps(
            _json_value(safe_arguments), ensure_ascii=False, sort_keys=True
        )
        state_changing, destructive = self._tool_risks.get(name, (False, False))
        attempt = capture.tool_attempt(
            tool_name=name,
            arguments=safe_arguments,
            raw_arguments=raw_arguments,
            call_id=f"agentcheck-{name}-{len(capture.attempts) + 1}",
            state_changing=state_changing,
            destructive=destructive,
        )
        started_at = utc_now()
        try:
            raw = self._gateway.invoke(name, safe_arguments, self._world_state)
        except BaseException as exc:
            controlled = getattr(exc, "outcome", None)
            if isinstance(controlled, ToolOutcome):
                # A gateway refusal. Record it as canonical evidence, then let
                # it keep travelling: swallowing it here would hand the agent a
                # turn in which a blocked call looks like it never happened.
                self._record(capture, attempt, controlled, started_at)
                raise
            capture.event(
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
        if _inspect.isawaitable(raw):
            # ToolGatewayProtocol permits an awaitable; ToolRuntime.call is
            # synchronous by contract and cannot await one without owning an
            # event loop the custom agent is already running inside.
            close = getattr(raw, "close", None)
            if callable(close):
                close()  # no unawaited-coroutine warning on the way out
            raise AdapterRuntimeError(
                "the custom adapter requires a synchronous tool gateway"
            )
        outcome = self._record(capture, attempt, raw, started_at)
        if outcome.status is ToolOutcomeStatus.BLOCKED:
            # The gateway returns rather than raises when arguments fail the
            # declared schema. The ToolRuntime contract is the stricter of the
            # two: a refusal is never handed back as if it were a result.
            raise ToolCallBlockedError(
                (
                    f"tool {name!r} was refused by AgentCheck: "
                    f"{outcome.error.code if outcome.error else 'blocked'}"
                ),
                outcome,
            )
        return outcome

    def _record(
        self,
        capture: _Capture,
        attempt: ToolAttempt,
        value: Any,
        started_at: Any,
    ) -> ToolOutcome:
        (
            status,
            result,
            error,
            outcome_id,
            transition_ids,
            metadata,
        ) = _gateway_result_parts(value)
        canonical_transition_ids = capture.link_transitions(transition_ids, attempt)
        outcome = capture.tool_result(
            attempt=attempt,
            status=status,
            result=result,
            error=error,
            started_at=started_at,
            gateway_outcome_id=outcome_id,
            state_transition_ids=canonical_transition_ids,
            gateway_metadata=metadata,
        )
        return outcome


class _GatewayToolRuntime(_BaseGatewayToolRuntime):
    """The ``ToolRuntime`` a custom agent is handed. A bridge, not a second gateway.

    Everything that decides what a tool call *means* -- schema validation,
    fixture selection, invocation-index and prerequisite ordering, world-state
    effects, the unknown-tool allowlist, the tool-call and retry budgets -- lives
    in ``ToolGateway`` and is reached by exactly one call to ``invoke`` below.
    This class adds two things and nothing else: canonical evidence for the
    attempt and its outcome, and the raise/return split the ``ToolRuntime``
    contract promises.

    That split is the whole of the policy here:

    * A *simulated* outcome is returned, whatever its status. ``error``,
      ``timeout``, ``malformed``, ``empty``, ``partial`` and ``stale`` are things
      a real tool does, so a custom loop must be able to see them and decide.
    * A *refusal* is raised. An undeclared tool, arguments that violate the
      declared schema, a missing fixture, an exhausted budget: in each case
      AgentCheck has no controlled answer, and returning a plausible one would
      be inventing the tool output that the run is supposed to be evidence
      about.

    ``call`` is synchronous and, unlike the OpenAI Agents and PydanticAI
    adapters, this one has no hook that sees a batch of tool calls before
    dispatch: a custom agent's own orchestration decides, in its own code,
    when and how to call ``call``. If that orchestration issues concurrent
    calls itself -- multiple threads, or tasks that actually interleave
    around a real ``await`` rather than calling this synchronously one at a
    time -- fixture assignment and invocation-index ordering are not
    guaranteed to follow any particular order. This is deliberately left
    unsupported rather than guessed at: a custom agent that wants
    deterministic concurrent dispatch must issue its own calls to ``call``
    strictly in the order they were decided, matching what this class was
    proven safe for. See ``docs/concurrent-tool-decisions.md``.

    An agent whose ``start``/``resume`` are coroutine functions is instead
    handed ``_AsyncGatewayToolRuntime`` (below), which has the same real-thread
    caveat but a different, provable guarantee for asyncio-native scheduling
    -- see that class's docstring.
    """

    def call(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome:
        return self._do_call(name, arguments)


class _AsyncGatewayToolRuntime(_BaseGatewayToolRuntime):
    """The ``AsyncToolRuntime`` given to an async ``start``/``resume`` pair.

    ``call`` here has no ``await`` in its own body -- it returns exactly what
    the synchronous ``_do_call`` (inherited unchanged) computes. That is the
    whole of the safety argument: a coroutine with no internal yield point
    cannot be paused mid-call by the event loop, so however the agent chooses
    to schedule several of these (sequential ``await``, ``asyncio.gather``,
    ``asyncio.create_task``) each call still runs to completion, in the order
    it was scheduled, before the next one's body executes. See
    ``agentcheck.custom.AsyncToolRuntime`` for what this does and does not
    claim about concurrency.
    """

    async def call(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome:
        return self._do_call(name, arguments)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class CustomAgentAdapter(FrameworkAdapter):
    """Evaluate an agent that implements ``CustomAgentProtocol``."""

    framework = FRAMEWORK_NAME

    # -- inspection ---------------------------------------------------------

    def inspect(
        self,
        target: Any,
        *,
        source: str | None = None,
        identity_locator: str | None = None,
        declared_tool_risk: "Mapping[str, ToolRiskDeclaration] | None" = None,
        mcp_manifest: "McpManifest | None" = None,
    ) -> AgentSpec:
        """Describe the declared surface. Nothing on the target is executed.

        The declaration *is* the specification here, which is what makes this
        adapter's inspect the shortest of the three: there are no private
        framework attributes to read and no callables to classify. Reading
        ``tools`` and, optionally, a static ``instructions`` string is the whole
        of the target access.
        """

        del mcp_manifest  # No external-toolset concept on this adapter yet.
        locator = source or f"{type(target).__module__}.{type(target).__name__}"
        tool_risk_assertions: dict[str, ToolRiskAssertion] = {}
        definitions: list[ToolDefinition] = []
        for item in _declared_tools(target):
            definition, assertion = _spec_tool(item, declared_tool_risk=declared_tool_risk)
            definitions.append(definition)
            tool_risk_assertions[definition.name] = assertion
        raw_instructions = getattr(target, "instructions", None)
        instructions = (
            raw_instructions
            if isinstance(raw_instructions, str) and raw_instructions.strip()
            else None
        )

        tool_properties: list[AgentProperty[Any]] = []
        fingerprint_tools: list[Any] = []
        for index, definition in enumerate(definitions):
            fingerprint_tools.append(definition.model_dump(mode="json"))
            tool_properties.append(
                _property(
                    definition,
                    locator=f"{locator}.tools[{index}]",
                    summary=(
                        "Tool name, description, JSON Schema and risk flags as the "
                        "agent declared them."
                    ),
                    kind=SourceKind.TOOL_SCHEMA,
                    # Declared by the author rather than reconstructed from a
                    # framework object, so unlike the SDK adapters this is a
                    # first-hand reading, not an inference.
                    confidence=1.0,
                    inferred=False,
                    authoritative=True,
                )
            )

        capabilities = [
            _property(
                extracted.capability,
                locator=f"tool:{extracted.tool_name}",
                summary=(
                    "Argument surface read from the declared schema; action kind is "
                    "inferred, while side-effect risk comes from the declaration."
                ),
                kind=SourceKind.TOOL_SCHEMA,
                confidence=extracted.confidence,
                inferred=True,
                authoritative=False,
            )
            for extracted in extract_capabilities(definitions)
        ]

        name = getattr(target, "name", None)
        agent_name = name if isinstance(name, str) and name.strip() else type(target).__name__

        fingerprint: dict[str, Any] = {
            "source": locator,
            "name": agent_name,
            "framework_version": None,
            "model": None,
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
                    agent_name,
                    locator=f"{locator}.name",
                    summary="Agent name read from the declared object.",
                ),
                framework=_property(
                    FRAMEWORK_NAME,
                    locator=locator,
                    summary="The target declares the custom-agent contract directly.",
                    kind=SourceKind.FRAMEWORK_METADATA,
                ),
                framework_version=_unknown_property(
                    None,
                    locator=locator,
                    summary="A custom agent has no framework version to report.",
                ),
                provider=_unknown_property(
                    None,
                    locator=locator,
                    summary=(
                        "A custom agent's provider is chosen inside its own loop; "
                        "AgentCheck never sees it."
                    ),
                ),
                model=_unknown_property(
                    None,
                    locator=locator,
                    summary=(
                        "A custom agent's model is chosen inside its own loop; "
                        "AgentCheck never sees it."
                    ),
                ),
            ),
            interface=InterfaceSpec(
                entrypoint=_property(
                    locator,
                    locator=locator,
                    summary="Entrypoint AgentCheck imported.",
                ),
                input_modalities=_property(
                    ("text",),
                    locator=f"{locator}.start",
                    summary="start() accepts a text message.",
                ),
                output_modalities=_property(
                    ("text",),
                    locator=f"{locator}.start",
                    summary="TurnResult.output is rendered as text or canonical JSON.",
                ),
                input_schema=_unknown_property(
                    None,
                    locator=f"{locator}.start",
                    summary="The custom contract declares no input schema.",
                ),
                output_schema=_unknown_property(
                    None,
                    locator=f"{locator}.start",
                    summary="TurnResult.output is deliberately untyped.",
                ),
                interactive=_property(
                    True,
                    locator=f"{locator}.resume",
                    summary="resume() continues a turn from the previous state.",
                ),
            ),
            instructions=InstructionsSpec(
                system=(
                    _property(
                        instructions,
                        locator=f"{locator}.instructions",
                        summary="Static instruction text declared by the agent.",
                        kind=SourceKind.SYSTEM_INSTRUCTION,
                    )
                    if instructions is not None
                    else _unknown_property(
                        None,
                        locator=f"{locator}.instructions",
                        summary=(
                            "The agent declares no static instructions; any prompt it "
                            "uses is built inside its own loop."
                        ),
                    )
                ),
                developer=_unknown_property(
                    None,
                    locator=locator,
                    summary="The custom contract has no developer instruction channel.",
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
                    locator=locator,
                    summary="Turn limits come from the scenario, not the agent.",
                ),
                max_tool_calls=_unknown_property(
                    None,
                    locator=locator,
                    summary="Tool-call limits come from the scenario, not the agent.",
                ),
                timeout_seconds=_unknown_property(
                    None,
                    locator=locator,
                    summary="Timeouts come from the scenario, not the agent.",
                ),
                token_budget=_unknown_property(
                    None,
                    locator=locator,
                    summary="Token budgets come from the scenario, not the agent.",
                ),
                cost_budget_usd=_unknown_property(
                    None,
                    locator=locator,
                    summary="Cost budgets come from the scenario, not the agent.",
                ),
            ),
            observability=ObservabilitySpec(
                supported_event_types=_property(
                    (
                        "user_turn",
                        "assistant_output",
                        "tool_attempt",
                        "tool_result",
                        "error",
                        "final_output",
                    ),
                    locator=f"{locator}.adapter",
                    summary=(
                        "Canonical events this adapter emits. Model events are absent "
                        "because a custom agent's model calls are not observed."
                    ),
                ),
                usage_metrics=_property(
                    (),
                    locator=f"{locator}.adapter",
                    summary=(
                        "No usage is observable: token accounting happens inside the "
                        "agent's own provider calls."
                    ),
                ),
                provider_request_ids=_property(
                    False,
                    locator=f"{locator}.adapter",
                    summary="AgentCheck issues no provider request for a custom agent.",
                ),
                source_event_links=_property(
                    True,
                    locator=f"{locator}.adapter",
                    summary="Tool events link back to the user turn being answered.",
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
                ),
            ),
        )

    # -- support decision ---------------------------------------------------

    def preflight(
        self, target: Any, *, mcp_manifest: "McpManifest | None" = None
    ) -> PreflightReport:
        """Refuse a target that cannot be driven safely, before it is driven.

        Every check is structural. Nothing here calls ``start``, ``resume`` or
        any other target code to find out what it would have declared: a
        declaration that has to be executed to be read is not a declaration.
        """

        del mcp_manifest  # No external-toolset concept on this adapter yet.
        if not self._implements_any_of_the_contract(target):
            # One diagnosis, not three symptoms. An object with none of the
            # contract on it is almost always the wrong object -- a config
            # pointing at a dict, a module, or an SDK agent belonging to another
            # adapter -- and listing its three missing members invites the
            # developer to add them to the wrong thing.
            return PreflightReport(
                framework=FRAMEWORK_NAME,
                issues=(
                    SupportIssue(
                        code="not_a_custom_agent",
                        message=(
                            f"The configured entrypoint resolved to "
                            f"{type(target).__name__}, which implements none of "
                            "the custom-agent contract. AgentCheck expects an "
                            "object with a `tools` sequence of ToolDefinition "
                            "and `start(message, tools)` / `resume(state, "
                            "message, tools)` methods (agentcheck.custom). If "
                            "this target is an OpenAI Agents or PydanticAI "
                            "agent, set that adapter instead of \"custom\"."
                        ),
                        location="entrypoint",
                    ),
                ),
            )
        issues: list[SupportIssue] = []
        issues.extend(self._tool_issues(target))
        issues.extend(self._turn_method_issues(target))
        return PreflightReport(framework=FRAMEWORK_NAME, issues=tuple(issues))

    @staticmethod
    def _implements_any_of_the_contract(target: Any) -> bool:
        """Whether the object is an attempt at the contract at all.

        Deliberately generous: one recognisable member is enough to treat the
        target as a custom agent whose implementation is incomplete, and to
        report precisely what is missing from it.
        """

        if _raw_tools(target) is not None:
            return True
        return any(
            callable(_turn_method(target, name)) for name in _REQUIRED_TURN_METHODS
        )

    @staticmethod
    def _tool_issues(target: Any) -> list[SupportIssue]:
        issues: list[SupportIssue] = []
        raw = _raw_tools(target)
        if raw is None:
            return [
                SupportIssue(
                    code="missing_tools_declaration",
                    message=(
                        "A custom agent must declare its tools as a sequence of "
                        "ToolDefinition. AgentCheck will not discover them by "
                        "running the agent."
                    ),
                    location="tools",
                )
            ]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            return [
                SupportIssue(
                    code="invalid_tools_declaration",
                    message=(
                        "`tools` must be a sequence of ToolDefinition, not "
                        f"{type(raw).__name__}."
                    ),
                    location="tools",
                )
            ]

        seen: set[str] = set()
        for index, item in enumerate(raw):
            location = f"tools[{index}]"
            if not isinstance(item, ToolDefinition):
                issues.append(
                    SupportIssue(
                        code="invalid_tool_definition",
                        message=(
                            "Declared tools must be agentcheck ToolDefinition values, "
                            f"not {type(item).__name__}. AgentCheck accepts no other "
                            "tool shape, which is why it can never be handed a live "
                            "handler."
                        ),
                        location=location,
                    )
                )
                continue
            if item.name in seen:
                issues.append(
                    SupportIssue(
                        code="duplicate_tool_name",
                        message=(
                            f"Tool {item.name!r} is declared more than once, so a call "
                            "to it would be ambiguous."
                        ),
                        location=location,
                    )
                )
            seen.add(item.name)
            schema = dict(item.input_schema)
            try:
                offline_validator(schema, check_formats=True)
            except (SchemaError, UnsafeSchemaReference) as exc:
                issues.append(
                    SupportIssue(
                        code="invalid_tool_schema",
                        message=(
                            f"The schema for tool {item.name!r} is not safe Draft "
                            f"2020-12 JSON Schema: {type(exc).__name__}."
                        ),
                        location=location,
                    )
                )
        return issues

    @staticmethod
    def _turn_method_issues(target: Any) -> list[SupportIssue]:
        issues: list[SupportIssue] = []
        expected_arity = {"start": _START_ARITY, "resume": _RESUME_ARITY}
        found_methods: dict[str, Any] = {}
        for name in _REQUIRED_TURN_METHODS:
            method = _turn_method(target, name)
            if method is None or not callable(method):
                issues.append(
                    SupportIssue(
                        code=f"missing_{name}",
                        message=(
                            f"A custom agent must implement {name}(); AgentCheck "
                            "drives user turns through it."
                        ),
                        location=name,
                    )
                )
                continue
            found_methods[name] = method
            arity = _non_self_arity(method)
            if arity == expected_arity[name]:
                continue
            found = (
                "its signature could not be read, or it uses *args/**kwargs"
                if arity is None
                else f"this one accepts {arity}"
            )
            issues.append(
                SupportIssue(
                    code=f"incompatible_{name}_signature",
                    message=(
                        f"{name}() must accept exactly {expected_arity[name]} "
                        "positional arguments, as CustomAgentProtocol declares "
                        f"them: {found}."
                    ),
                    location=name,
                )
            )
        if len(found_methods) == len(_REQUIRED_TURN_METHODS):
            is_coroutine = {
                name: _inspect.iscoroutinefunction(method)
                for name, method in found_methods.items()
            }
            if len(set(is_coroutine.values())) > 1:
                described = ", ".join(
                    f"{name}() is {'async' if coro else 'sync'}"
                    for name, coro in sorted(is_coroutine.items())
                )
                issues.append(
                    SupportIssue(
                        code="mismatched_turn_method_concurrency",
                        message=(
                            "start() and resume() must both be synchronous or both be "
                            f"coroutine functions, not a mix ({described}). AgentCheck "
                            "picks one ToolRuntime shape (sync or async) for the whole "
                            "agent and cannot switch between turns."
                        ),
                        location="start/resume",
                    )
                )
        return issues

    # -- preparation --------------------------------------------------------

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
        mcp_manifest: "McpManifest | None" = None,
    ) -> PreparedTarget:
        """Bind the declared agent to a gateway, or refuse before it runs.

        There is no sanitized copy to build. The other adapters return a rebuilt
        agent because the original holds live handlers; this one returns the
        target itself, because the only tool surface AgentCheck was given is
        inert data and the only way back to a tool is the ``ToolRuntime``
        constructed here.
        """

        del mcp_manifest  # No external-toolset concept on this adapter yet.
        if controlled_model:
            # Refused, not recorded as a caveat. AgentCheck substitutes a
            # deterministic model by rebuilding the target around one, and a
            # custom agent has no such seam: its model calls happen inside the
            # loop this adapter merely starts and resumes. Accepting the request
            # and running anyway would let a caller believe an offline
            # substitution had happened while the target reached for its own
            # provider -- which the worker's denied egress would then refuse,
            # somewhere much less legible than here.
            raise UnsupportedTargetError(
                [
                    SupportIssue(
                        code="controlled_model_unsupported",
                        message=(
                            "This adapter cannot substitute a controlled offline "
                            "model: a custom agent owns its own model calls, and "
                            "AgentCheck never sees them. Remove "
                            "`controlled_model` from the configuration, or make "
                            "the agent's own loop use a deterministic model of "
                            "its choosing for evaluation runs."
                        ),
                        location="config.controlled_model",
                    )
                ]
            )
        report = self.preflight(target)
        report.require_supported()
        spec = self.inspect(
            target,
            source=source,
            identity_locator=identity_locator,
            declared_tool_risk=declared_tool_risk,
        )
        require_known_tool_risk_names(spec, declared_tool_risk)

        tool_risks: dict[str, tuple[bool, bool]] = {}
        for item in spec.tools.items:
            definition = item.value
            # `definition.state_changing`/`.destructive` are already the fully
            # resolved developer declaration -- from the agent's own
            # ToolDefinition, optionally overridden by `agentcheck.json`. Using
            # `spec.capabilities` here instead (as this code once did) reruns
            # name-based inference, which silently discarded the author's
            # explicit declaration whenever the tool's name carried no lexical
            # signal of its own.
            tool_risks[definition.name] = (
                definition.state_changing,
                definition.destructive,
            )

        self._require_matching_surface(gateway, tuple(sorted(tool_risks)))

        capture_holder: dict[str, _Capture | None] = {"capture": None}
        runtime_cls = (
            _AsyncGatewayToolRuntime
            if _inspect.iscoroutinefunction(_turn_method(target, "start"))
            else _GatewayToolRuntime
        )
        runtime = runtime_cls(
            gateway=gateway,
            world_state=world_state,
            capture_holder=capture_holder,
            tool_risks=tool_risks,
        )
        return PreparedTarget(
            framework=FRAMEWORK_NAME,
            runtime_agent=target,
            spec=spec,
            tool_names=tuple(sorted(tool_risks)),
            gateway=gateway,
            world_state=world_state,
            event_sink=event_sink,
            metadata={
                "capture_holder": capture_holder,
                "tool_risks": tool_risks,
                "tool_runtime": runtime,
                "consumed": False,
            },
        )

    @staticmethod
    def _require_matching_surface(
        gateway: ToolGatewayProtocol, declared: tuple[str, ...]
    ) -> None:
        """Refuse a gateway whose allowlist is not the declared tool surface.

        In the worker the two are built from the same inspected spec and cannot
        diverge. Any other caller can pair them by hand, and a mismatch in
        either direction is a wiring bug that would otherwise surface as a
        confusing mid-run block, or -- worse -- as a tool the agent may reach
        that its own specification never mentioned.
        """

        allowed = getattr(gateway, "allowed_tools", None)
        if allowed is None:
            return
        if isinstance(allowed, (str, bytes)) or not isinstance(allowed, Sequence):
            return
        allowed_names = {str(name) for name in allowed}
        undeclared = sorted(allowed_names - set(declared))
        unreachable = sorted(set(declared) - allowed_names)
        if not undeclared and not unreachable:
            return
        details: list[str] = []
        if unreachable:
            details.append(
                "declared but not in the gateway allowlist: " + ", ".join(unreachable)
            )
        if undeclared:
            details.append(
                "in the gateway allowlist but not declared: " + ", ".join(undeclared)
            )
        raise UnsupportedTargetError(
            [
                SupportIssue(
                    code="tool_surface_divergence",
                    message=(
                        "The declared tool surface and the gateway allowlist must be "
                        "identical (" + "; ".join(details) + ")."
                    ),
                    location="tools",
                )
            ]
        )

    # -- execution ----------------------------------------------------------

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
        """Drive the scenario's user turns and normalize what the agent did.

        AgentCheck owns the *turns*; the agent owns its loop within a turn. One
        opening ``start`` plus one ``resume`` per scripted follow-up, with the
        agent free to call as many tools as it likes inside each -- every one of
        them through the same gateway, the same fixtures and the same budget.

        The agent's ``state`` lives in this frame and nowhere else. It is never
        written to ``prepared``, to run metadata, to an artifact or to any wire
        format, so it is never serialized and pickle is not involved in the
        design at all. Isolation is the worker process, which is where the
        object already is.
        """

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
        prompt, turn_id, turn_metadata = _runner_input(input_text)
        prepared.metadata["consumed"] = True
        initial_world = _world_snapshot(prepared.world_state)

        started_at = utc_now()
        capture = _Capture(run_id=run_id)
        holder = prepared.metadata["capture_holder"]
        holder["capture"] = capture
        runtime = prepared.metadata["tool_runtime"]
        sink = prepared.event_sink
        agent = prepared.runtime_agent

        capture.event(
            CanonicalEventType.USER_TURN,
            {"text": prompt, "turn_id": turn_id},
            metadata={**turn_metadata, "scenario_input": True},
        )
        await _flush_events(capture, sink)

        termination = RunTermination.COMPLETED
        termination_reason: str | None = None
        final_output: str | None = None
        stages_executed = 0
        delivered = 0
        state: Any = None
        last_output: Any = None
        completed = False
        is_async = _inspect.iscoroutinefunction(agent.start)
        try:
            while True:
                stages_executed += 1
                if stages_executed == 1:
                    result = (
                        await agent.start(prompt, runtime)
                        if is_async
                        else agent.start(prompt, runtime)
                    )
                else:
                    result = (
                        await agent.resume(state, prompt, runtime)
                        if is_async
                        else agent.resume(state, prompt, runtime)
                    )
                turn_result = _require_turn_result(result, stage=stages_executed)
                state = turn_result.state
                last_output = turn_result.output
                capture.event(
                    CanonicalEventType.ASSISTANT_OUTPUT,
                    {"text": _text_output(turn_result.output)},
                    metadata={
                        **_json_object(dict(turn_result.metadata)),
                        "stage": stages_executed,
                    },
                )
                await _flush_events(capture, sink)

                if delivered >= len(scripted):
                    completed = True
                    break
                if max_turns - stages_executed < 1:
                    termination = RunTermination.MAX_MODEL_TURNS
                    termination_reason = (
                        f"The scenario's {max_turns}-turn budget was spent before "
                        f"scripted user turn {delivered + 1} of {len(scripted)} could "
                        "be delivered."
                    )
                    capture.event(
                        CanonicalEventType.ERROR,
                        {
                            "error_type": "BudgetExceeded",
                            "resource": "model_turns",
                            "message": termination_reason,
                        },
                    )
                    break
                turn = scripted[delivered]
                prompt = turn.content
                capture.event(
                    CanonicalEventType.USER_TURN,
                    {"text": turn.content, "turn_id": turn.turn_id},
                    metadata={
                        **dict(turn.metadata),
                        "scenario_input": True,
                        "followup_index": delivered,
                    },
                )
                delivered += 1

            if completed:
                final_output = _text_output(last_output)
                capture.event(CanonicalEventType.FINAL_OUTPUT, {"text": final_output})
        except BaseException as exc:  # noqa: BLE001 - mapped below, never swallowed
            mapped, termination_reason = _map_exception(exc)
            if mapped is None:
                holder["capture"] = None
                raise
            termination = mapped
            capture.event(
                CanonicalEventType.ERROR,
                {"error_type": type(exc).__name__, "message": termination_reason},
            )
        finally:
            # Drops the runtime's way back into this capture, and lets the
            # agent's opaque state fall out of scope with the frame.
            holder["capture"] = None
            state = None
        await _flush_events(capture, sink)

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
            final_world_state=_world_snapshot(prepared.world_state),
            final_output=final_output,
            # Every field stays None: a custom agent's tokens are spent inside
            # its own provider calls, and an unobserved metric is not zero.
            usage=UsageMetrics(),
            latency_ms=max(0.0, (ended_at - started_at).total_seconds() * 1_000),
            provider_request_ids=(),
            metadata={
                "framework": FRAMEWORK_NAME,
                "framework_version": None,
                "usage_unknown": True,
                # Read by the evaluator, which will not compare an unobserved
                # model-turn count against a budget. See
                # agentcheck/evaluate/engine.py::_observed_model_turns.
                "model_turns_observable": False,
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
) -> tuple[str, str | None, dict[str, Any]]:
    """The opening message ``start`` is called with.

    ``start(message, tools)`` takes one message, so a seeded assistant turn
    cannot be replayed without fabricating a turn the agent never produced.
    Multi-turn openings are rejected rather than approximated; ``followup_turns``
    is the supported way to script later user turns.
    """

    if isinstance(input_value, str):
        return input_value, None, {}
    turns = tuple(input_value)
    if not turns:
        raise ValueError("an agent run requires at least one conversation turn")
    if len(turns) != 1 or turns[0].role != ConversationRole.USER:
        raise ValueError(
            "this adapter seeds exactly one opening user turn; use followup_turns "
            "for later user turns"
        )
    turn = turns[0]
    return turn.content, turn.turn_id, dict(turn.metadata)


def _require_turn_result(value: Any, *, stage: int) -> TurnResult:
    if isinstance(value, TurnResult):
        return value
    raise AdapterRuntimeError(
        f"turn {stage} returned {type(value).__name__}, not a TurnResult. The "
        "custom contract requires a TurnResult so output, opaque state and "
        "metadata stay distinguishable."
    )


async def _flush_events(capture: _Capture, sink: EventSinkProtocol | None) -> None:
    """Hand buffered events to an optional live sink, in order, exactly once.

    Tool calls happen inside a synchronous agent turn, so events accumulate
    during the turn and drain here between turns. Ordering and completeness are
    preserved; only the timing differs from an adapter whose whole loop is async.
    """

    if sink is None:
        capture.flushed = len(capture.events)
        return
    while capture.flushed < len(capture.events):
        event = capture.events[capture.flushed]
        capture.flushed += 1
        emitted = sink.emit(event)
        if _inspect.isawaitable(emitted):
            await emitted


def _budget_resource(exc: BaseException) -> str | None:
    """The exhausted budget behind an exception, following ``__cause__``.

    A gateway budget block reaches the agent as ``ToolCallBlockedError`` raised
    ``from`` the ``BudgetExceeded`` that caused it. If the agent lets it
    propagate, the run should terminate as the budget it actually hit rather
    than as a generic adapter error.
    """

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        resource = getattr(current, "resource", None)
        if resource is not None:
            return str(resource)
        current = current.__cause__
    return None


def _map_exception(exc: BaseException) -> tuple[RunTermination | None, str]:
    """Map a failure onto a termination, or signal a re-raise with ``None``."""

    resource = _budget_resource(exc)
    if resource is not None:
        return (
            {
                "model_turns": RunTermination.MAX_MODEL_TURNS,
                "tool_calls": RunTermination.MAX_TOOL_CALLS,
                "tokens": RunTermination.TOKEN_BUDGET,
                "cost_usd": RunTermination.COST_BUDGET,
                "wall_time": RunTermination.WALL_CLOCK_TIMEOUT,
            }.get(resource, RunTermination.ADAPTER_ERROR),
            str(exc)[:4_000],
        )
    if type(exc).__name__ == "CancelledError":
        return RunTermination.CANCELLED, "Agent run was cancelled."
    if isinstance(exc, Exception):
        return RunTermination.ADAPTER_ERROR, str(exc)[:4_000]
    return None, ""


def _agentcheck_version() -> str:
    from agentcheck import __version__

    return __version__


__all__ = ["FRAMEWORK_NAME", "CustomAgentAdapter"]
