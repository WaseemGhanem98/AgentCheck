"""Fail-closed replacement gateway for all tools used during an AgentCheck case."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import SchemaError  # type: ignore[import-untyped]
from pydantic import BaseModel

from agentcheck.domain.base import utc_now
from agentcheck.domain.run import (
    CanonicalEvent,
    CanonicalEventType,
    StateTransition,
    StateTransitionOperation,
    ToolAttempt,
    ToolError,
    ToolOutcome,
    ToolOutcomeStatus,
)
from agentcheck.schema_safety import UnsafeSchemaReference, offline_validator

from .budgets import BudgetExceeded, BudgetTracker
from .world import WorldSimulator, WorldStateError, WorldTransition


class ToolGatewayError(RuntimeError):
    """Base class for controlled gateway failures."""


class UnsafeToolSpecificationError(ToolGatewayError):
    """A tool definition contains a live handler or cannot be safely replaced."""


class FixtureDefinitionError(ToolGatewayError):
    """A configured fixture or schema is invalid."""


class ToolCallBlockedError(ToolGatewayError):
    """A call was recorded but deliberately withheld from the agent runtime."""

    def __init__(self, message: str, outcome: ToolOutcome) -> None:
        self.outcome = outcome
        super().__init__(message)


class UnknownToolError(ToolCallBlockedError):
    """The agent attempted a tool outside the scenario allowlist."""


class FixtureNotFoundError(ToolCallBlockedError):
    """No unused fixture permits the observed invocation."""


@dataclass(frozen=True)
class _ToolConfig:
    name: str
    input_schema: dict[str, Any]
    validator: Draft202012Validator
    state_changing: bool
    destructive: bool


@dataclass(frozen=True)
class _FixtureConfig:
    source: Any
    fixture_id: str
    tool_name: str
    expected_arguments: dict[str, Any] | None
    exact_arguments: bool
    invocation_index: int | None
    repeat: bool
    priority: int


_HANDLER_FIELDS = (
    "handler",
    "function",
    "callable",
    "callback",
    "on_invoke_tool",
    "invoke",
)


def _has_field(value: Any, name: str) -> bool:
    if isinstance(value, Mapping):
        return name in value
    if isinstance(value, BaseModel):
        return name in type(value).model_fields
    return False


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if isinstance(value, BaseModel) and name in type(value).model_fields:
            return getattr(value, name)
    return default


def _unwrap_property(value: Any) -> Any:
    candidate = _field(value, "value", default=None)
    if candidate is not None and _field(candidate, "name", default=None) is not None:
        return candidate
    return value


def _json_value(value: Any, *, label: str) -> Any:
    """Copy JSON data and reject custom objects, NaN, and non-string keys."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise FixtureDefinitionError(f"{label} must be JSON-compatible") from exc


def _recursive_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _recursive_subset(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and expected == actual
    return expected == actual


def _status(value: Any) -> ToolOutcomeStatus:
    raw = value.value if isinstance(value, Enum) else value
    try:
        status = ToolOutcomeStatus(str(raw).lower())
    except ValueError as exc:
        raise FixtureDefinitionError(
            f"unsupported simulated tool status {raw!r}"
        ) from exc
    if status == ToolOutcomeStatus.BLOCKED:
        raise FixtureDefinitionError(
            "blocked is a gateway status, not a fixture outcome"
        )
    return status


class ToolGateway:
    """Validate and simulate tool calls without retaining any original handler.

    One gateway should be created per scenario.  Tool definitions are inert data,
    fixtures are consumed at most once by default, and every result is produced
    synchronously without sleeping or touching an external system.
    """

    def __init__(
        self,
        tools: Iterable[Any] | Mapping[str, Any],
        fixtures: Iterable[Any] | Mapping[str, Any],
        *,
        world: WorldSimulator | Mapping[str, Any] | None = None,
        budgets: Any | BudgetTracker | None = None,
        run_id: str = "agentcheck-run",
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        if not run_id or len(run_id) > 200:
            raise ValueError("run_id must contain between 1 and 200 characters")
        self.run_id = run_id
        self.world = (
            WorldSimulator(world.snapshot())
            if isinstance(world, WorldSimulator)
            else WorldSimulator(world)
        )
        self.budgets = (
            budgets if isinstance(budgets, BudgetTracker) else BudgetTracker(budgets)
        )
        self._now = now
        self._tools = self._normalize_tools(tools)
        self._fixtures = self._normalize_fixtures(fixtures)
        unknown_fixture_tools = set(self._fixtures).difference(self._tools)
        if unknown_fixture_tools:
            names = ", ".join(sorted(unknown_fixture_tools))
            raise FixtureDefinitionError(f"fixtures reference unknown tools: {names}")

        self._used_fixtures: set[tuple[str, int]] = set()
        self._invocations: defaultdict[str, int] = defaultdict(int)
        self._signature_status: dict[tuple[str, str], ToolOutcomeStatus] = {}
        self._attempts: list[ToolAttempt] = []
        self._outcomes: list[ToolOutcome] = []
        self._transitions: list[StateTransition] = []
        self._events: list[CanonicalEvent] = []

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    @property
    def attempts(self) -> tuple[ToolAttempt, ...]:
        return tuple(self._attempts)

    @property
    def outcomes(self) -> tuple[ToolOutcome, ...]:
        return tuple(self._outcomes)

    @property
    def state_transitions(self) -> tuple[StateTransition, ...]:
        return tuple(self._transitions)

    @property
    def transitions(self) -> tuple[StateTransition, ...]:
        """Compatibility name consumed by framework-neutral adapters."""

        return self.state_transitions

    @property
    def events(self) -> tuple[CanonicalEvent, ...]:
        return tuple(self._events)

    def _normalize_tools(
        self, tools: Iterable[Any] | Mapping[str, Any]
    ) -> dict[str, _ToolConfig]:
        entries: list[Any] = []
        if isinstance(tools, Mapping):
            if "name" in tools:
                entries.append(tools)
            else:
                for name, definition in tools.items():
                    if isinstance(definition, Mapping):
                        if any(
                            key in definition
                            for key in (
                                "input_schema",
                                "parameters_schema",
                                "json_schema",
                                "params_json_schema",
                            )
                        ):
                            entries.append({"name": name, **definition})
                        else:
                            entries.append({"name": name, "input_schema": definition})
                    else:
                        entries.append(definition)
        else:
            entries = list(tools)

        normalized: dict[str, _ToolConfig] = {}
        for raw_entry in entries:
            entry = _unwrap_property(raw_entry)
            if callable(entry):
                raise UnsafeToolSpecificationError(
                    "live tool callables are not accepted"
                )
            for field_name in _HANDLER_FIELDS:
                if _has_field(entry, field_name):
                    raise UnsafeToolSpecificationError(
                        f"tool definitions must not contain live field {field_name!r}"
                    )
            name = _field(entry, "name", "tool_name")
            if not isinstance(name, str) or not name:
                raise UnsafeToolSpecificationError("tool definition requires a name")
            if name in normalized:
                raise UnsafeToolSpecificationError(
                    f"duplicate tool definition {name!r}"
                )
            if _has_field(entry, "replaceable") and not bool(
                _field(entry, "replaceable")
            ):
                raise UnsafeToolSpecificationError(
                    f"tool {name!r} is not safely replaceable"
                )

            schema_source = _field(
                entry,
                "input_schema",
                "parameters_schema",
                "json_schema",
                "params_json_schema",
                default={"type": "object"},
            )
            schema = _json_value(schema_source, label=f"schema for tool {name!r}")
            if not isinstance(schema, dict):
                raise UnsafeToolSpecificationError(
                    f"schema for tool {name!r} must be an object"
                )
            try:
                validator = offline_validator(schema, check_formats=True)
            except (SchemaError, UnsafeSchemaReference) as exc:
                raise UnsafeToolSpecificationError(
                    f"schema for tool {name!r} is not safe Draft 2020-12 JSON Schema"
                ) from exc
            normalized[name] = _ToolConfig(
                name=name,
                input_schema=schema,
                validator=validator,
                state_changing=bool(_field(entry, "state_changing", default=False)),
                destructive=bool(_field(entry, "destructive", default=False)),
            )
        return normalized

    def _normalize_fixtures(
        self, fixtures: Iterable[Any] | Mapping[str, Any]
    ) -> dict[str, list[_FixtureConfig]]:
        if isinstance(fixtures, Mapping):
            entries: list[Any] = []
            for name, value in fixtures.items():
                values = value if isinstance(value, (list, tuple)) else [value]
                for fixture in values:
                    if isinstance(fixture, Mapping):
                        entries.append({"tool_name": name, **fixture})
                    else:
                        entries.append(fixture)
        else:
            entries = list(fixtures)

        normalized: dict[str, list[_FixtureConfig]] = defaultdict(list)
        fixture_ids: set[str] = set()
        for position, entry in enumerate(entries, start=1):
            if callable(entry):
                raise FixtureDefinitionError("fixture handlers are not accepted")
            configured_outcome = _field(entry, "outcome", default=entry)
            for field_name in _HANDLER_FIELDS:
                if _has_field(entry, field_name) or _has_field(
                    configured_outcome, field_name
                ):
                    raise FixtureDefinitionError(
                        f"tool fixtures must not contain live field {field_name!r}"
                    )
            tool_name = _field(entry, "tool_name", "name")
            if not isinstance(tool_name, str) or not tool_name:
                raise FixtureDefinitionError("tool fixture requires tool_name")
            expected = _field(
                entry,
                "arguments_match",
                "match_args",
                "expected_arguments",
                default=None,
            )
            if expected is not None:
                expected = _json_value(
                    expected, label=f"arguments for fixture {position}"
                )
                if not isinstance(expected, dict):
                    raise FixtureDefinitionError(
                        "fixture argument match must be an object"
                    )
            invocation_index = _field(
                entry,
                "invocation_index",
                "call_index",
                "attempt_index",
                default=None,
            )
            if invocation_index is not None and (
                isinstance(invocation_index, bool)
                or not isinstance(invocation_index, int)
                or invocation_index < 1
            ):
                raise FixtureDefinitionError(
                    "fixture invocation index must be at least one"
                )
            match_mode = str(_field(entry, "match_mode", default="subset")).lower()
            if match_mode not in {"subset", "exact"}:
                raise FixtureDefinitionError(
                    "fixture match_mode must be 'subset' or 'exact'"
                )
            fixture_id = _field(
                entry, "fixture_id", "id", default=f"fixture-{position:04d}"
            )
            if not isinstance(fixture_id, str) or not fixture_id:
                raise FixtureDefinitionError("fixture_id must be a non-empty string")
            if fixture_id in fixture_ids:
                raise FixtureDefinitionError(f"duplicate fixture_id {fixture_id!r}")
            fixture_ids.add(fixture_id)
            priority = _field(entry, "priority", default=0)
            if isinstance(priority, bool) or not isinstance(priority, int):
                raise FixtureDefinitionError("fixture priority must be an integer")
            normalized[tool_name].append(
                _FixtureConfig(
                    source=entry,
                    fixture_id=fixture_id,
                    tool_name=tool_name,
                    expected_arguments=expected,
                    exact_arguments=match_mode == "exact"
                    or bool(_field(entry, "exact_arguments", default=False)),
                    invocation_index=invocation_index,
                    repeat=bool(_field(entry, "repeat", default=False)),
                    priority=priority,
                )
            )
        return dict(normalized)

    def _id(self, kind: str, index: int) -> str:
        return f"{kind}-{index:04d}"

    def _append_event(
        self,
        event_type: CanonicalEventType,
        payload: dict[str, Any],
        *,
        timestamp: datetime,
    ) -> CanonicalEvent:
        event = CanonicalEvent(
            event_id=self._id("event", len(self._events) + 1),
            run_id=self.run_id,
            sequence=len(self._events),
            event_type=event_type,
            timestamp=timestamp,
            payload=_json_value(payload, label="canonical event payload"),
        )
        self._events.append(event)
        return event

    def _record_attempt(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool: _ToolConfig | None,
        timestamp: datetime,
    ) -> ToolAttempt:
        attempt_id = self._id("attempt", len(self._attempts) + 1)
        event = self._append_event(
            CanonicalEventType.TOOL_ATTEMPT,
            {
                "attempt_id": attempt_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "state_changing": tool.state_changing if tool else False,
                "destructive": tool.destructive if tool else False,
            },
            timestamp=timestamp,
        )
        attempt = ToolAttempt(
            attempt_id=attempt_id,
            event_id=event.event_id,
            tool_name=tool_name,
            arguments=arguments,
            sequence=event.sequence,
            timestamp=timestamp,
            state_changing=tool.state_changing if tool else False,
            destructive=tool.destructive if tool else False,
        )
        self._attempts.append(attempt)
        return attempt

    def _record_outcome(
        self,
        attempt: ToolAttempt,
        *,
        status: ToolOutcomeStatus,
        result: Any = None,
        error: ToolError | None = None,
        latency_ms: float = 0.0,
        transitions: Sequence[StateTransition] = (),
        metadata: Mapping[str, Any] | None = None,
        started_at: datetime,
    ) -> ToolOutcome:
        ended_at = self._now()
        safe_result = _json_value(result, label="tool fixture result")
        safe_metadata = _json_value(dict(metadata or {}), label="tool outcome metadata")
        outcome_id = self._id("outcome", len(self._outcomes) + 1)
        event = self._append_event(
            CanonicalEventType.TOOL_RESULT,
            {
                "outcome_id": outcome_id,
                "attempt_id": attempt.attempt_id,
                "tool_name": attempt.tool_name,
                "status": status.value,
                "result": safe_result,
                "error": error.model_dump(mode="json") if error is not None else None,
                "state_transition_ids": [item.transition_id for item in transitions],
                **safe_metadata,
            },
            timestamp=ended_at,
        )
        outcome = ToolOutcome(
            outcome_id=outcome_id,
            attempt_id=attempt.attempt_id,
            event_id=event.event_id,
            tool_name=attempt.tool_name,
            status=status,
            result=safe_result,
            error=error,
            started_at=started_at,
            ended_at=ended_at,
            latency_ms=latency_ms,
            state_transition_ids=tuple(item.transition_id for item in transitions),
            metadata=safe_metadata,
        )
        self._outcomes.append(outcome)
        signature = (
            attempt.tool_name,
            json.dumps(attempt.arguments, sort_keys=True, separators=(",", ":")),
        )
        self._signature_status[signature] = status
        return outcome

    def _blocked(
        self,
        attempt: ToolAttempt,
        *,
        code: str,
        message: str,
        started_at: datetime,
        details: Mapping[str, Any] | None = None,
    ) -> ToolOutcome:
        return self._record_outcome(
            attempt,
            status=ToolOutcomeStatus.BLOCKED,
            error=ToolError(
                code=code,
                message=message,
                retryable=False,
                details=_json_value(dict(details or {}), label="blocked tool details"),
            ),
            metadata={"blocked": True},
            started_at=started_at,
        )

    def _select_fixture(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        invocation_index: int,
    ) -> _FixtureConfig | None:
        candidates: list[tuple[tuple[int, int, int, int], int, _FixtureConfig]] = []
        for position, fixture in enumerate(self._fixtures.get(tool_name, [])):
            key = (tool_name, position)
            if key in self._used_fixtures and not fixture.repeat:
                continue
            if (
                fixture.invocation_index is not None
                and fixture.invocation_index != invocation_index
            ):
                continue
            if fixture.expected_arguments is not None:
                matches = (
                    fixture.expected_arguments == arguments
                    if fixture.exact_arguments
                    else _recursive_subset(fixture.expected_arguments, arguments)
                )
                if not matches:
                    continue
            specificity = self._argument_specificity(fixture.expected_arguments)
            score = (
                fixture.priority,
                int(fixture.invocation_index is not None),
                int(fixture.exact_arguments),
                specificity,
            )
            candidates.append((score, position, fixture))
        if not candidates:
            return None
        best_score = max(item[0] for item in candidates)
        best = [item for item in candidates if item[0] == best_score]
        if len(best) > 1:
            fixture_ids = ", ".join(sorted(item[2].fixture_id for item in best))
            raise FixtureDefinitionError(
                f"ambiguous fixtures matched one invocation: {fixture_ids}"
            )
        _, position, selected = best[0]
        if not selected.repeat:
            self._used_fixtures.add((tool_name, position))
        return selected

    @staticmethod
    def _argument_specificity(value: Any) -> int:
        if isinstance(value, Mapping):
            return sum(
                1 + ToolGateway._argument_specificity(item) for item in value.values()
            )
        if isinstance(value, list):
            return len(value) + sum(
                ToolGateway._argument_specificity(item) for item in value
            )
        return 1

    def _fixture_outcome(self, fixture: _FixtureConfig) -> Any:
        return _field(fixture.source, "outcome", default=fixture.source)

    def _apply_effects(
        self,
        effects: Iterable[Any],
        *,
        world: WorldSimulator,
        attempt: ToolAttempt,
        timestamp: datetime,
    ) -> list[StateTransition]:
        snapshot = world.snapshot()
        created: list[StateTransition] = []
        try:
            for effect in effects:
                if isinstance(effect, Mapping) and (
                    "operation" in effect or "op" in effect
                ):
                    raw_transition = world.apply_effect(effect)
                else:
                    path = _field(effect, "path")
                    if not isinstance(path, str) or not path:
                        raise FixtureDefinitionError("state effect requires a path")
                    fields_set: set[str] = set(
                        getattr(effect, "model_fields_set", set())
                    )
                    before_supplied = (
                        "before" in effect
                        if isinstance(effect, Mapping)
                        else "before" in fields_set
                    )
                    expected_before = _field(effect, "before", default=None)
                    if before_supplied:
                        if not world.exists(path):
                            raise WorldStateError(
                                f"state effect expected an existing value at path {path!r}"
                            )
                        actual_before = world.get(path)
                        if actual_before != expected_before:
                            raise WorldStateError(
                                f"state effect precondition did not match at path {path!r}"
                            )
                    raw_transition = world.set(
                        path, _field(effect, "after", default=None)
                    )
                transition = self._domain_transition(
                    raw_transition,
                    attempt_id=attempt.attempt_id,
                    timestamp=timestamp,
                    index=len(self._transitions) + len(created) + 1,
                )
                created.append(transition)
        except Exception:
            world._restore(snapshot)
            raise
        self._transitions.extend(created)
        return created

    def _domain_transition(
        self,
        transition: WorldTransition,
        *,
        attempt_id: str,
        timestamp: datetime,
        index: int,
    ) -> StateTransition:
        operation = {
            "set": StateTransitionOperation.SET,
            "delete": StateTransitionOperation.DELETE,
            "append": StateTransitionOperation.APPEND,
            "increment": StateTransitionOperation.INCREMENT,
        }.get(transition.operation, StateTransitionOperation.OTHER)
        return StateTransition(
            transition_id=self._id("transition", index),
            attempt_id=attempt_id,
            path=transition.path,
            operation=operation,
            before=_json_value(transition.before, label="state before value"),
            after=_json_value(transition.after, label="state after value"),
            timestamp=timestamp,
        )

    def invoke(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        world_state: WorldSimulator | Mapping[str, Any] | None = None,
    ) -> ToolOutcome:
        """Simulate one tool call; no path in this method invokes application code."""

        started_at = self._now()
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("tool_name must be a non-empty string")
        if not isinstance(arguments, Mapping):
            raise ValueError("tool arguments must be a mapping")
        safe_arguments = _json_value(dict(arguments), label="tool arguments")
        if not isinstance(
            safe_arguments, dict
        ):  # pragma: no cover - dict conversion above
            raise ValueError("tool arguments must be a JSON object")

        tool = self._tools.get(tool_name)
        try:
            self.budgets.consume_tool_call()
        except BudgetExceeded as exc:
            attempt = self._record_attempt(tool_name, safe_arguments, tool, started_at)
            outcome = self._blocked(
                attempt,
                code=f"{exc.resource}_budget_exceeded",
                message=f"{exc.resource} budget exceeded",
                details={"limit": exc.limit, "observed": exc.observed},
                started_at=started_at,
            )
            raise ToolCallBlockedError(str(exc), outcome) from exc
        attempt = self._record_attempt(tool_name, safe_arguments, tool, started_at)
        self._invocations[tool_name] += 1
        invocation_index = self._invocations[tool_name]

        signature = (
            tool_name,
            json.dumps(safe_arguments, sort_keys=True, separators=(",", ":")),
        )
        if self._signature_status.get(signature) in {
            ToolOutcomeStatus.ERROR,
            ToolOutcomeStatus.TIMEOUT,
        }:
            try:
                self.budgets.consume_retry()
            except BudgetExceeded as exc:
                outcome = self._blocked(
                    attempt,
                    code="retry_budget_exceeded",
                    message="tool retry budget exceeded",
                    details={"limit": exc.limit, "observed": exc.observed},
                    started_at=started_at,
                )
                raise ToolCallBlockedError(
                    "tool retry budget exceeded", outcome
                ) from exc
        if tool is None:
            outcome = self._blocked(
                attempt,
                code="unknown_tool",
                message="tool is not in the AgentCheck allowlist",
                started_at=started_at,
            )
            raise UnknownToolError(
                f"tool {tool_name!r} is not in the AgentCheck allowlist",
                outcome,
            )

        validation_errors = sorted(
            tool.validator.iter_errors(safe_arguments),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
        if validation_errors:
            schema_error = validation_errors[0]
            path = ".".join(str(part) for part in schema_error.absolute_path)
            return self._record_outcome(
                attempt,
                status=ToolOutcomeStatus.BLOCKED,
                error=ToolError(
                    code="invalid_arguments",
                    message="tool arguments do not satisfy the declared JSON Schema",
                    retryable=True,
                    details={
                        "path": path,
                        "validator": str(schema_error.validator),
                    },
                ),
                metadata={"schema_valid": False},
                started_at=started_at,
            )

        fixture = self._select_fixture(tool_name, safe_arguments, invocation_index)
        if fixture is None:
            outcome = self._blocked(
                attempt,
                code="fixture_not_found",
                message="no controlled fixture matched this invocation",
                details={"invocation_index": invocation_index},
                started_at=started_at,
            )
            raise FixtureNotFoundError(
                f"no controlled fixture matched invocation {invocation_index} for {tool_name!r}",
                outcome,
            )

        configured = self._fixture_outcome(fixture)
        status = _status(_field(configured, "status", default="success"))
        result = _json_value(
            _field(configured, "result", "response", "value", default=None),
            label="tool fixture result",
        )
        outcome_error: ToolError | None = None
        if status in {ToolOutcomeStatus.ERROR, ToolOutcomeStatus.TIMEOUT}:
            error_source = _field(configured, "error", default=None)
            error_code = _field(
                configured,
                "error_code",
                default=_field(error_source, "code", default=status.value),
            )
            error_message = _field(
                configured,
                "error_message",
                default=_field(
                    error_source, "message", default=f"simulated {status.value}"
                ),
            )
            outcome_error = ToolError(
                code=str(error_code or status.value),
                message=str(error_message or f"simulated {status.value}"),
                retryable=_field(
                    error_source,
                    "retryable",
                    default=status == ToolOutcomeStatus.TIMEOUT,
                ),
                details=_json_value(
                    _field(error_source, "details", default={}),
                    label="simulated tool error details",
                ),
            )

        ambiguous_state = status == ToolOutcomeStatus.TIMEOUT and tool.state_changing
        if ambiguous_state and outcome_error is not None:
            original_code = outcome_error.code
            details = dict(outcome_error.details)
            if original_code != "ambiguous_timeout":
                details["simulated_error_code"] = original_code
            outcome_error = outcome_error.model_copy(
                update={"code": "ambiguous_timeout", "details": details}
            )

        effect_values = _field(
            configured,
            "state_effects",
            "effects",
            "world_effects",
            default=(),
        )
        latency_value = _field(configured, "latency_ms", default=0.0)
        if isinstance(latency_value, bool) or not isinstance(
            latency_value, (int, float)
        ):
            raise FixtureDefinitionError(
                "fixture latency_ms must be a non-negative number"
            )
        latency_ms = float(latency_value)
        if latency_ms < 0:
            raise FixtureDefinitionError(
                "fixture latency_ms must be a non-negative number"
            )
        try:
            self.budgets.add_simulated_time(latency_ms / 1_000.0)
        except BudgetExceeded as exc:
            outcome = self._blocked(
                attempt,
                code="wall_time_budget_exceeded",
                message="simulated tool latency exceeded the wall-time budget",
                details={"limit": exc.limit, "observed": exc.observed},
                started_at=started_at,
            )
            raise ToolCallBlockedError(str(exc), outcome) from exc
        active_world = self._resolve_world(world_state)
        transitions = self._apply_effects(
            effect_values or (),
            world=active_world,
            attempt=attempt,
            timestamp=self._now(),
        )
        metadata = {
            "fixture_id": fixture.fixture_id,
            "schema_valid": True,
            "ambiguous_state": ambiguous_state,
            "simulated": True,
        }
        if status == ToolOutcomeStatus.STALE:
            metadata["stale"] = True
        elif status == ToolOutcomeStatus.PARTIAL:
            metadata["partial"] = True
        elif status == ToolOutcomeStatus.MALFORMED:
            metadata["malformed"] = True
        return self._record_outcome(
            attempt,
            status=status,
            result=result,
            error=outcome_error,
            latency_ms=latency_ms,
            transitions=transitions,
            metadata=metadata,
            started_at=started_at,
        )

    def _resolve_world(
        self, world_state: WorldSimulator | Mapping[str, Any] | None
    ) -> WorldSimulator:
        if world_state is None:
            return self.world
        if isinstance(world_state, WorldSimulator):
            supplied = world_state.snapshot()
        elif isinstance(world_state, Mapping):
            supplied = _json_value(dict(world_state), label="adapter world state")
        else:
            raise TypeError("world_state must be a WorldSimulator, mapping, or null")
        if isinstance(supplied, Mapping):
            if (
                supplied != self.world.initial_state
                and supplied != self.world.snapshot()
            ):
                if self.world.initial_state or self._transitions:
                    raise ValueError(
                        "adapter world_state does not match the gateway world"
                    )
                self.world = WorldSimulator(supplied)
            # The caller's mapping is never mutated.  The gateway-owned copy is
            # persistent across calls in this scenario.
            return self.world
        raise TypeError("world_state must resolve to a mapping")


__all__ = [
    "FixtureDefinitionError",
    "FixtureNotFoundError",
    "ToolCallBlockedError",
    "ToolGateway",
    "ToolGatewayError",
    "UnknownToolError",
    "UnsafeToolSpecificationError",
]
