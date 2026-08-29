"""Provider-neutral execution records captured by AgentCheck workers."""

from __future__ import annotations

from enum import Enum
from typing import Literal, NamedTuple, Sequence

from pydantic import Field, model_validator

from .base import ContractModel, JsonObject, JsonValue, UtcDatetime
from .evaluation import CaseEvaluation
from .scenario import ACTION_SCENARIO_PREFIX, Scenario
from .verdict import Verdict


CANONICAL_EVENT_CONTRACT_VERSION: Literal["agentcheck.canonical_event.v1"] = (
    "agentcheck.canonical_event.v1"
)
CANONICAL_RUN_CONTRACT_VERSION: Literal["agentcheck.canonical_run.v1"] = (
    "agentcheck.canonical_run.v1"
)


class CanonicalEventType(str, Enum):
    USER_TURN = "user_turn"
    ASSISTANT_OUTPUT = "assistant_output"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_ATTEMPT = "tool_attempt"
    TOOL_RESULT = "tool_result"
    GUARDRAIL = "guardrail"
    HANDOFF = "handoff"
    ERROR = "error"
    FINAL_OUTPUT = "final_output"


class CanonicalEvent(ContractModel):
    contract_version: Literal["agentcheck.canonical_event.v1"] = (
        CANONICAL_EVENT_CONTRACT_VERSION
    )
    event_id: str = Field(min_length=1, max_length=200)
    run_id: str = Field(min_length=1, max_length=200)
    sequence: int = Field(ge=0)
    event_type: CanonicalEventType
    timestamp: UtcDatetime
    payload: JsonObject = Field(default_factory=dict)
    metadata: JsonObject = Field(default_factory=dict)
    source_event_ids: tuple[str, ...] = ()


class ToolAttempt(ContractModel):
    attempt_id: str = Field(min_length=1, max_length=200)
    event_id: str = Field(min_length=1, max_length=200)
    tool_name: str = Field(min_length=1, max_length=200)
    arguments: JsonObject = Field(default_factory=dict)
    sequence: int = Field(ge=0)
    timestamp: UtcDatetime
    state_changing: bool = False
    destructive: bool = False


class ToolOutcomeStatus(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    ERROR = "error"
    TIMEOUT = "timeout"
    MALFORMED = "malformed"
    PARTIAL = "partial"
    STALE = "stale"
    BLOCKED = "blocked"


class ToolError(ContractModel):
    code: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=4_000)
    retryable: bool | None = None
    details: JsonObject = Field(default_factory=dict)


class StateTransitionOperation(str, Enum):
    SET = "set"
    DELETE = "delete"
    APPEND = "append"
    INCREMENT = "increment"
    OTHER = "other"


class StateTransition(ContractModel):
    transition_id: str = Field(min_length=1, max_length=200)
    attempt_id: str | None = Field(default=None, max_length=200)
    path: str = Field(min_length=1, max_length=1_000)
    operation: StateTransitionOperation
    before: JsonValue = None
    after: JsonValue = None
    timestamp: UtcDatetime


class ToolOutcome(ContractModel):
    outcome_id: str = Field(min_length=1, max_length=200)
    attempt_id: str = Field(min_length=1, max_length=200)
    event_id: str = Field(min_length=1, max_length=200)
    tool_name: str = Field(min_length=1, max_length=200)
    status: ToolOutcomeStatus
    result: JsonValue = None
    error: ToolError | None = None
    started_at: UtcDatetime | None = None
    ended_at: UtcDatetime | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    state_transition_ids: tuple[str, ...] = ()
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome_shape(self) -> "ToolOutcome":
        if (self.started_at is None) != (self.ended_at is None):
            raise ValueError("tool outcome timestamps must be supplied together")
        if (
            self.started_at is not None
            and self.ended_at is not None
            and self.ended_at < self.started_at
        ):
            raise ValueError("tool outcome cannot end before it starts")
        if self.status == ToolOutcomeStatus.SUCCESS and self.error is not None:
            raise ValueError("successful tool outcomes cannot contain an error")
        if self.status in {
            ToolOutcomeStatus.ERROR,
            ToolOutcomeStatus.TIMEOUT,
            ToolOutcomeStatus.BLOCKED,
        } and self.error is None:
            raise ValueError(f"{self.status.value} tool outcomes require an error")
        return self


class UsageMetrics(ContractModel):
    """Unknown provider metrics stay ``None`` and are never interpreted as zero."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_total(self) -> "UsageMetrics":
        if (
            self.total_tokens is not None
            and self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens < self.input_tokens + self.output_tokens
        ):
            raise ValueError("total_tokens cannot be below input_tokens + output_tokens")
        return self


class RunTermination(str, Enum):
    COMPLETED = "completed"
    WALL_CLOCK_TIMEOUT = "wall_clock_timeout"
    MAX_MODEL_TURNS = "max_model_turns"
    MAX_TOOL_CALLS = "max_tool_calls"
    TOKEN_BUDGET = "token_budget"
    COST_BUDGET = "cost_budget"
    ADAPTER_ERROR = "adapter_error"
    PROVIDER_ERROR = "provider_error"
    WORKER_ERROR = "worker_error"
    CANCELLED = "cancelled"


class CanonicalRun(ContractModel):
    contract_version: Literal["agentcheck.canonical_run.v1"] = CANONICAL_RUN_CONTRACT_VERSION
    run_id: str = Field(min_length=1, max_length=200)
    scenario_id: str = Field(min_length=1, max_length=200)
    target_id: str = Field(min_length=1, max_length=200)
    started_at: UtcDatetime
    ended_at: UtcDatetime
    termination: RunTermination
    termination_reason: str | None = Field(default=None, max_length=4_000)
    events: tuple[CanonicalEvent, ...] = ()
    tool_attempts: tuple[ToolAttempt, ...] = ()
    tool_outcomes: tuple[ToolOutcome, ...] = ()
    state_transitions: tuple[StateTransition, ...] = ()
    initial_world_state: JsonObject = Field(default_factory=dict)
    final_world_state: JsonObject = Field(default_factory=dict)
    final_output: str | None = Field(default=None, max_length=100_000)
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    latency_ms: float | None = Field(default=None, ge=0.0)
    provider_request_ids: tuple[str, ...] = ()
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_trajectory_integrity(self) -> "CanonicalRun":
        if self.ended_at < self.started_at:
            raise ValueError("canonical run cannot end before it starts")

        event_ids: set[str] = set()
        sequences: list[int] = []
        for event in self.events:
            if event.run_id != self.run_id:
                raise ValueError("every canonical event must belong to the run")
            if event.event_id in event_ids:
                raise ValueError("canonical event IDs must be unique")
            event_ids.add(event.event_id)
            sequences.append(event.sequence)
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise ValueError("canonical events must have unique ascending sequence numbers")

        attempt_by_id: dict[str, ToolAttempt] = {}
        for attempt in self.tool_attempts:
            if attempt.attempt_id in attempt_by_id:
                raise ValueError("tool attempt IDs must be unique")
            if attempt.event_id not in event_ids:
                raise ValueError("tool attempts must reference a canonical event")
            attempt_by_id[attempt.attempt_id] = attempt

        transition_ids = {transition.transition_id for transition in self.state_transitions}
        if len(transition_ids) != len(self.state_transitions):
            raise ValueError("state transition IDs must be unique")

        outcome_ids: set[str] = set()
        for outcome in self.tool_outcomes:
            if outcome.outcome_id in outcome_ids:
                raise ValueError("tool outcome IDs must be unique")
            outcome_ids.add(outcome.outcome_id)
            recorded_attempt = attempt_by_id.get(outcome.attempt_id)
            if recorded_attempt is None:
                raise ValueError("tool outcomes must reference a recorded attempt")
            if recorded_attempt.tool_name != outcome.tool_name:
                raise ValueError("tool outcome name must match its attempt")
            if outcome.event_id not in event_ids:
                raise ValueError("tool outcomes must reference a canonical event")
            if not set(outcome.state_transition_ids).issubset(transition_ids):
                raise ValueError("tool outcomes reference an unknown state transition")
        return self


class ActionPathExercise(NamedTuple):
    """Legacy run-only split retained for public API compatibility.

    A pass on an action-path case means one of two very different things: the
    agent called a tool and behaved correctly, or it never called one and every
    trajectory check held vacuously. This type predates complete scenario and
    evaluation binding. Evidence-facing summaries use
    :func:`measured_action_path_exercise` instead.
    """

    exercised: tuple[str, ...]
    not_exercised: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.exercised) + len(self.not_exercised)


def action_path_exercise(runs: Sequence[CanonicalRun]) -> ActionPathExercise:
    """Split observed action-path runs by whether any tool was attempted.

    This compatibility helper cannot know about missing runs, infrastructure
    evaluations, or the scenario's focal tool. Do not use it for a complete
    evidence metric; use :func:`measured_action_path_exercise` instead.
    """

    exercised: list[str] = []
    not_exercised: list[str] = []
    for run in runs:
        if not str(run.scenario_id).startswith(ACTION_SCENARIO_PREFIX):
            continue
        if run.tool_attempts:
            exercised.append(run.scenario_id)
        else:
            not_exercised.append(run.scenario_id)
    return ActionPathExercise(tuple(exercised), tuple(not_exercised))


class MeasuredActionPathExercise(NamedTuple):
    """Evidence-safe action-path classification over the complete denominator."""

    exercised: tuple[str, ...]
    not_exercised: tuple[str, ...]
    unmeasured: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.exercised) + len(self.not_exercised) + len(self.unmeasured)


def _focal_action_tool(scenario: Scenario) -> str | None:
    if "path:action" not in scenario.dimension_tags:
        return None
    tagged = {
        tag.split(":", 1)[1]
        for tag in scenario.dimension_tags
        if tag.startswith("tool:") and tag != "tool:"
    }
    declared = {
        item.tool_name
        for item in (
            *scenario.required_tool_behavior,
            *scenario.allowed_tool_behavior,
        )
    }
    focal = tagged.intersection(declared)
    if len(focal) != 1:
        return None
    return next(iter(focal))


def measured_action_path_exercise(
    scenarios: Sequence[Scenario],
    runs: Sequence[CanonicalRun],
    evaluations: Sequence[CaseEvaluation],
) -> MeasuredActionPathExercise:
    """Classify every action path from bound run and evaluation evidence.

    The scenario set owns the denominator. Deriving it from ``runs`` would make
    an infrastructure-failed case disappear. A path is exercised only when its
    single focal tool has an observed attempt in the single canonical run bound
    to a non-infrastructure evaluation. Missing, duplicate, ambiguous, or
    infrastructure evidence remains unmeasured.
    """

    exercised: list[str] = []
    not_exercised: list[str] = []
    unmeasured: list[str] = []
    runs_by_scenario: dict[str, list[CanonicalRun]] = {}
    for run in runs:
        runs_by_scenario.setdefault(run.scenario_id, []).append(run)
    evaluations_by_scenario: dict[str, list[CaseEvaluation]] = {}
    for evaluation in evaluations:
        evaluations_by_scenario.setdefault(evaluation.scenario_id, []).append(
            evaluation
        )

    for scenario in scenarios:
        if not str(scenario.scenario_id).startswith(ACTION_SCENARIO_PREFIX):
            continue
        matching_runs = runs_by_scenario.get(scenario.scenario_id, [])
        matching_evaluations = evaluations_by_scenario.get(scenario.scenario_id, [])
        focal_tool = _focal_action_tool(scenario)
        if (
            len(matching_runs) != 1
            or len(matching_evaluations) != 1
            or focal_tool is None
        ):
            unmeasured.append(scenario.scenario_id)
            continue
        run = matching_runs[0]
        evaluation = matching_evaluations[0]
        if (
            evaluation.verdict is Verdict.INFRA_ERROR
            or evaluation.run_id != run.run_id
        ):
            unmeasured.append(scenario.scenario_id)
            continue
        if any(
            attempt.tool_name == focal_tool for attempt in run.tool_attempts
        ):
            exercised.append(scenario.scenario_id)
        else:
            not_exercised.append(scenario.scenario_id)
    return MeasuredActionPathExercise(
        tuple(exercised), tuple(not_exercised), tuple(unmeasured)
    )
