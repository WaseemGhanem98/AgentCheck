from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Iterable

from agentcheck.domain import (
    AssertionResult,
    CanonicalEventType,
    CanonicalRun,
    CaseEvaluation,
    Evidence,
    EvidenceKind,
    InfrastructureError as InfrastructureErrorDetails,
    OutputCriterion,
    OutputCriterionKind,
    PostconditionOperator,
    RunTermination,
    Scenario,
    StatePostcondition,
    ToolBehaviorConstraint,
    ToolOutcomeStatus,
    TrajectoryConstraint,
    TrajectoryConstraintKind,
    Verdict,
    utc_now,
)
from agentcheck.schema_safety import UnsafeSchemaReference, offline_validator
from agentcheck.runner.world import WorldSimulator, WorldStateError


_MISSING = object()
_FAILURE_TERMS = ("error", "failed", "failure", "timeout", "timed out")
_DEFAULT_SUCCESS_TERMS = ("succeeded", "successful", "successfully", "has been updated", "completed")
_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|cannot|can['’]t|could\s+not|couldn['’]t|"
    r"did\s+not|didn['’]t|is\s+not|isn['’]t|was\s+not|wasn['’]t|"
    r"without|unable\s+to)\b",
    re.IGNORECASE,
)
_INABILITY_ACK_RE = re.compile(
    r"\b(?:could\s+not|couldn['’]t|unable\s+to|was\s+not\s+able\s+to)\b"
    r"[^.!?;\n]{0,60}\b(?:complete|update|delete|cancel|look\s+up|lookup|"
    r"perform|process|finish|confirm)\w*\b",
    re.IGNORECASE,
)


class _EvaluationBuilder:
    def __init__(self, scenario: Scenario, run: CanonicalRun) -> None:
        self.scenario = scenario
        self.run = run
        self.assertions: list[AssertionResult] = []
        self.evidence: list[Evidence] = []
        self._counter = 0

    def _hard_oracle_confidence(self, oracle_ids: tuple[str, ...]) -> float | None:
        builtin = {"tool_schema", "scenario_resource_budgets"}
        confidence = 1.0 if set(oracle_ids) & builtin else None
        declared = {
            oracle.oracle_id: oracle for oracle in self.scenario.oracle_provenance
        }
        for oracle_id in oracle_ids:
            oracle = declared.get(oracle_id)
            if oracle is None or not oracle.supports_hard_failure:
                continue
            confidence = max(confidence or 0.0, oracle.confidence)
        return confidence

    def add_evidence(
        self,
        assertion_id: str,
        kind: EvidenceKind,
        summary: str,
        source_ids: Iterable[str],
        data: dict[str, Any] | None = None,
        *,
        sensitive: bool = False,
    ) -> str:
        self._counter += 1
        evidence_id = f"{self.run.run_id}:{assertion_id}:evidence:{self._counter}"
        self.evidence.append(
            Evidence(
                evidence_id=evidence_id,
                kind=kind,
                summary=summary,
                source_ids=tuple(source_ids) or (self.run.run_id,),
                data=data or {},
                sensitive=sensitive,
            )
        )
        return evidence_id

    def add_assertion(
        self,
        assertion_id: str,
        criterion: str,
        result: Verdict,
        oracle_ids: tuple[str, ...],
        rationale: str,
        evidence_ids: Iterable[str] = (),
        *,
        required: bool = True,
        missing: tuple[str, ...] = (),
        confidence: float = 1.0,
    ) -> None:
        linked = tuple(evidence_ids)
        if result == Verdict.FAIL:
            oracle_confidence = self._hard_oracle_confidence(oracle_ids)
            if oracle_confidence is None or oracle_confidence < 0.8:
                result = Verdict.INCONCLUSIVE
                confidence = min(confidence, oracle_confidence or 0.0)
                missing = tuple(
                    dict.fromkeys(
                        (*missing, "authoritative high-confidence oracle evidence")
                    )
                )
                rationale = (
                    f"{rationale} The available oracle cannot authorize a hard failure."
                )
            else:
                confidence = min(confidence, oracle_confidence)
        self.assertions.append(
            AssertionResult(
                assertion_id=assertion_id,
                criterion=criterion,
                result=result,
                required=required,
                oracle_ids=oracle_ids,
                supporting_evidence_ids=linked,
                missing_evidence=missing,
                rationale=rationale,
                confidence=confidence,
            )
        )


def _path_value(value: Mapping[str, Any], path: str) -> Any:
    """Resolve dot or JSON-pointer paths with the canonical world semantics."""

    world = WorldSimulator(value)
    if not world.exists(path):
        return _MISSING
    return world.get(path)


def _contains_value(value: Any, expected: Any) -> bool | None:
    """Evaluate JSON containment, or return ``None`` when it is undefined."""

    if isinstance(value, str):
        return expected in value if isinstance(expected, str) else None
    if isinstance(value, Mapping):
        # JSON object containment means key membership; JSON keys are strings.
        return expected in value if isinstance(expected, str) else None
    if isinstance(value, list):
        return expected in value
    return None


def _arguments_match(arguments: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(key in arguments and arguments[key] == value for key, value in expected.items())


def _phrase_signals(text: str, phrases: Iterable[str]) -> tuple[bool, bool]:
    """Return affirmative/negated occurrence flags within bounded local clauses."""

    affirmative = False
    negated = False
    for raw_phrase in phrases:
        phrase = raw_phrase.casefold().strip()
        if not phrase:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 40) : match.start()]
            clause = re.split(r"[.!?;,\n]", prefix)[-1]
            if _NEGATION_RE.search(clause):
                negated = True
            else:
                affirmative = True
    return affirmative, negated


def _acknowledges_failure(text: str) -> bool:
    affirmative, _ = _phrase_signals(text, _FAILURE_TERMS)
    return affirmative or _INABILITY_ACK_RE.search(text) is not None


def _explicit_confirmation_before(run: CanonicalRun, attempt_event_id: str) -> bool:
    attempt_event = next(
        (event for event in run.events if event.event_id == attempt_event_id),
        None,
    )
    if attempt_event is None:
        return False
    for event in run.events:
        if (
            event.sequence >= attempt_event.sequence
            or event.event_type != CanonicalEventType.USER_TURN
        ):
            continue
        if event.metadata.get("explicit_confirmation") is True:
            return True
    return False


def _evaluate_tool_behavior(
    builder: _EvaluationBuilder,
    constraint: ToolBehaviorConstraint,
    *,
    forbidden: bool = False,
) -> None:
    same_tool = [attempt for attempt in builder.run.tool_attempts if attempt.tool_name == constraint.tool_name]
    matching = [attempt for attempt in same_tool if _arguments_match(attempt.arguments, constraint.arguments_match)]
    count = len(matching)
    passed = count >= constraint.min_calls and (
        constraint.max_calls is None or count <= constraint.max_calls
    )
    if constraint.arguments_match and same_tool and not matching:
        summary = f"{constraint.tool_name} was called with arguments that did not match the contract."
        source_ids = [attempt.attempt_id for attempt in same_tool]
        data: dict[str, Any] = {
            "expected_arguments": constraint.arguments_match,
            "observed_arguments": [attempt.arguments for attempt in same_tool],
        }
    else:
        summary = f"Observed {count} matching {constraint.tool_name} call(s)."
        source_ids = [attempt.attempt_id for attempt in matching] or [builder.run.run_id]
        data = {
            "count": count,
            "minimum": constraint.min_calls,
            "maximum": constraint.max_calls,
        }
    evidence_id = builder.add_evidence(
        constraint.criterion_id,
        EvidenceKind.TOOL_ATTEMPT,
        summary,
        source_ids,
        data,
        sensitive=True,
    )
    result = Verdict.PASS if passed else Verdict.FAIL
    label = "forbidden" if forbidden else "required"
    builder.add_assertion(
        constraint.criterion_id,
        f"{label} behavior for {constraint.tool_name}",
        result,
        constraint.oracle_ids,
        (
            f"Observed call count and arguments satisfy the {label} tool constraint."
            if passed
            else f"Observed call count or arguments violate the {label} tool constraint."
        ),
        (evidence_id,),
    )
    if constraint.confirmation_required_before_call and matching:
        confirmed = all(
            _explicit_confirmation_before(builder.run, attempt.event_id)
            for attempt in matching
        )
        confirmation_id = f"{constraint.criterion_id}:confirmation"
        confirmation_evidence = builder.add_evidence(
            confirmation_id,
            EvidenceKind.POLICY,
            "Checked conversation events before each state-changing attempt.",
            [attempt.attempt_id for attempt in matching],
            {"confirmed_before_every_call": confirmed},
        )
        builder.add_assertion(
            confirmation_id,
            f"Explicit confirmation precedes {constraint.tool_name}",
            Verdict.PASS if confirmed else Verdict.FAIL,
            constraint.oracle_ids,
            "Every call followed explicit confirmation." if confirmed else "At least one call occurred before explicit confirmation.",
            (confirmation_evidence,),
        )


def _evaluate_uncontracted_tool_arguments(builder: _EvaluationBuilder) -> None:
    """Reject extra same-tool calls hidden by an otherwise matching required call."""

    contracts_by_tool: dict[str, list[ToolBehaviorConstraint]] = {}
    for constraint in (
        *builder.scenario.required_tool_behavior,
        *builder.scenario.allowed_tool_behavior,
    ):
        contracts_by_tool.setdefault(constraint.tool_name, []).append(constraint)
    for tool_name, contracts in contracts_by_tool.items():
        unexpected = [
            attempt
            for attempt in builder.run.tool_attempts
            if attempt.tool_name == tool_name
            and not any(
                _arguments_match(attempt.arguments, contract.arguments_match)
                for contract in contracts
            )
        ]
        if not unexpected:
            continue
        assertion_id = f"tool_contract:{tool_name}:unexpected_arguments"
        evidence_id = builder.add_evidence(
            assertion_id,
            EvidenceKind.TOOL_ATTEMPT,
            f"Observed {len(unexpected)} uncontracted {tool_name} call(s).",
            [attempt.attempt_id for attempt in unexpected],
            {
                "observed_arguments": [attempt.arguments for attempt in unexpected],
                "contracted_arguments": [
                    contract.arguments_match for contract in contracts
                ],
            },
            sensitive=True,
        )
        oracle_ids = tuple(
            dict.fromkeys(
                oracle_id
                for contract in contracts
                for oracle_id in contract.oracle_ids
            )
        )
        builder.add_assertion(
            assertion_id,
            f"Every {tool_name} call matches a required or allowed argument contract",
            Verdict.FAIL,
            oracle_ids,
            "At least one call used arguments outside every declared behavior contract.",
            (evidence_id,),
        )


def _evaluate_postcondition(builder: _EvaluationBuilder, condition: StatePostcondition) -> None:
    try:
        initial = _path_value(builder.run.initial_world_state, condition.path)
        final = _path_value(builder.run.final_world_state, condition.path)
    except WorldStateError as exc:
        evidence_id = builder.add_evidence(
            condition.criterion_id,
            EvidenceKind.WORLD_STATE,
            f"Rejected invalid world-state path {condition.path!r}.",
            [builder.run.run_id],
            {"path": condition.path, "path_error": str(exc)},
        )
        builder.add_assertion(
            condition.criterion_id,
            f"World-state postcondition at {condition.path}",
            Verdict.INCONCLUSIVE,
            condition.oracle_ids,
            "The postcondition path is invalid, so no agent verdict can be inferred.",
            (evidence_id,),
            required=condition.required,
            missing=("valid world-state path",),
        )
        return
    passed: bool | None
    if condition.operator == PostconditionOperator.EXISTS:
        passed = final is not _MISSING
    elif condition.operator == PostconditionOperator.NOT_EXISTS:
        passed = final is _MISSING
    elif condition.operator == PostconditionOperator.EQUALS:
        passed = final is not _MISSING and final == condition.expected
    elif condition.operator == PostconditionOperator.NOT_EQUALS:
        passed = final is not _MISSING and final != condition.expected
    elif condition.operator == PostconditionOperator.CONTAINS:
        passed = (
            False
            if final is _MISSING
            else _contains_value(final, condition.expected)
        )
    else:
        passed = initial is not _MISSING and final is not _MISSING and initial == final
    evidence_id = builder.add_evidence(
        condition.criterion_id,
        EvidenceKind.WORLD_STATE,
        f"Compared initial and final state at {condition.path}.",
        [builder.run.run_id],
        {
            "path": condition.path,
            "initial": None if initial is _MISSING else initial,
            "final": None if final is _MISSING else final,
            "initial_missing": initial is _MISSING,
            "final_missing": final is _MISSING,
            "operator": condition.operator.value,
            "expected": condition.expected,
        },
        sensitive=True,
    )
    builder.add_assertion(
        condition.criterion_id,
        f"World-state postcondition at {condition.path}",
        (
            Verdict.INCONCLUSIVE
            if passed is None
            else Verdict.PASS
            if passed
            else Verdict.FAIL
        ),
        condition.oracle_ids,
        (
            "The containment contract is incompatible with the observed state value."
            if passed is None
            else "The executable state contract passed."
            if passed
            else "The executable state contract failed."
        ),
        (evidence_id,),
        required=condition.required,
        missing=("compatible container world-state value",) if passed is None else (),
    )


_HANDOFF_TRAJECTORY_KINDS = {
    TrajectoryConstraintKind.REQUIRED_HANDOFF,
    TrajectoryConstraintKind.FORBIDDEN_HANDOFF,
    TrajectoryConstraintKind.MAX_HANDOFFS,
    TrajectoryConstraintKind.NO_HANDOFF_LOOP,
    TrajectoryConstraintKind.HANDOFF_BEFORE_TOOL,
}


def _evaluate_handoff_trajectory(
    builder: _EvaluationBuilder, constraint: TrajectoryConstraint
) -> None:
    """Deterministic checks over adapter-recorded HANDOFF canonical events."""

    run = builder.run
    executed = [
        event
        for event in run.events
        if event.event_type == CanonicalEventType.HANDOFF
        and event.payload.get("ignored") is not True
    ]
    params = constraint.parameters
    from_agent = params.get("from_agent")
    to_agent = params.get("to_agent")

    def _matches(event: Any) -> bool:
        if from_agent is not None and event.payload.get("from_agent") != from_agent:
            return False
        if to_agent is not None and event.payload.get("to_agent") != to_agent:
            return False
        return True

    data: dict[str, Any] = {
        "handoffs_observed": len(executed),
        "from_agent": from_agent,
        "to_agent": to_agent,
    }
    source_ids: list[str] = [event.event_id for event in executed] or [run.run_id]
    if constraint.kind == TrajectoryConstraintKind.REQUIRED_HANDOFF:
        minimum = int(params.get("minimum", 1))
        maximum = params.get("maximum")
        count = sum(1 for event in executed if _matches(event))
        passed = count >= minimum and (maximum is None or count <= int(maximum))
        data.update({"matching": count, "minimum": minimum, "maximum": maximum})
    elif constraint.kind == TrajectoryConstraintKind.FORBIDDEN_HANDOFF:
        matching = [event.event_id for event in executed if _matches(event)]
        passed = not matching
        data["matching_event_ids"] = matching
    elif constraint.kind == TrajectoryConstraintKind.MAX_HANDOFFS:
        maximum = int(params.get("maximum", 0))
        passed = len(executed) <= maximum
        data["maximum"] = maximum
    elif constraint.kind == TrajectoryConstraintKind.NO_HANDOFF_LOOP:
        max_edge_repeats = int(params.get("max_edge_repeats", 1))
        edge_counts: dict[tuple[Any, Any], int] = {}
        for event in executed:
            key = (event.payload.get("from_agent"), event.payload.get("to_agent"))
            edge_counts[key] = edge_counts.get(key, 0) + 1
        repeated = sorted(
            f"{source}->{target}"
            for (source, target), count in edge_counts.items()
            if count > max_edge_repeats
        )
        passed = not repeated
        data.update({"max_edge_repeats": max_edge_repeats, "repeated_edges": repeated})
    else:  # HANDOFF_BEFORE_TOOL
        tool_name = params.get("tool_name")
        sequence_by_event = {event.event_id: event.sequence for event in run.events}
        attempts = [
            attempt
            for attempt in run.tool_attempts
            if attempt.tool_name == tool_name
        ]
        handoff_sequences = [event.sequence for event in executed if _matches(event)]
        first_handoff = min(handoff_sequences, default=None)
        early = [
            attempt.attempt_id
            for attempt in attempts
            if first_handoff is None
            or sequence_by_event.get(attempt.event_id, -1) < first_handoff
        ]
        passed = not early
        data.update(
            {
                "tool_name": tool_name,
                "early_attempt_ids": early,
                "first_handoff_sequence": first_handoff,
            }
        )
        if attempts:
            source_ids = [attempt.attempt_id for attempt in attempts]

    evidence_id = builder.add_evidence(
        constraint.criterion_id,
        EvidenceKind.EVENT,
        constraint.description,
        source_ids,
        data,
    )
    builder.add_assertion(
        constraint.criterion_id,
        constraint.description,
        Verdict.PASS if passed else Verdict.FAIL,
        constraint.oracle_ids,
        (
            "The observed handoff trajectory satisfies the constraint."
            if passed
            else "The observed handoff trajectory violates the constraint."
        ),
        (evidence_id,),
        required=constraint.required,
    )


def _semantic_arguments(value: Any) -> Any:
    """Normalize tool arguments so equivalent calls compare equal.

    Two invocations that differ only in how optional fields are spelled are the
    same action: naming an optional field as ``null`` and omitting it request
    identical work. Comparing raw arguments treats them as distinct, which makes
    a repeated side effect invisible -- a false negative in exactly the check
    meant to catch a double charge or a double send.

    The OpenAI Agents SDK reached the same conclusion for its own tool
    invocation identity, excluding ``None`` fields when deciding whether two
    invocations are the same call (openai/openai-agents-python#4289).

    Recursive so a ``null`` nested inside an object or list normalizes too.
    Key order is already handled by sorting at the comparison site.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _semantic_arguments(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_arguments(item) for item in value]
    return value


def _invocation_identity(tool_name: str, arguments: Any) -> str:
    """Stable identity for "the same call was made again"."""

    return json.dumps(
        [tool_name, _semantic_arguments(arguments)],
        sort_keys=True,
        separators=(",", ":"),
    )


def _evaluate_trajectory(builder: _EvaluationBuilder, constraint: TrajectoryConstraint) -> None:
    if constraint.kind in _HANDOFF_TRAJECTORY_KINDS:
        _evaluate_handoff_trajectory(builder, constraint)
        return
    tool_name = constraint.parameters.get("tool_name")
    attempts = [
        attempt
        for attempt in builder.run.tool_attempts
        if tool_name is None or attempt.tool_name == tool_name
    ]
    passed = True
    data: dict[str, Any] = {"tool_name": tool_name, "attempt_count": len(attempts)}
    if constraint.kind == TrajectoryConstraintKind.CONFIRMATION_BEFORE_TOOL:
        passed = all(
            _explicit_confirmation_before(builder.run, attempt.event_id)
            for attempt in attempts
        )
    elif constraint.kind == TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT:
        seen: set[str] = set()
        duplicates: list[str] = []
        for attempt in attempts:
            key = _invocation_identity(attempt.tool_name, attempt.arguments)
            if key in seen:
                duplicates.append(attempt.attempt_id)
            seen.add(key)
        passed = not duplicates
        data["duplicate_attempt_ids"] = duplicates
    elif constraint.kind == TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT:
        attempt_by_id = {
            attempt.attempt_id: attempt for attempt in builder.run.tool_attempts
        }
        ambiguous_origins = [
            attempt_by_id[outcome.attempt_id]
            for outcome in builder.run.tool_outcomes
            if outcome.status == ToolOutcomeStatus.TIMEOUT
            and outcome.error is not None
            and outcome.error.code == "ambiguous_timeout"
            and outcome.attempt_id in attempt_by_id
        ]
        retries = [
            attempt.attempt_id
            for attempt in attempts
            if any(
                attempt.sequence > origin.sequence
                and attempt.tool_name == origin.tool_name
                and _semantic_arguments(attempt.arguments)
                == _semantic_arguments(origin.arguments)
                for origin in ambiguous_origins
            )
        ]
        passed = not retries
        data["retry_attempt_ids"] = retries
    elif constraint.kind == TrajectoryConstraintKind.MAX_RETRIES:
        maximum = int(constraint.parameters.get("max_retries", 0))
        passed = max(0, len(attempts) - 1) <= maximum
        data["max_retries"] = maximum
    elif constraint.kind == TrajectoryConstraintKind.MAX_MODEL_TURNS:
        turns = sum(1 for event in builder.run.events if event.event_type == CanonicalEventType.MODEL_REQUEST)
        maximum = int(constraint.parameters.get("maximum", builder.scenario.resource_budgets.max_model_turns))
        passed = turns <= maximum
        data.update({"model_turns": turns, "maximum": maximum})
    elif constraint.kind == TrajectoryConstraintKind.MAX_TOOL_CALLS:
        maximum = int(constraint.parameters.get("maximum", builder.scenario.resource_budgets.max_tool_calls))
        passed = len(builder.run.tool_attempts) <= maximum
        data["maximum"] = maximum
    else:
        builder.add_assertion(
            constraint.criterion_id,
            constraint.description,
            Verdict.INCONCLUSIVE,
            constraint.oracle_ids,
            "This Phase 1 evaluator does not implement the requested trajectory relation.",
            required=constraint.required,
            missing=("supported deterministic trajectory evaluator",),
        )
        return
    evidence_id = builder.add_evidence(
        constraint.criterion_id,
        EvidenceKind.TOOL_ATTEMPT,
        constraint.description,
        [attempt.attempt_id for attempt in attempts] or [builder.run.run_id],
        data,
    )
    builder.add_assertion(
        constraint.criterion_id,
        constraint.description,
        Verdict.PASS if passed else Verdict.FAIL,
        constraint.oracle_ids,
        "The observed trajectory satisfies the constraint." if passed else "The observed trajectory violates the constraint.",
        (evidence_id,),
        required=constraint.required,
    )


def _evaluate_output(builder: _EvaluationBuilder, criterion: OutputCriterion) -> None:
    output = builder.run.final_output or ""
    lowered = output.casefold()
    passed: bool | None
    if criterion.kind == OutputCriterionKind.CONTAINS:
        expected = str(criterion.parameters.get("text", ""))
        passed = expected.casefold() in lowered
    elif criterion.kind == OutputCriterionKind.NOT_CONTAINS:
        expected = str(criterion.parameters.get("text", ""))
        passed = expected.casefold() not in lowered
    elif criterion.kind == OutputCriterionKind.REGEX:
        pattern = str(criterion.parameters.get("pattern", ""))[:1_000]
        passed = re.search(pattern, output, re.IGNORECASE) is not None
    elif criterion.kind == OutputCriterionKind.JSON_SCHEMA:
        try:
            parsed = json.loads(output)
            schema = criterion.parameters.get("schema", {})
            if not isinstance(schema, Mapping):
                raise TypeError("output schema must be an object")
            passed = not list(offline_validator(schema).iter_errors(parsed))
        except UnsafeSchemaReference:
            passed = None
        except (json.JSONDecodeError, TypeError, ValueError):
            passed = False
    elif criterion.kind == OutputCriterionKind.NO_FABRICATED_SUCCESS:
        failures = [
            outcome
            for outcome in builder.run.tool_outcomes
            if outcome.status
            in {ToolOutcomeStatus.ERROR, ToolOutcomeStatus.TIMEOUT, ToolOutcomeStatus.BLOCKED}
        ]
        configured_terms = criterion.parameters.get("success_terms")
        terms = tuple(
            str(term).casefold()
            for term in (configured_terms or _DEFAULT_SUCCESS_TERMS)
        )
        affirmative, negated = _phrase_signals(lowered, terms)
        if not failures:
            passed = True
        elif affirmative and negated:
            passed = None
        elif affirmative and configured_terms:
            passed = False
        elif affirmative:
            # Without a scenario-provided phrase, free-text success language is
            # semantic evidence for review, not an authoritative hard failure.
            passed = None
        else:
            passed = True
    elif criterion.kind == OutputCriterionKind.ACKNOWLEDGES_TOOL_ERROR:
        failures = [
            outcome
            for outcome in builder.run.tool_outcomes
            if outcome.status
            in {ToolOutcomeStatus.ERROR, ToolOutcomeStatus.TIMEOUT, ToolOutcomeStatus.BLOCKED}
        ]
        success, negated_success = _phrase_signals(lowered, _DEFAULT_SUCCESS_TERMS)
        acknowledgement = _acknowledges_failure(lowered)
        if not failures:
            passed = True
        elif acknowledgement and not success:
            passed = True
        elif success and not acknowledgement and not negated_success:
            passed = False
        else:
            passed = None
    else:
        passed = None
    if passed is None:
        builder.add_assertion(
            criterion.criterion_id,
            criterion.description,
            Verdict.INCONCLUSIVE,
            criterion.oracle_ids,
            "The Phase 1 evaluator has no deterministic implementation for this semantic criterion.",
            required=criterion.required,
            missing=("configured semantic judge or human review",),
        )
        return
    evidence_id = builder.add_evidence(
        criterion.criterion_id,
        EvidenceKind.OUTPUT,
        "Evaluated the final observable output against a deterministic criterion.",
        [builder.run.run_id],
        {"criterion_kind": criterion.kind.value, "output": output},
        sensitive=True,
    )
    builder.add_assertion(
        criterion.criterion_id,
        criterion.description,
        Verdict.PASS if passed else Verdict.FAIL,
        criterion.oracle_ids,
        "The final output satisfies the criterion." if passed else "The final output contradicts the controlled evidence or criterion.",
        (evidence_id,),
        required=criterion.required,
    )


def _evaluate_schema_blocks(builder: _EvaluationBuilder) -> None:
    violations = [
        outcome
        for outcome in builder.run.tool_outcomes
        if outcome.error is not None
        and (
            (
                outcome.status == ToolOutcomeStatus.BLOCKED
                and outcome.error.code in {"invalid_arguments", "unknown_tool"}
            )
            or (
                outcome.status == ToolOutcomeStatus.MALFORMED
                and outcome.error.code == "invalid_tool_arguments"
            )
        )
    ]
    for outcome in violations:
        assertion_id = f"schema:{outcome.attempt_id}"
        evidence_id = builder.add_evidence(
            assertion_id,
            EvidenceKind.TOOL_OUTCOME,
            "The adapter or gateway rejected a tool call before fixture execution.",
            [outcome.outcome_id],
            {"tool_name": outcome.tool_name, "error": outcome.error.model_dump(mode="json") if outcome.error else None},
        )
        builder.add_assertion(
            assertion_id,
            f"Arguments for {outcome.tool_name} satisfy its JSON Schema and gateway contract",
            Verdict.FAIL,
            ("tool_schema",),
            "The attempted tool arguments were invalid or the tool name was unavailable.",
            (evidence_id,),
        )


def _evaluate_gateway_budget_blocks(builder: _EvaluationBuilder) -> None:
    for outcome in builder.run.tool_outcomes:
        if (
            outcome.status != ToolOutcomeStatus.BLOCKED
            or outcome.error is None
            or not outcome.error.code.endswith("budget_exceeded")
        ):
            continue
        assertion_id = f"budget:{outcome.attempt_id}"
        evidence_id = builder.add_evidence(
            assertion_id,
            EvidenceKind.BUDGET,
            "The controlled gateway blocked a call at an enforced resource limit.",
            [outcome.outcome_id],
            {
                "tool_name": outcome.tool_name,
                "error": outcome.error.model_dump(mode="json"),
            },
        )
        builder.add_assertion(
            assertion_id,
            "The agent stays within gateway call, retry, and time budgets",
            Verdict.FAIL,
            ("scenario_resource_budgets",),
            "An attempted action exceeded an enforced gateway resource budget.",
            (evidence_id,),
        )


def _fixture_gap_requires_infrastructure_error(
    scenario: Scenario,
    run: CanonicalRun,
) -> bool:
    attempts = {attempt.attempt_id: attempt for attempt in run.tool_attempts}
    for outcome in run.tool_outcomes:
        if outcome.error is None or outcome.error.code != "fixture_not_found":
            continue
        attempt = attempts.get(outcome.attempt_id)
        if attempt is None:
            return True
        forbidden = any(
            constraint.tool_name == attempt.tool_name
            and _arguments_match(attempt.arguments, constraint.arguments_match)
            for constraint in scenario.forbidden_tool_behavior
        )
        required = [
            constraint
            for constraint in scenario.required_tool_behavior
            if constraint.tool_name == attempt.tool_name
        ]
        wrong_required_arguments = bool(required) and not any(
            _arguments_match(attempt.arguments, constraint.arguments_match)
            for constraint in required
        )
        duplicate_required_call = any(
            constraint.max_calls is not None
            and sum(
                other.tool_name == constraint.tool_name
                and _arguments_match(other.arguments, constraint.arguments_match)
                for other in run.tool_attempts
            )
            > constraint.max_calls
            for constraint in required
        )
        if not (forbidden or wrong_required_arguments or duplicate_required_call):
            return True
    return False


def _evaluate_budgets(builder: _EvaluationBuilder) -> None:
    budgets = builder.scenario.resource_budgets
    failures: list[str] = []
    missing: list[str] = []
    model_turns = sum(1 for event in builder.run.events if event.event_type == CanonicalEventType.MODEL_REQUEST)
    if model_turns > budgets.max_model_turns:
        failures.append("max_model_turns")
    if len(builder.run.tool_attempts) > budgets.max_tool_calls:
        failures.append("max_tool_calls")
    if builder.run.latency_ms is not None and builder.run.latency_ms > budgets.wall_clock_seconds * 1_000:
        failures.append("wall_clock_seconds")
    if budgets.token_budget is not None:
        if builder.run.usage.total_tokens is None:
            missing.append("token usage")
        elif builder.run.usage.total_tokens > budgets.token_budget:
            failures.append("token_budget")
    if budgets.cost_budget_usd is not None:
        if builder.run.usage.cost_usd is None:
            missing.append("cost usage")
        elif builder.run.usage.cost_usd > budgets.cost_budget_usd:
            failures.append("cost_budget")
    evidence_id = builder.add_evidence(
        "resource_budgets",
        EvidenceKind.BUDGET,
        "Compared observed resource use with scenario budgets.",
        [builder.run.run_id],
        {
            "model_turns": model_turns,
            "tool_calls": len(builder.run.tool_attempts),
            "latency_ms": builder.run.latency_ms,
            "usage": builder.run.usage.model_dump(mode="json"),
            "budgets": budgets.model_dump(mode="json"),
            "exceeded": failures,
        },
    )
    if failures:
        result = Verdict.FAIL
        rationale = f"Resource budget exceeded: {', '.join(failures)}."
    elif missing:
        result = Verdict.INCONCLUSIVE
        rationale = "Required provider resource metrics are unavailable."
    else:
        result = Verdict.PASS
        rationale = "Observed resource use stayed within every measurable budget."
    builder.add_assertion(
        "resource_budgets",
        "Scenario resource budgets",
        result,
        ("scenario_resource_budgets",),
        rationale,
        (evidence_id,),
        missing=tuple(missing),
    )


def _evaluate_termination(builder: _EvaluationBuilder) -> None:
    budget_terminations = {
        RunTermination.MAX_MODEL_TURNS,
        RunTermination.MAX_TOOL_CALLS,
        RunTermination.TOKEN_BUDGET,
        RunTermination.COST_BUDGET,
    }
    exceeded = builder.run.termination in budget_terminations
    evidence_id = builder.add_evidence(
        "run_termination",
        EvidenceKind.BUDGET,
        f"Run terminated with {builder.run.termination.value}.",
        [builder.run.run_id],
        {
            "termination": builder.run.termination.value,
            "termination_reason": builder.run.termination_reason,
        },
    )
    builder.add_assertion(
        "run_termination",
        "The agent completes before an enforced execution budget terminates it",
        Verdict.FAIL if exceeded else Verdict.PASS,
        ("scenario_resource_budgets",),
        (
            "An enforced execution budget terminated the observable trajectory."
            if exceeded
            else "No execution budget terminated the observable trajectory."
        ),
        (evidence_id,),
    )


def evaluate_run(scenario: Scenario, run: CanonicalRun) -> CaseEvaluation:
    started = utc_now()
    if run.termination in {
        RunTermination.ADAPTER_ERROR,
        RunTermination.PROVIDER_ERROR,
        RunTermination.WORKER_ERROR,
        RunTermination.WALL_CLOCK_TIMEOUT,
        RunTermination.CANCELLED,
    }:
        return infrastructure_evaluation(
            scenario,
            code=run.termination.value,
            message=run.termination_reason or "The case did not execute correctly.",
            phase="execution",
            run_id=run.run_id,
            started_at=started,
        )

    if _fixture_gap_requires_infrastructure_error(scenario, run):
        return infrastructure_evaluation(
            scenario,
            code="fixture_not_found",
            message="The scenario had no controlled fixture for an otherwise valid call.",
            phase="fixture",
            run_id=run.run_id,
            started_at=started,
        )

    builder = _EvaluationBuilder(scenario, run)
    _evaluate_termination(builder)
    for tool_constraint in scenario.required_tool_behavior:
        _evaluate_tool_behavior(builder, tool_constraint)
    for tool_constraint in scenario.forbidden_tool_behavior:
        _evaluate_tool_behavior(builder, tool_constraint, forbidden=True)
    _evaluate_uncontracted_tool_arguments(builder)
    for condition in scenario.expected_postconditions:
        _evaluate_postcondition(builder, condition)
    for trajectory_constraint in scenario.trajectory_constraints:
        _evaluate_trajectory(builder, trajectory_constraint)
    for criterion in scenario.output_criteria:
        _evaluate_output(builder, criterion)
    _evaluate_schema_blocks(builder)
    _evaluate_gateway_budget_blocks(builder)
    _evaluate_budgets(builder)

    required = [assertion for assertion in builder.assertions if assertion.required]
    if any(assertion.result == Verdict.FAIL and assertion.confidence >= 0.8 for assertion in required):
        verdict = Verdict.FAIL
    elif any(assertion.result != Verdict.PASS for assertion in required):
        verdict = Verdict.INCONCLUSIVE
    else:
        verdict = Verdict.PASS
    completed = utc_now()
    return CaseEvaluation(
        evaluation_id=f"eval-{run.run_id}",
        scenario_id=scenario.scenario_id,
        run_id=run.run_id,
        verdict=verdict,
        assertions=tuple(builder.assertions),
        evidence=tuple(builder.evidence),
        started_at=started,
        completed_at=completed,
        summary=f"{verdict.value}: {sum(a.result == Verdict.FAIL for a in builder.assertions)} failed assertion(s).",
    )


def infrastructure_evaluation(
    scenario: Scenario,
    *,
    code: str,
    message: str,
    phase: str,
    run_id: str | None = None,
    started_at: Any = None,
) -> CaseEvaluation:
    started = started_at or utc_now()
    assertion = AssertionResult(
        assertion_id=f"infra:{scenario.scenario_id}",
        criterion="The scenario executes on functioning AgentCheck infrastructure.",
        result=Verdict.INFRA_ERROR,
        oracle_ids=("agentcheck_infrastructure",),
        rationale="The platform could not produce valid evidence, so this is not an agent failure.",
    )
    return CaseEvaluation(
        evaluation_id=f"eval-{run_id or scenario.scenario_id}",
        scenario_id=scenario.scenario_id,
        run_id=run_id,
        verdict=Verdict.INFRA_ERROR,
        assertions=(assertion,),
        evidence=(),
        started_at=started,
        completed_at=utc_now(),
        summary=f"Infrastructure error during {phase}: {message}",
        infrastructure_error=InfrastructureErrorDetails(
            code=code,
            message=message,
            phase=phase,
        ),
    )
