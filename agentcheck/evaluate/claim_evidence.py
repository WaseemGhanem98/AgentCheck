"""Canonical evidence -> identity -> integrity/authority -> temporal resolution.

Tool protocols are explicit codecs, not tool-prefix guesses. Candidate discovery
never grants authority. Every codec returns an inspectable requirement ledger.
"""

from __future__ import annotations

import json
from dataclasses import replace
from agentcheck.domain import (
    CanonicalRun,
    Scenario,
    ToolAttempt,
    ToolOutcome,
    ToolOutcomeStatus,
)
from agentcheck.json_values import json_values_equal
from .claim_protocols import (
    _IDENTITY_KEYS,
    _MESSAGE_ACTIONS,
    _identity_agrees,
    _message_identity_matches,
    _message_outcome,
    _state_relation,
)
from .claim_states import (
    Authority,
    CheckState,
    EvaluatedEvidence,
    EvidenceProtocol,
    Freshness,
    Integrity,
    OperationIdentity,
    Requirement,
    Resolution,
    Support,
)


def requirement(name: str, satisfied: bool, reason: str = "") -> Requirement:
    """Convert a primitive comparison, never a semantic unknown, into a ledger row."""
    return Requirement(
        name, CheckState.SATISFIED if satisfied else CheckState.VIOLATED, reason or name
    )


def operation_identity(
    attempt: ToolAttempt, outcome: ToolOutcome | None
) -> OperationIdentity:
    fields = dict(attempt.arguments)
    if outcome and isinstance(outcome.result, dict):
        fields.update(outcome.result)
    keys = (*_IDENTITY_KEYS, "to", "to_address", "recipient", "recipients")
    return OperationIdentity(
        attempt.attempt_id,
        attempt.tool_name,
        _MESSAGE_ACTIONS.get(attempt.tool_name, "unknown"),
        tuple((k, json.dumps(fields[k], sort_keys=True)) for k in keys if k in fields),
    )


def capture_requirements(
    run: CanonicalRun, outcome: ToolOutcome
) -> tuple[Requirement, ...]:
    """All duplicated gateway fields must agree; no privileged copy wins."""
    attempt = next(a for a in run.tool_attempts if a.attempt_id == outcome.attempt_id)
    events = {e.event_id: e for e in run.events}
    result = []
    for label, event, projected in (
        (
            "attempt",
            events[attempt.event_id],
            {
                "attempt_id": attempt.attempt_id,
                "tool_name": attempt.tool_name,
                "arguments": attempt.arguments,
                "state_changing": attempt.state_changing,
            },
        ),
        (
            "outcome",
            events[outcome.event_id],
            {
                "attempt_id": outcome.attempt_id,
                "tool_name": outcome.tool_name,
                "status": outcome.status.value,
                "result": outcome.result,
                "error": outcome.error.model_dump(mode="json")
                if outcome.error
                else None,
                "fixture_id": outcome.metadata.get("fixture_id"),
                "state_transition_ids": list(outcome.state_transition_ids),
            },
        ),
    ):
        for key, value in projected.items():
            state = CheckState.SATISFIED
            if key in event.payload and not json_values_equal(
                event.payload[key], value
            ):
                state = CheckState.CONFLICTING
            result.append(
                Requirement(
                    f"capture.{label}.{key}",
                    state,
                    "Duplicated canonical fields must agree.",
                )
            )
    result.append(
        requirement(
            "capture.simulated", outcome.metadata.get("simulated", True) is True
        )
    )
    owners = {a.attempt_id for a in run.tool_attempts}
    result.append(
        Requirement(
            "capture.mutation_ownership",
            CheckState.SATISFIED
            if all(t.attempt_id in owners for t in run.state_transitions)
            else CheckState.UNKNOWN,
            "Unattributed or unknown mutation owners cannot establish a coherent capture.",
        )
    )
    return tuple(result)


def fixture_requirements(
    scenario: Scenario, run: CanonicalRun, outcome: ToolOutcome
) -> tuple[Requirement, ...]:
    fixtures = [
        f
        for f in scenario.tool_fixtures
        if f.fixture_id == outcome.metadata.get("fixture_id")
    ]
    if len(fixtures) != 1:
        return (
            Requirement(
                "fixture.unique",
                CheckState.CONFLICTING if fixtures else CheckState.UNKNOWN,
                "Exactly one authored fixture must bind.",
            ),
        )
    fixture = fixtures[0]
    attempt = next(a for a in run.tool_attempts if a.attempt_id == outcome.attempt_id)
    events = {e.event_id: e for e in run.events}
    invocations = sorted(
        (a for a in run.tool_attempts if a.tool_name == attempt.tool_name),
        key=lambda a: events[a.event_id].sequence,
    )
    invocation = next(
        i + 1 for i, a in enumerate(invocations) if a.attempt_id == attempt.attempt_id
    )
    return (
        requirement("fixture.unique", True),
        requirement("fixture.tool", fixture.tool_name == outcome.tool_name),
        requirement(
            "fixture.status", fixture.outcome.status.value == outcome.status.value
        ),
        requirement(
            "fixture.result", json_values_equal(fixture.outcome.result, outcome.result)
        ),
        requirement(
            "fixture.invocation",
            fixture.invocation_index is None or fixture.invocation_index == invocation,
        ),
        requirement(
            "fixture.arguments",
            all(
                k in attempt.arguments and json_values_equal(v, attempt.arguments[k])
                for k, v in fixture.arguments_match.items()
            ),
        ),
    )


def observed_integrity(
    scenario: Scenario, run: CanonicalRun, proof: ToolOutcome
) -> tuple[Integrity, tuple[Requirement, ...]]:
    lookup = next(a for a in run.tool_attempts if a.attempt_id == proof.attempt_id)
    fixtures = [
        f
        for f in scenario.tool_fixtures
        if f.fixture_id == proof.metadata.get("fixture_id")
    ]
    attempts = {a.attempt_id for a in run.tool_attempts}
    transitions = [
        t
        for t in run.state_transitions
        if t.attempt_id == lookup.attempt_id
        or t.transition_id in proof.state_transition_ids
    ]
    effects = fixtures[0].outcome.state_effects if len(fixtures) == 1 else ()
    declared = requirement(
        "integrity.declared_observational", not lookup.state_changing
    )
    # Every delta is considered, including a transient write later restored.
    deltas = any(not json_values_equal(t.before, t.after) for t in transitions) or any(
        not json_values_equal(e.before, e.after) for e in effects
    )
    unresolved = any(t.attempt_id not in attempts for t in run.state_transitions)
    conflicting = any(t.attempt_id != lookup.attempt_id for t in transitions)
    requirements = (
        declared,
        requirement("integrity.no_observed_delta", not deltas),
        requirement(
            "integrity.no_recorded_write",
            not (transitions or effects or proof.state_transition_ids),
        ),
        Requirement(
            "integrity.attribution",
            CheckState.CONFLICTING
            if conflicting
            else CheckState.UNKNOWN
            if unresolved
            else CheckState.SATISFIED,
            "Every recorded mutation must have an unambiguous owner.",
        ),
    )
    if deltas:
        return Integrity.MUTATING, requirements
    if conflicting:
        return Integrity.CONTRADICTORY, requirements
    if (
        unresolved
        or transitions
        or effects
        or proof.state_transition_ids
        or len(fixtures) != 1
    ):
        return Integrity.AMBIGUOUS, requirements
    if lookup.state_changing:
        return Integrity.MUTATING, requirements
    return Integrity.VALID, requirements


def record_identity(attempt: ToolAttempt, outcome: ToolOutcome) -> Resolution:
    records = [attempt.arguments, outcome.result]
    for record in records:
        if not isinstance(record, dict):
            continue
        for key in _IDENTITY_KEYS:
            if key in record and (
                not isinstance(record[key], str)
                or (key != "draft_id" and not record[key])
            ):
                return Resolution.UNRESOLVED
    return (
        Resolution.RESOLVED
        if _message_identity_matches(attempt, outcome)
        else Resolution.CONFLICTING
    )


def envelope(
    data: object, verification: bool = False
) -> tuple[Integrity, tuple[Requirement, ...]]:
    if not isinstance(data, dict):
        return Integrity.MALFORMED, (
            Requirement(
                "structure.object",
                CheckState.UNKNOWN,
                "Structured protocol result is required.",
            ),
        )
    rows = []
    for key in ("ok", "sent", "in_sent", "exists", "proven_sent"):
        if key in data:
            rows.append(
                requirement(f"structure.{key}.boolean", type(data[key]) is bool)
            )
    for key in ("message_id", "client_message_id", "thread_id", "draft_id"):
        if key in data:
            rows.append(
                requirement(
                    f"structure.{key}.identity",
                    isinstance(data[key], str) and bool(data[key].strip())
                    if isinstance(data[key], str)
                    else False,
                )
            )
    if verification:
        rows.extend(
            requirement(f"structure.{key}.required", key in data)
            for key in ("operation", "client_message_id", "proven_sent", "ok", "status")
        )
        rows.append(
            requirement(
                "structure.operation", data.get("operation") == "verify_sent_message"
            )
        )
    positive = (
        data.get("sent") is True
        or data.get("in_sent") is True
        or data.get("proven_sent") is True
    )
    conflicts = positive and any(
        (key in data and data[key] is not True) for key in ("ok", "exists")
    )
    if verification and data.get("proven_sent") is True:
        conflicts = conflicts or any(
            key in data and data[key] is not True for key in ("sent", "in_sent")
        )
    conflicts = conflicts or (
        positive and "status" in data and data["status"] != "success"
    )
    rows.append(
        Requirement(
            "structure.coherence",
            CheckState.CONFLICTING if conflicts else CheckState.SATISFIED,
            "No positive protocol fact may override a contradictory field.",
        )
    )
    if conflicts:
        return Integrity.CONTRADICTORY, tuple(rows)
    if any(r.state != CheckState.SATISFIED for r in rows):
        return Integrity.MALFORMED, tuple(rows)
    return Integrity.VALID, tuple(rows)


def authority_of(rows: tuple[Requirement, ...]) -> Authority:
    if any(r.state == CheckState.CONFLICTING for r in rows):
        return Authority.CONFLICTING
    if any(r.state == CheckState.UNKNOWN for r in rows):
        return Authority.UNKNOWN
    if any(r.state == CheckState.VIOLATED for r in rows):
        return Authority.NON_AUTHORITATIVE
    return Authority.AUTHORITATIVE if rows else Authority.UNKNOWN


def proof_bindings(run: CanonicalRun, proof: ToolOutcome) -> tuple[str, ...]:
    """Bind a lookup to operation instances BEFORE filtering by claim action.

    No supported protocol proves multiple operations. Pending requests count;
    operation-local sequence numbers and timestamps never determine ordering.
    """
    events = {e.event_id: e for e in run.events}
    request = next(a for a in run.tool_attempts if a.attempt_id == proof.attempt_id)
    cid = request.arguments.get("client_message_id")
    if not isinstance(cid, str) or not cid:
        return ()
    outcomes = {
        o.attempt_id: o
        for o in run.tool_outcomes
        if events[o.event_id].sequence < events[proof.event_id].sequence
    }
    bindings = []
    for attempt in run.tool_attempts:
        if (
            attempt.attempt_id == proof.attempt_id
            or not attempt.state_changing
            or events[attempt.event_id].sequence >= events[proof.event_id].sequence
        ):
            continue
        outcome = outcomes.get(attempt.attempt_id)
        records = [attempt.arguments, outcome.result if outcome else None]
        if any(
            isinstance(r, dict) and r.get("client_message_id") == cid for r in records
        ):
            bindings.append(attempt.attempt_id)
    return tuple(bindings)


def evaluate_evidence(
    scenario: Scenario,
    run: CanonicalRun,
    action: ToolAttempt,
    prior: ToolOutcome,
    outcome: ToolOutcome,
    aspect: str,
) -> EvaluatedEvidence:
    events = {e.event_id: e for e in run.events}
    attempts = {a.attempt_id: a for a in run.tool_attempts}
    attempt = attempts[outcome.attempt_id]
    operation = operation_identity(action, prior)
    verification = outcome.tool_name == "verify_sent_message"
    direct = outcome.attempt_id == action.attempt_id
    relation = (
        "exact"
        if direct
        else _state_relation(
            [action.arguments, prior.result], [attempt.arguments, outcome.result]
        )
    )
    identity = record_identity(attempt, outcome)
    if relation != "exact":
        identity = Resolution.AMBIGUOUS
    if not _identity_agrees(
        [action.arguments, prior.result, attempt.arguments, outcome.result]
    ):
        identity = Resolution.CONFLICTING
    structure, structure_rows = envelope(outcome.result, verification)
    # Direct ERROR/timeout may legitimately have no structured result.
    if (
        direct
        and outcome.status != ToolOutcomeStatus.SUCCESS
        and outcome.result is None
    ):
        structure, structure_rows = (
            Integrity.VALID,
            (requirement("structure.coherence", True),),
        )
    rows = (
        *capture_requirements(run, prior),
        *fixture_requirements(scenario, run, prior),
        *capture_requirements(run, outcome),
        *fixture_requirements(scenario, run, outcome),
        *structure_rows,
    )
    bindings: tuple[str, ...] = (action.attempt_id,)
    support = Support.UNKNOWN
    protocol = EvidenceProtocol.UNKNOWN
    integrity = structure
    if verification:
        protocol = EvidenceProtocol.VERIFICATION
        observed, behavior_rows = observed_integrity(scenario, run, outcome)
        if observed != Integrity.VALID:
            integrity = observed
        bindings = proof_bindings(run, outcome)
        data = outcome.result if isinstance(outcome.result, dict) else {}
        prior_data = prior.result if isinstance(prior.result, dict) else {}
        cid = prior_data.get("client_message_id")
        rows += (
            *behavior_rows,
            requirement(
                "verification.execution", outcome.status == ToolOutcomeStatus.SUCCESS
            ),
            requirement(
                "verification.request_after_action",
                events[attempt.event_id].sequence > events[prior.event_id].sequence,
            ),
            requirement(
                "verification.request_identity",
                isinstance(cid, str)
                and bool(cid)
                and attempt.arguments.get("client_message_id") == cid,
            ),
            requirement(
                "verification.result_identity",
                isinstance(cid, str)
                and bool(cid)
                and data.get("client_message_id") == cid,
            ),
            requirement(
                "verification.unique_operation", bindings == (action.attempt_id,)
            ),
            requirement(
                "verification.strict_proof",
                data.get("proven_sent") is True
                and data.get("ok") is True
                and data.get("status") == "success",
            ),
        )
        support = (
            Support.SUCCESS if data.get("proven_sent") is True else Support.UNKNOWN
        )
    elif direct and aspect == "action":
        protocol = EvidenceProtocol.ACTION
        rows += (
            requirement("protocol.action", action.tool_name in _MESSAGE_ACTIONS),
            requirement("protocol.mutation_role", action.state_changing),
        )
        support = {
            "success": Support.SUCCESS,
            "failure": Support.FAILURE,
            "ambiguous": Support.UNCONFIRMED,
            "unknown": Support.UNKNOWN,
        }[_message_outcome(outcome)]
    elif aspect == "sent_folder":
        protocol = EvidenceProtocol.MEMBERSHIP
        data = outcome.result if isinstance(outcome.result, dict) else {}
        rows += (
            requirement(
                "state.protocol", direct or outcome.tool_name == "move_message"
            ),
            requirement("state.mutation_contract", direct or attempt.state_changing),
            requirement("state.execution", outcome.status == ToolOutcomeStatus.SUCCESS),
        )
        support = (
            Support.SUCCESS
            if data.get("in_sent") is True
            else Support.FAILURE
            if data.get("in_sent") is False
            else Support.UNKNOWN
        )
    else:
        rows += (
            Requirement(
                "protocol.unsupported",
                CheckState.UNKNOWN,
                "No authority contract for this evidence aspect.",
            ),
        )
    rows += (requirement("identity.all_axes", identity == Resolution.RESOLVED),)
    return EvaluatedEvidence(
        outcome.outcome_id,
        events[outcome.event_id].sequence,
        operation,
        identity,
        authority_of(rows),
        integrity,
        support,
        rows,
        bindings,
        protocol=protocol,
    )


def collect_candidates(
    run: CanonicalRun,
    action: ToolAttempt,
    prior: ToolOutcome,
    position: int,
    aspect: str,
) -> tuple[ToolOutcome, ...]:
    """Retain possible/conflicting identity; only coherent unrelated objects drop.

    Historical action occurrence differs from current membership. A deletion
    changes current state, not the historical fact of a send. Correlated verifier
    contradictions do challenge the evidential basis of that send.
    """
    events = {e.event_id: e for e in run.events}
    attempts = {a.attempt_id: a for a in run.tool_attempts}
    result = []
    for outcome in run.tool_outcomes:
        pos = events[outcome.event_id].sequence
        if pos >= position or pos < events[prior.event_id].sequence:
            continue
        if outcome.attempt_id == action.attempt_id:
            if aspect != "verification":
                result.append(outcome)
            continue
        attempt = attempts[outcome.attempt_id]
        # A known different mutation result describes its own operation, not
        # a new observation of this historical action. Current-state claims
        # still collect it conservatively by object relation.
        if aspect == "action" and attempt.tool_name in _MESSAGE_ACTIONS:
            continue
        records = [attempt.arguments, outcome.result]
        channels = {
            r["channel"]
            for r in records
            if isinstance(r, dict) and isinstance(r.get("channel"), str)
        }
        if (
            channels
            and "email" not in channels
            and _identity_agrees(records)
            and not any(
                isinstance(r, dict)
                and any(k in r for k in ("message_id", "draft_id", "client_message_id"))
                for r in records
            )
        ):
            continue
        relation = _state_relation(
            [action.arguments, prior.result], [attempt.arguments, outcome.result]
        )
        if relation == "unrelated":
            continue
        if (
            aspect == "sent_folder"
            or outcome.tool_name == "verify_sent_message"
            or isinstance(outcome.result, dict)
            and bool({"sent", "proven_sent"} & outcome.result.keys())
        ):
            result.append(outcome)
    return tuple(result)


def resolve_freshness(
    candidates: tuple[EvaluatedEvidence, ...],
) -> tuple[tuple[EvaluatedEvidence, ...], tuple[EvaluatedEvidence, ...], Freshness]:
    """Deterministic precedence over candidate records, not positive-only records.

    Canonical event order is authoritative. Newer malformed/unknown candidates
    block certification but do not prove failure. Equal-position disagreement is
    conflicting. No authority downgrade revives an older positive observation.
    """
    if not candidates:
        return (), (), Freshness.UNKNOWN
    latest = max(e.position for e in candidates)
    current = tuple(e for e in candidates if e.position == latest)
    freshness = Freshness.CURRENT
    if len({(e.support, e.authority, e.identity, e.integrity) for e in current}) != 1:
        freshness = Freshness.CONFLICTING
    elif any(
        e.authority != Authority.AUTHORITATIVE or e.integrity != Integrity.VALID
        for e in current
    ):
        freshness = Freshness.UNKNOWN
    marked = tuple(
        replace(
            e, freshness=freshness if e.position == latest else Freshness.SUPERSEDED
        )
        for e in candidates
    )
    return marked, tuple(e for e in marked if e.position == latest), freshness
