"""Bounded email protocol codecs; no verdicts are assigned here."""

from __future__ import annotations
import re
from agentcheck.domain import (
    CanonicalEventType,
    CanonicalRun,
    Scenario,
    ToolAttempt,
    ToolOutcome,
    ToolOutcomeStatus,
)
from agentcheck.json_values import json_values_equal
from .confirmation import tool_evidence_is_consistent


def _consistent(scenario: Scenario, run: CanonicalRun) -> bool:
    if not tool_evidence_is_consistent(scenario, run):
        return False
    events = {e.event_id: e for e in run.events}
    final = [e for e in run.events if e.event_type == CanonicalEventType.FINAL_OUTPUT]
    if final and (
        len(final) != 1
        or final[0] != run.events[-1]
        or final[0].payload.get("text") != run.final_output
    ):
        return False
    if any(
        not isinstance(e.payload.get("text"), str)
        for e in run.events
        if e.event_type == CanonicalEventType.ASSISTANT_OUTPUT
    ):
        return False
    attempts = {a.attempt_id: a for a in run.tool_attempts}
    outcomes = run.tool_outcomes
    if (
        len({o.attempt_id for o in outcomes}) != len(outcomes)
        or len({o.event_id for o in outcomes}) != len(outcomes)
        or {o.event_id for o in outcomes}
        != {
            e.event_id
            for e in run.events
            if e.event_type == CanonicalEventType.TOOL_RESULT
        }
    ):
        return False
    for o in outcomes:
        a = attempts.get(o.attempt_id)
        e = events.get(o.event_id)
        if (
            a is None
            or e is None
            or o.tool_name != a.tool_name
            or e.sequence <= events[a.event_id].sequence
        ):
            return False
        for key, value in (
            ("attempt_id", o.attempt_id),
            ("tool_name", o.tool_name),
            ("status", o.status.value),
            ("result", o.result),
            ("error", o.error.model_dump(mode="json") if o.error else None),
        ):
            if key in e.payload and not json_values_equal(e.payload[key], value):
                return False
    return True


def _fixture_bound(scenario: Scenario, run: CanonicalRun, outcome: ToolOutcome) -> bool:
    """Structured verification is authority only when bound to the authored fixture."""
    fixtures = [
        f
        for f in scenario.tool_fixtures
        if f.fixture_id == outcome.metadata.get("fixture_id")
    ]
    if len(fixtures) != 1:
        return False
    f = fixtures[0]
    attempt = next(a for a in run.tool_attempts if a.attempt_id == outcome.attempt_id)
    invocation = [
        a.attempt_id for a in run.tool_attempts if a.tool_name == attempt.tool_name
    ].index(attempt.attempt_id) + 1
    return (
        f.tool_name == outcome.tool_name
        and f.outcome.status.value == outcome.status.value
        and json_values_equal(f.outcome.result, outcome.result)
        and (f.invocation_index is None or f.invocation_index == invocation)
        and all(
            k in attempt.arguments and json_values_equal(attempt.arguments[k], v)
            for k, v in f.arguments_match.items()
        )
    )


_MESSAGE_ACTIONS = {
    "send_email": "send",
    "send_mail": "send",
    "reply_to_thread": "reply",
    "reply_email": "reply",
}


def _recipients_agree(records: list[object]) -> bool:
    values = []
    for record in records:
        if isinstance(record, dict):
            for key in ("to", "to_address", "recipient", "recipients"):
                if key in record:
                    value = record[key]
                    value = [value] if isinstance(value, str) else value
                    if (
                        not isinstance(value, list)
                        or not value
                        or not all(isinstance(item, str) and item for item in value)
                    ):
                        return False
                    values.append(value)
    return not values or all(
        json_values_equal(values[0], value) for value in values[1:]
    )


_IDENTITY_KEYS = (
    "subject",
    "body",
    "thread_id",
    "message_id",
    "draft_id",
    "client_message_id",
    "channel",
)


def _identity_agrees(records: list[object]) -> bool:
    if not _recipients_agree(records):
        return False
    for key in _IDENTITY_KEYS:
        values = [r[key] for r in records if isinstance(r, dict) and key in r]
        # Some supported tool contracts use an empty draft ID for a new send.
        # It cannot bind a claim reference, and still conflicts with a nonempty ID.
        if any(not isinstance(v, str) for v in values):
            return False
        if values and any(not json_values_equal(values[0], v) for v in values[1:]):
            return False
    return True


def _message_identity_matches(attempt: ToolAttempt, outcome: ToolOutcome) -> bool:
    data = outcome.result
    if not isinstance(data, dict):
        return True  # It cannot establish a positive outcome below.
    if "operation" in data and data["operation"] != attempt.tool_name:
        return False
    if any(
        "channel" in r and r["channel"] != "email" for r in (attempt.arguments, data)
    ):
        return False
    return _identity_agrees([attempt.arguments, data])


def _message_reference(record: object, reference: str) -> bool:
    if not isinstance(record, dict):
        return False
    keys: tuple[str, ...]
    if "@" in reference:
        keys = ("to", "to_address", "recipient", "recipients")
    elif re.match(r"(?:msg|message)[-_]", reference, re.I):
        keys = ("message_id",)
    elif re.match(r"(?:thr|thread)[-_]", reference, re.I):
        keys = ("thread_id",)
    elif re.match(r"draft[-_]", reference, re.I):
        keys = ("draft_id",)
    else:
        return False
    # IDs and recipients retain source spelling. No case folding, role aliasing
    # between IDs, arbitrary recursive body scan, or requested/result override.
    return any(
        record.get(key) == reference
        or isinstance(record.get(key), list)
        and reference in record[key]
        for key in keys
    )


def _message_outcome(outcome: ToolOutcome) -> str:
    if outcome.status in {ToolOutcomeStatus.ERROR, ToolOutcomeStatus.BLOCKED}:
        data = outcome.result
        # A failed execution cannot settle contradictory or malformed action
        # evidence. Only an absent flag or strict false agrees with non-send.
        if isinstance(data, dict) and "sent" in data and data["sent"] is not False:
            return "unknown"
        return "failure" if not outcome.state_transition_ids else "ambiguous"
    if outcome.status != ToolOutcomeStatus.SUCCESS:
        return "ambiguous"
    data = outcome.result
    if not isinstance(data, dict):
        return "unknown"
    if "ok" in data and type(data["ok"]) is not bool:
        return "unknown"
    sent = data.get("sent")
    if sent is True:
        if ("ok" in data and data["ok"] is not True) or (
            "status" in data and data["status"] != "success"
        ):
            return "unknown"
        return "success"
    if sent is False:
        if "status" in data and data["status"] not in (
            "success",
            "failed",
            "failure",
            "error",
        ):
            return "unknown"
        return "failure"
    # Absent, null, strings and integers do not encode a boolean outcome.
    return "unknown"


_OBJECT_KEYS = ("message_id", "draft_id", "client_message_id")


def _state_relation(identity: list[object], records: list[object]) -> str:
    """Return exact / possible / unrelated before considering a tool's protocol.

    Conflicting axes with another shared ID are ambiguous, never discarded.
    With no shared object axis we cannot safely exclude a later observation.
    A different, well-formed object ID with no shared object ID is unrelated.
    """
    matched = False
    different = False
    for key in _OBJECT_KEYS:
        left = {
            r[key]
            for r in identity
            if isinstance(r, dict) and isinstance(r.get(key), str) and r[key]
        }
        right = {
            r[key]
            for r in records
            if isinstance(r, dict) and isinstance(r.get(key), str) and r[key]
        }
        matched = matched or bool(left & right)
        different = different or bool(left and right and not left & right)
    if matched:
        return "exact" if _identity_agrees([*identity, *records]) else "possible"
    if different and _identity_agrees(records):
        return "unrelated"
    return "possible"


def _state_envelope_coherent(data: object) -> bool:
    """A positive state flag cannot override a contradictory result envelope."""
    return (
        isinstance(data, dict)
        and ("ok" not in data or data["ok"] is True)
        and ("status" not in data or data["status"] == "success")
        and ("exists" not in data or type(data["exists"]) is bool)
    )
