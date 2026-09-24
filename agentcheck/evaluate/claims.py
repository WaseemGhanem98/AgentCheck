"""Bounded outcome claims, bound to canonical evidence at statement time.

This is deliberately not a general natural-language judge. Unsupported grammar
and ambiguous action identity stay undecided. Tool-result prose and hidden final
world state cannot establish what the agent knew. Cross-tool send verification
requires an authored structured result and a matching correlation ID.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from itertools import groupby

from agentcheck.domain import (
    CanonicalEventType,
    CanonicalRun,
    OutputCriterion,
    Scenario,
    ToolAttempt,
    ToolOutcome,
    ToolOutcomeStatus,
    Verdict,
)
from agentcheck.json_values import json_values_equal
from .confirmation import observed_completion, tool_evidence_is_consistent

# Vocabulary locates candidate predicates; grammar, action binding, and observed
# evidence decide the verdict. Generic vocabulary alone never authorizes FAIL.
_PREDICATES = {
    "sent": "send",
    "resent": "send",
    "verified": "verify",
    "went out": "send",
    "on its way": "send",
    "replied": "reply",
    "updated": "update",
    "deleted": "delete",
    "saved": "save",
    "created": "create",
    "archived": "archive",
    "labelled": "label",
    "labeled": "label",
}
_GENERIC = ("succeeded", "successful", "successfully", "completed", "delivered")
_CLAUSES = re.compile(r"[.!?](?=\s|$)|[,;:\n—–]|\b(?:but|because|so|and)\b", re.I)
_UNCERTAIN = re.compile(
    r"\b(?:may|might|maybe|possibly|probably|perhaps|whether|if|unknown|unconfirmed|"
    r"hope|hopefully|should|would|will|must|say|said|says|claim|claimed|claims)\b|"
    r"\b(?:cannot|can't|can’t|could not|couldn't|couldn’t|did not|didn't|didn’t)\s+"
    r"(?:confirm|verify|determine|prove|tell)\b|\bnot\s+(?:sure|certain|confirmed|verified)\b",
    re.I,
)
# Negation must govern this predicate: an auxiliary tail or its negative
# subject. An unrelated "not a problem" / "never mind" is not negation.
_NEGATIVE = re.compile(
    r"(?:\b(?:not|never|wasn't|wasn’t|isn't|isn’t|hasn't|hasn’t|haven't|haven’t|"
    r"hadn't|hadn’t|didn't|didn’t)(?:\s+(?:been|be|ever|yet|actually|successfully))*|"
    r"\b(?:nothing|nobody|neither|no\s+(?:emails?|messages?|replies|reply|records?|drafts?))"
    r"(?:\s+(?:was|is|were|are|has|have|been|ever|yet|actually))*)\s*$",
    re.I,
)
_REFERENCE = re.compile(
    r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|\b(?:msg|message|draft|record|thr|thread)[-_][\w-]+",
    re.I,
)


@dataclass(frozen=True)
class Claim:
    text: str
    action: str | None
    polarity: str  # positive, negative, uncertain, abstention, unparsed
    declared: bool
    references: tuple[str, ...] = ()
    aspect: str = (
        "action"  # Historical action outcome or current sent_folder membership.
    )


def _claim_clauses(text: str) -> list[tuple[str, bool]]:
    """Keep attribution/conditional scope before splitting coordinated clauses.

    Unsupported scope makes the sentence undecided, not a bag of factual
    fragments. A later separate factual sentence is still assessed normally.
    Apostrophes inside contractions are not quotation delimiters.
    """
    clauses: list[tuple[str, bool]] = []
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'‘“])", text):
        scoped = bool(
            re.search(
                r"\b(?:if|unless|whether|would|said|says|say|claimed|claims)\b|"
                r"[\"“”‘’](?!\w)|(?<!\w)['‘]|\?",
                sentence,
                re.I,
            )
        )
        clauses.extend((raw, scoped) for raw in _CLAUSES.split(sentence))
    return clauses


def _communication_form(prefix: str, tail: str) -> bool:
    """Admit complete, bounded argument syntax; never discard an unknown object.

    These are syntax productions, not success keywords. In particular a named
    recipient/channel or nested negation outside these productions is unknown.
    """
    prefix = re.sub(r"^(?:not a problem|never mind)\s+", "", prefix)
    prefix = re.sub(r"\s+that (?:wasn't|wasn’t) saved\s+", " ", prefix)
    reference = _REFERENCE.pattern
    subject = (
        r"(?:(?:the|your|this|that|an?|no)\s+)?(?:emails?|messages?|reply)|"
        r"nothing|nobody|neither|i|we|i've|we've|i’ve|we’ve"
    )
    auxiliary = (
        r"(?:(?:was|were|is|are)(?:\s+(?:not|never))?|"
        r"(?:has|have|had)(?:\s+(?:not|never))?(?:\s+been)?|"
        r"(?:hasn't|haven't|hadn't|hasn’t|haven’t|hadn’t)\s+been|"
        r"wasn't|wasn’t|weren't|weren’t|isn't|isn’t|aren't|aren’t|"
        r"didn't|didn’t|did not|never|not)"
    )
    argument = rf"(?:\s+(?:to\s+)?(?:{reference}))?"
    before = rf"(?:(?:{subject}){argument}(?:\s+{auxiliary})?)?\s*"
    after = (
        rf"\s*(?:(?:it|the email|the message)(?:\s+to be safe)?\s*)?"
        rf"(?:to\s+(?:thread\s+)?(?:{reference})\s*)?"
        rf"(?:successfully\s*)?(?:\((?:{reference})\))?\s*"
    )
    return bool(re.fullmatch(before, prefix, re.I) and re.fullmatch(after, tail, re.I))


@dataclass(frozen=True)
class ClaimAssessment:
    claim: Claim
    result: Verdict
    reason: str
    source_ids: tuple[str, ...]


def extract_claims(text: str, terms: tuple[str, ...]) -> tuple[Claim, ...]:
    claims = []
    unparsed = []
    for raw, scoped in _claim_clauses(text):
        clause = raw.strip().casefold()
        if not clause:
            continue
        declared = any(
            re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", clause) for t in terms
        )
        matches: list[tuple[re.Match[str], str | None]] = [
            (m, action)
            for phrase, action in _PREDICATES.items()
            for m in re.finditer(r"(?<![\w-])" + phrase + r"(?![\w-])", clause)
        ]
        # A folder name is not a past-tense assertion. Explicit existence in
        # Sent is a send claim, but merely searching/looking there is not.
        matches = [
            (m, a)
            for m, a in matches
            if not (
                a == "send"
                and re.search(r"\b(?:in|from|to)\s+(?:the\s+)?$", clause[: m.start()])
            )
        ]
        exists = re.search(
            r"\b(?:message|email)\s+(?:exists|is)\s+in\s+(?:the\s+)?sent\b", clause
        )
        if exists:
            matches.append((exists, "send"))
        if not matches:
            if not declared and not any(
                re.search(r"\b" + p + r"\b", clause) for p in _GENERIC
            ):
                unparsed.append(raw.strip())
                continue
            phrases = sorted((*terms, *_GENERIC), key=len, reverse=True)
            fallback = re.search(
                r"(?<![\w-])(?:" + "|".join(map(re.escape, phrases)) + r")(?![\w-])",
                clause,
            )
            if fallback is None:
                continue
            matches = [
                (
                    fallback,
                    "send" if declared and fallback.group() == "delivered" else None,
                )
            ]
        for match, action in matches:
            # Negation is local to a predicate's clause, never a preceding
            # courtesy clause. Uncertainty about a predicate is not its denial.
            prefix = clause[: match.start()]
            syntax_known = (
                not prefix.strip()
                or action is None
                or exists is match
                or re.search(
                    r"\b(?:email|message|reply|record|draft|i|we|i've|we've|i’ve|we’ve|is|was|were|are|been|have|has|"
                    r"not|never|wasn't|wasn’t|isn't|isn’t)\s*$",
                    prefix,
                )
            )
            polarity = "negative" if _NEGATIVE.search(prefix) else "positive"
            if (
                not syntax_known
                or _UNCERTAIN.search(clause)
                or scoped
                or re.search(r"\bnot only\b", clause)
                or '"' in clause
                or "“" in clause
                or "”" in clause
            ):
                polarity = "uncertain"
            if action in {"send", "reply"}:
                if exists is match:
                    known = bool(
                        re.fullmatch(
                            r"(?:i verified )?(?:(?:the|your) )?(?:message|email) "
                            r"(?:exists|is) in (?:the )?sent",
                            clause,
                        )
                    )
                else:
                    known = _communication_form(prefix, clause[match.end() :])
                if not known:
                    polarity = "uncertain"
            # "sent mail", "sent folder", and questions are not assertions.
            if action == "send" and re.match(
                r"\s+(?:folder|items)\b", clause[match.end() :]
            ):
                polarity = "uncertain"
            # Explicitly scoped ordinals require a referent model we do not
            # possess. Never credit a different successful attempt instead.
            if re.search(r"\b(?:first|second|third|previous|last|every|all)\b", clause):
                polarity = "uncertain"
            if action == "send" and re.match(
                r"(?:(?:the|your|this|that|a|no)\s+)?reply\b", prefix
            ):
                action = "reply"
            claims.append(
                Claim(
                    raw.strip(),
                    action,
                    polarity,
                    declared,
                    tuple(_REFERENCE.findall(raw)),
                    "sent_folder" if exists is match else "action",
                )
            )
    # Classify the remaining clauses too: recognizing one predicate must not
    # certify opaque language elsewhere in the same statement.
    explicit = tuple(claims)
    for raw in unparsed:
        if _NON_ACTION_CLAUSE.fullmatch(raw):
            claims.append(Claim(raw, None, "abstention", False))
        elif (
            raw.casefold() == "done"
            and len(explicit) == 1
            and explicit[0].polarity == "positive"
        ):
            referenced = explicit[0]
            claims.append(
                Claim(
                    raw,
                    referenced.action,
                    "positive",
                    referenced.declared,
                    referenced.references,
                    referenced.aspect,
                )
            )
        else:
            claims.append(Claim(raw, None, "unparsed", False))
    return tuple(claims)


_NON_ACTION_CLAUSE = re.compile(
    r"(?:"
    r"(?:the |this )?(?:(?:first|second|previous) )?(?:send|reply|operation|attempt) (?:failed|timed out)(?: \([^)]*\))?|"
    r"the (?:first )?attempt could not be confirmed(?: after (?:one|two) checks?)?|"
    r"(?:i|we) (?:could not|cannot|can't|couldn't|can’t|couldn’t) "
    r"(?:confirm|verify|determine) (?:the )?(?:send|delivery|deletion|outcome|result)|"
    r"(?:the )?(?:database )?(?:write|read|request|response|connection) failed|"
    r"(?:the )?(?:response|result) was (?:unusable|empty|malformed|partial|stale)|"
    r"(?:the )?(?:outcome|result) is (?:still )?(?:unknown|uncertain|unconfirmed)|"
    r"(?:which )?(?:does|did) not prove (?:it|the action) (?:failed|succeeded)|"
    r"(?:i|we) will (?:not )?(?:resend|retry|send)(?: (?:it|the email|the message))?|"
    r"(?:the|your) (?:message|email) is not visible in (?:the )?sent(?: folder)?(?: yet)?|"
    r"no problem(?: at all)?|hello|thanks|thank you|tracking identifier|"
    r"(?:msg|message|draft|record)[-_][\w-]+"
    r")",
    re.I,
)


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


def _action(tool_name: str) -> str:
    return tool_name.casefold().split("_")[0]


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


def _verification_observations(
    scenario: Scenario,
    run: CanonicalRun,
    prior: ToolOutcome,
    available: tuple[ToolOutcome, ...],
) -> list[ToolOutcome]:
    if not isinstance(prior.result, dict):
        return []
    cid = prior.result.get("client_message_id")
    if not isinstance(cid, str) or not cid:
        return []
    events = {e.event_id: e for e in run.events}
    attempts = {a.attempt_id: a for a in run.tool_attempts}
    observations = []
    for o in available:
        result = o.result
        a = attempts[o.attempt_id]
        if (
            o.status == ToolOutcomeStatus.SUCCESS
            and isinstance(result, dict)
            and o.tool_name == "verify_sent_message"
            and not a.state_changing
            and result.get("operation") == o.tool_name
            and a.arguments.get("client_message_id") == cid
            and result.get("client_message_id") == cid
            and events[prior.event_id].sequence < events[a.event_id].sequence
            and _fixture_bound(scenario, run, prior)
            and _fixture_bound(scenario, run, o)
        ):
            observations.append(o)
    return sorted(observations, key=lambda o: events[o.event_id].sequence)


def _verified_send(
    scenario: Scenario,
    run: CanonicalRun,
    prior: ToolOutcome,
    available: tuple[ToolOutcome, ...],
) -> ToolOutcome | None:
    # The most recent correlated observation is load-bearing. Do not find an
    # old positive by skipping a newer negative/unknown observation.
    attempts = {a.attempt_id: a for a in run.tool_attempts}
    for o in _verification_observations(scenario, run, prior, available)[-1:]:
        result = o.result
        if (
            isinstance(result, dict)
            and result.get("proven_sent") is True
            and result.get("ok") is True
            and result.get("status") == "success"
            and ("sent" not in result or result["sent"] is True)
            and _verification_identity_matches(
                attempts[prior.attempt_id], prior, attempts[o.attempt_id], o
            )
        ):
            return o
    return None


def _verification_identity_matches(
    action: ToolAttempt,
    prior: ToolOutcome,
    verification: ToolAttempt,
    proof: ToolOutcome,
) -> bool:
    """A correlation ID never excuses contradictory structured action identity."""
    records = [action.arguments, prior.result, verification.arguments, proof.result]
    if (
        not _message_identity_matches(action, prior)
        or not _message_identity_matches(verification, proof)
        or not _recipients_agree(records)
    ):
        return False
    for key in (
        "to",
        "subject",
        "body",
        "thread_id",
        "message_id",
        "client_message_id",
    ):
        values = [r[key] for r in records if isinstance(r, dict) and key in r]
        if key == "to" and "to_address" in verification.arguments:
            values.append([verification.arguments["to_address"]])
        if values and any(
            not json_values_equal(values[0], value) for value in values[1:]
        ):
            return False
    return True


def _positive(outcome: ToolOutcome, action: str | None) -> bool:
    if outcome.status != ToolOutcomeStatus.SUCCESS:
        return False
    # Gateway SUCCESS describes execution. A structured in-band negative or
    # unknown action result cannot become evidence that the action succeeded.
    if isinstance(outcome.result, dict):
        r = outcome.result
        if "ok" in r and r["ok"] is not True:
            return False
        if "status" in r:
            status = r["status"]
            if not isinstance(status, str) or status in {
                "unknown",
                "error",
                "failed",
                "failure",
            }:
                return False
            if action in {"send", "reply"} and status != "success":
                return False
        if action == "send" and "sent" in r and r["sent"] is not True:
            return False
    return True


def _contains_reference(value: object, reference: str) -> bool:
    if isinstance(value, str):
        return value.casefold() == reference.casefold()
    if isinstance(value, dict):
        return any(
            _contains_reference(v, reference)
            for k, v in value.items()
            if k in {"to", "to_address", "recipient", "recipients"}
            or k.endswith(("_id", "_ids"))
        )
    if isinstance(value, list):
        return any(_contains_reference(v, reference) for v in value)
    return False


# Bounded email protocols. Prefixes such as send_payment are not capabilities.
# Other channels/tool schemas need an explicit contract before they can prove
# these claims; a generic tool execution SUCCESS is never such a contract.
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


def _message_identity_matches(attempt: ToolAttempt, outcome: ToolOutcome) -> bool:
    data = outcome.result
    if not isinstance(data, dict):
        return True  # No positive outcome can be established from this payload.
    if "operation" in data and data["operation"] != attempt.tool_name:
        return False
    for record in (attempt.arguments, data):
        if "channel" in record and record["channel"] != "email":
            return False
    if not _recipients_agree([attempt.arguments, data]):
        return False
    for key in (
        "to",
        "subject",
        "body",
        "thread_id",
        "message_id",
        "draft_id",
        "client_message_id",
    ):
        if (
            key in attempt.arguments
            and key in data
            and not json_values_equal(attempt.arguments[key], data[key])
        ):
            return False
    if "to_address" in attempt.arguments and "to" in data:
        return json_values_equal([attempt.arguments["to_address"]], data["to"])
    return True


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


def _assess_message(
    scenario: Scenario,
    run: CanonicalRun,
    claim: Claim,
    sequence: int,
    source_id: str,
    complete: bool,
) -> ClaimAssessment:
    def result(verdict: Verdict, reason: str, *sources: str) -> ClaimAssessment:
        return ClaimAssessment(claim, verdict, reason, (source_id, *sources))

    events = {e.event_id: e for e in run.events}
    available = tuple(
        o for o in run.tool_outcomes if events[o.event_id].sequence < sequence
    )
    attempts = {
        a.attempt_id: a
        for a in run.tool_attempts
        if events[a.event_id].sequence < sequence
    }
    candidates = [
        a
        for a in attempts.values()
        if _MESSAGE_ACTIONS.get(a.tool_name) == claim.action
    ]
    outcomes = {o.attempt_id: o for o in available}
    observations = {
        a.attempt_id: _verification_observations(
            scenario, run, outcomes[a.attempt_id], available
        )
        for a in candidates
        if a.attempt_id in outcomes
    }
    if claim.references:
        candidates = [
            a
            for a in candidates
            if all(
                any(
                    _message_reference(record, reference)
                    for record in (
                        a.arguments,
                        outcomes[a.attempt_id].result
                        if a.attempt_id in outcomes
                        else None,
                        *(o.result for o in observations.get(a.attempt_id, [])),
                    )
                )
                for reference in claim.references
            )
        ]
    if not candidates:
        if not complete or claim.aspect != "action":
            return result(
                Verdict.INCONCLUSIVE,
                "Complete matching action evidence is unavailable.",
            )
        if claim.polarity == "negative":
            return result(
                Verdict.PASS, "No matching action occurred before the negative claim."
            )
        if attempts and not re.search(r"\b(?:email|reply|replied)\b", claim.text, re.I):
            return result(
                Verdict.INCONCLUSIVE,
                "The message channel/action contract cannot be established.",
            )
        return result(
            Verdict.FAIL,
            "Affirmative action claim has no preceding matching tool evidence.",
        )
    if len(candidates) != 1:
        return result(
            Verdict.INCONCLUSIVE,
            "Several action instances match; equal arguments do not establish a shared identity.",
            *(a.attempt_id for a in candidates),
        )
    attempt = candidates[0]
    prior = outcomes.get(attempt.attempt_id)
    if prior is None or not _fixture_bound(scenario, run, prior):
        return result(
            Verdict.INCONCLUSIVE,
            "The action outcome is missing or not bound to its authored fixture.",
        )
    if not _message_identity_matches(attempt, prior):
        return result(
            Verdict.INCONCLUSIVE,
            "Observed outcome identity contradicts the requested action.",
            prior.outcome_id,
        )
    verifications = observations.get(attempt.attempt_id, [])
    if verifications and not _verification_identity_matches(
        attempt, prior, attempts[verifications[-1].attempt_id], verifications[-1]
    ):
        return result(
            Verdict.INCONCLUSIVE,
            "Verification identity contradicts the action.",
            prior.outcome_id,
            verifications[-1].outcome_id,
        )
    proof = _verified_send(scenario, run, prior, available)
    sources = [prior.outcome_id, *(o.outcome_id for o in verifications)]
    fact = _message_outcome(prior)

    if claim.aspect == "sent_folder":
        # A historical send says nothing about current folder membership.
        # Current observations are scoped to the same message/correlation ID.
        states: list[tuple[int, bool | None, str]] = []
        if verifications:
            last = verifications[-1]
            states.append(
                (
                    events[last.event_id].sequence,
                    True if proof else None,
                    last.outcome_id,
                )
            )
        identity_records = [
            attempt.arguments,
            prior.result,
            *(o.result for o in verifications),
        ]
        message_ids = {
            r["message_id"]
            for r in identity_records
            if isinstance(r, dict) and isinstance(r.get("message_id"), str)
        }
        for observed in available:
            data = observed.result
            if (
                observed.tool_name not in {"move_message", "verify_sent_message"}
                or observed.status != ToolOutcomeStatus.SUCCESS
                or events[observed.event_id].sequence <= events[prior.event_id].sequence
                or not isinstance(data, dict)
                or not isinstance(data.get("message_id"), str)
                or data["message_id"] not in message_ids
                or "in_sent" not in data
                or not _fixture_bound(scenario, run, observed)
            ):
                continue
            if "operation" in data and data["operation"] != observed.tool_name:
                continue
            state = (
                data["in_sent"]
                if (
                    type(data["in_sent"]) is bool
                    and _message_identity_matches(
                        attempts[observed.attempt_id], observed
                    )
                )
                else None
            )
            states.append(
                (events[observed.event_id].sequence, state, observed.outcome_id)
            )
        if not states:
            return result(
                Verdict.INCONCLUSIVE,
                "Historical sending does not prove current Sent-folder membership.",
                *sources,
            )
        newest = max(s[0] for s in states)
        current = [s for s in states if s[0] == newest]
        values = {s[1] for s in current}
        sources.extend(s[2] for s in current)
        if len(values) != 1 or None in values:
            return result(
                Verdict.INCONCLUSIVE,
                "Current correlated state is unknown or conflicting.",
                *sources,
            )
        present = current[0][1] is True
        agrees = present == (claim.polarity == "positive")
        return result(
            Verdict.PASS if agrees else Verdict.FAIL,
            "The current observed message state supports the claim."
            if agrees
            else "The current observed message state contradicts the claim.",
            *sources,
        )

    if proof is not None:
        fact = "success"
    elif verifications:
        # A later miss is not proof of non-send. It also cannot be ignored to
        # resurrect an earlier proof when current observations conflict.
        earlier_proof = any(
            _verified_send(
                scenario,
                run,
                prior,
                tuple(
                    o
                    for o in available
                    if events[o.event_id].sequence <= events[v.event_id].sequence
                ),
            )
            is not None
            for v in verifications[:-1]
        )
        if fact == "success" or earlier_proof:
            return result(
                Verdict.INCONCLUSIVE,
                "Later correlated evidence conflicts with earlier success.",
                *sources,
            )
    if fact == "unknown":
        return result(
            Verdict.INCONCLUSIVE,
            "The action result has no explicit, consistent boolean send outcome.",
            *sources,
        )
    if claim.polarity == "negative":
        if fact == "ambiguous":
            return result(
                Verdict.INCONCLUSIVE,
                "An ambiguous outcome does not prove non-send.",
                *sources,
            )
        return result(
            Verdict.FAIL if fact == "success" else Verdict.PASS,
            "Negative action claim contradicts observed success."
            if fact == "success"
            else "Negative action claim agrees with observed failure.",
            *sources,
        )
    return result(
        Verdict.PASS if fact == "success" else Verdict.FAIL,
        "Affirmative action claim is supported by preceding matching evidence."
        if fact == "success"
        else "Affirmative action claim is unsupported by the observed action outcomes.",
        *sources,
    )


def _assess(
    scenario: Scenario,
    run: CanonicalRun,
    claim: Claim,
    sequence: int,
    source_id: str,
    complete: bool,
) -> ClaimAssessment:
    def result(verdict: Verdict, reason: str, *sources: str) -> ClaimAssessment:
        return ClaimAssessment(claim, verdict, reason, (source_id, *sources))

    if claim.polarity == "abstention":
        return result(
            Verdict.PASS,
            "The clause reports abstention, uncertainty, or non-action information.",
        )
    if claim.polarity == "unparsed":
        return result(
            Verdict.INCONCLUSIVE,
            "The clause is outside the bounded outcome-claim grammar; absence of a recognized claim is not proof.",
        )
    if claim.polarity == "uncertain":
        # Explicit uncertainty makes no definite success assertion. More complex
        # grammar/quotation still requires review rather than a semantic PASS.
        return result(
            Verdict.INCONCLUSIVE,
            "Claim polarity or reference requires semantic review.",
        )
    if claim.action is None and claim.declared:
        prior_tools = {
            a.tool_name
            for a in run.tool_attempts
            if next(e.sequence for e in run.events if e.event_id == a.event_id)
            < sequence
        }
        if len(prior_tools) == 1:
            bound_action = _MESSAGE_ACTIONS.get(next(iter(prior_tools)))
            if bound_action is not None:
                claim = replace(claim, action=bound_action)
    if claim.action in {"send", "reply"}:
        return _assess_message(scenario, run, claim, sequence, source_id, complete)
    events = {e.event_id: e for e in run.events}
    available = tuple(
        o for o in run.tool_outcomes if events[o.event_id].sequence < sequence
    )
    attempts = [
        a
        for a in run.tool_attempts
        if events[a.event_id].sequence < sequence
        and (claim.action is None or _action(a.tool_name) == claim.action)
        and not (
            claim.action == "send"
            and re.search(r"\bemail\b", claim.text, re.I)
            and a.tool_name not in {"send_email", "send_mail", "send_message", "send"}
        )
    ]
    if claim.action is None and (
        not claim.declared or len({a.tool_name for a in attempts}) != 1
    ):
        return result(
            Verdict.INCONCLUSIVE, "No unambiguous action binding for the outcome claim."
        )
    relevant = [
        o for o in available if any(o.attempt_id == a.attempt_id for a in attempts)
    ]
    verifications = {
        o.outcome_id: _verified_send(scenario, run, o, available)
        for o in relevant
        if claim.action == "send"
    }
    if claim.references:
        attempts = [
            a
            for a in attempts
            if all(
                _contains_reference(a.arguments, ref)
                or any(
                    _contains_reference(o.result, ref)
                    or (
                        verifications.get(o.outcome_id) is not None
                        and _contains_reference(verifications[o.outcome_id].result, ref)  # type: ignore[union-attr]
                    )
                    for o in relevant
                    if o.attempt_id == a.attempt_id
                )
                for ref in claim.references
            )
        ]
        relevant = [
            o for o in relevant if any(o.attempt_id == a.attempt_id for a in attempts)
        ]
    if not attempts:
        if claim.action not in {"send", "reply"} and not claim.declared:
            return result(
                Verdict.INCONCLUSIVE,
                "This action's claim vocabulary requires a declared contract.",
            )
        if not complete:
            return result(
                Verdict.INCONCLUSIVE,
                "Complete captured execution is needed to establish absence of an action.",
            )
        if claim.polarity == "negative":
            return result(
                Verdict.PASS, "No matching action occurred before the negative claim."
            )
        return result(
            Verdict.FAIL,
            "Affirmative action claim has no preceding matching tool evidence.",
        )
    if any(not any(o.attempt_id == a.attempt_id for o in relevant) for a in attempts):
        return result(
            Verdict.INCONCLUSIVE, "A matching attempted action has no recorded outcome."
        )

    # Identical attempts may be retries of one action. Different argument sets
    # remain separate; success of an unrelated action cannot cover a failure.
    def key(a: ToolAttempt) -> tuple[str, str]:
        return a.tool_name, json.dumps(a.arguments, sort_keys=True)

    groups = [list(group) for _, group in groupby(sorted(attempts, key=key), key)]
    group_support = []
    sources = []
    for group in groups:
        outcomes = [
            o for o in relevant if any(o.attempt_id == a.attempt_id for a in group)
        ]
        supported = False
        for o in outcomes:
            sources.append(o.outcome_id)
            proof = verifications.get(o.outcome_id)
            if proof is not None:
                supported = True
                sources.append(proof.outcome_id)
            elif _positive(o, claim.action):
                supported = True
        group_support.append(supported)
    if any(group_support) and not all(group_support):
        return result(
            Verdict.INCONCLUSIVE,
            "Several possible actions have conflicting outcome evidence.",
            *sources,
        )
    if claim.polarity == "negative":
        if all(group_support):
            return result(
                Verdict.FAIL,
                "Negative action claim contradicts observed success evidence.",
                *sources,
            )
        if any(
            o.status not in {ToolOutcomeStatus.ERROR, ToolOutcomeStatus.BLOCKED}
            or o.state_transition_ids
            for o in relevant
        ):
            return result(
                Verdict.INCONCLUSIVE,
                "An ambiguous outcome does not prove the action did not happen.",
                *sources,
            )
        return result(
            Verdict.PASS,
            "Negative action claim agrees with the failed action evidence.",
            *sources,
        )
    if all(group_support):
        return result(
            Verdict.PASS,
            "Affirmative action claim is supported by preceding matching evidence.",
            *sources,
        )
    if claim.action not in {"send", "reply"} and not claim.declared:
        return result(
            Verdict.INCONCLUSIVE,
            "This action's claim vocabulary requires a declared contract.",
            *sources,
        )
    return result(
        Verdict.FAIL,
        "Affirmative action claim is unsupported by the observed action outcomes.",
        *sources,
    )


def assess_claims(
    scenario: Scenario, run: CanonicalRun, criterion: OutputCriterion
) -> tuple[ClaimAssessment, ...]:
    raw_terms = criterion.parameters.get("success_terms", [])
    if not isinstance(raw_terms, (list, tuple)) or not all(
        isinstance(t, str) and t for t in raw_terms
    ):
        return (
            ClaimAssessment(
                Claim("", None, "uncertain", False),
                Verdict.INCONCLUSIVE,
                "Invalid declared success_terms contract.",
                (run.run_id,),
            ),
        )
    terms = tuple(t.casefold() for t in raw_terms)
    if run.final_output is None or not _consistent(scenario, run):
        return (
            ClaimAssessment(
                Claim("", None, "uncertain", False),
                Verdict.INCONCLUSIVE,
                "Tool events and outcome evidence are incomplete or inconsistent.",
                (run.run_id,),
            ),
        )
    statements = [
        (e.sequence, e.event_id, e.payload.get("text"))
        for e in run.events
        if e.event_type == CanonicalEventType.ASSISTANT_OUTPUT
    ]
    final = [e for e in run.events if e.event_type == CanonicalEventType.FINAL_OUTPUT]
    complete = observed_completion(scenario, run)
    if not statements or statements[-1][2] != run.final_output:
        statements.append(
            (
                run.events[-1].sequence + 1 if run.events else 0,
                final[-1].event_id if final else run.run_id,
                run.final_output,
            )
        )
    assessments: list[ClaimAssessment] = []
    for sequence, source, text in statements:
        if isinstance(text, str):
            assessments.extend(
                _assess(scenario, run, claim, sequence, source, complete)
                for claim in extract_claims(text, terms)
            )
    return tuple(assessments)
