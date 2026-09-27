"""Own withdrawal acts and their discourse targets, independent of evidence.

A verb alone is never an act. An affirmative first-person performative (or an
explicit corrective judgment) must govern a claim noun phrase. Quoted material
is usable only as the complement of that frame, never as the act itself.
"""

from dataclasses import dataclass
from enum import Enum
import re


class TargetKind(str, Enum):
    SINGULAR = "singular"
    PLURAL = "plural"
    PARTIAL = "partial"
    ORDINAL = "ordinal"
    PROPOSITION = "proposition"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class WithdrawalTarget:
    kind: TargetKind
    text: str
    count: int | None = None
    ordinal: int | None = None
    proposition: str | None = None


class IntentKind(str, Enum):
    PERFORMATIVE = "performative"
    DIRECTIVE = "directive"
    ENDORSEMENT_REJECTION = "endorsement_rejection"
    CORRECTIVE_JUDGMENT = "corrective_judgment"


@dataclass(frozen=True)
class WithdrawalIntent:
    relation: str
    target: WithdrawalTarget
    kind: IntentKind = IntentKind.PERFORMATIVE


_NOUN = r"(?:claim|statement|assertion)"
# Morphology belongs to the intent grammar; target cardinality and proposition
# resolution are separate. Non-affirmative/modal/reported frames cannot match.
_VERB = r"(?:retract|withdraw|revoke|disavow|disclaim|correct|disregard|invalidate)"
# Tense/aspect changes the form, not the lifecycle effect. Keep affirmative
# active and completed passive frames separate so negation/modal scope survives.
_WITHDRAW_ACTIVE = r"(?:withdraw|am withdrawing|withdrew)"
_WITHDRAW_PASSIVE = r"(?:has|have) been withdrawn"


def withdrawal_intent(text: str) -> WithdrawalIntent | None:
    body = text.strip().rstrip(".!, ").replace("’", "'")
    reset = re.match(r"(?:correction:|actually[, :])\s*(.+)$", body, re.I)
    if reset:
        body = reset.group(1)
    relation = "retracts"
    kind = IntentKind.PERFORMATIVE
    act = re.fullmatch(
        rf"I (?P<verb>{_VERB}|{_WITHDRAW_ACTIVE}) (?P<target>.+)", body, re.I
    )
    target = None
    if act:
        target = act.group("target")
        if act.group("verb").lower() == "correct":
            relation = "corrects"
    else:
        passive = re.fullmatch(rf"(.+) {_WITHDRAW_PASSIVE}", body, re.I)
        take = re.fullmatch(r"I take (?:back (.+)|(.+) back)", body, re.I)
        wrong = re.fullmatch(
            r"(.+) (?:was|were) (?:wrong|incorrect|invalid)", body, re.I
        )
        dismissal = re.fullmatch(
            r"(?:please )?(?:disregard|ignore|forget) (.+)", body, re.I
        )
        endorsement = re.fullmatch(
            r"I (?:no longer stand by (.+)|(?:don't|do not) stand by (.+) anymore)",
            body,
            re.I,
        )
        standing = re.fullmatch(r"(.+) no longer (?:stand|stands)", body, re.I)
        neither = re.fullmatch(
            rf"Neither of (?:those|these|my) ({_NOUN}s) should stand", body, re.I
        )
        if passive:
            target = passive.group(1)
        elif take:
            target = take.group(1) or take.group(2)
        elif re.fullmatch(r"ignore that(?:, I was wrong)?", body, re.I):
            target = "that"
        elif wrong:
            target = wrong.group(1)
            kind = IntentKind.CORRECTIVE_JUDGMENT
            relation = "corrects" if reset else "retracts"
        elif endorsement or standing:
            match = endorsement or standing
            assert match is not None
            target = (
                (endorsement.group(1) or endorsement.group(2))
                if endorsement
                else match.group(1)
            )
            # Negating our endorsement of "either" rejects both; a judgment
            # about an unspecified member still leaves the member unresolved.
            kind = (
                IntentKind.ENDORSEMENT_REJECTION
                if endorsement
                else IntentKind.CORRECTIVE_JUDGMENT
            )
        elif neither:
            target = "both " + neither.group(1)
            kind = IntentKind.ENDORSEMENT_REJECTION
        elif dismissal:
            target = dismissal.group(1)
            kind = IntentKind.DIRECTIVE
    if target is None:
        return None
    resolved = withdrawal_target(target, kind)
    return WithdrawalIntent(relation, resolved, kind) if resolved else None


def withdrawal_target(
    text: str, intent: IntentKind = IntentKind.PERFORMATIVE
) -> WithdrawalTarget | None:
    target = text.strip()
    # Quantifiers select a set, an ordinal, or an unspecified subset. They are
    # not interchangeable: "one/either of those" never licenses choosing a claim.
    # Rejected endorsement distributes over "either" (neither is endorsed);
    # affirmative "withdraw either" leaves the choice unresolved.
    partial = re.fullmatch(
        rf"(one|either|some) of (?:those|these|my) {_NOUN}s", target, re.I
    )
    if partial:
        if (
            partial.group(1).lower() == "either"
            and intent == IntentKind.ENDORSEMENT_REJECTION
        ):
            return WithdrawalTarget(TargetKind.PLURAL, target, count=2)
        return WithdrawalTarget(
            TargetKind.PARTIAL,
            target,
            count=None if partial.group(1).lower() == "some" else 1,
        )
    quantified = re.fullmatch(
        rf"(each|both|all)(?: of (?:those|these|my))?(?: {_NOUN}s)?", target, re.I
    )
    if quantified:
        return WithdrawalTarget(
            TargetKind.PLURAL,
            target,
            count=2 if quantified.group(1).lower() == "both" else None,
        )
    plural = re.fullmatch(
        rf"(?:my(?: previous| earlier)?|those|these) {_NOUN}s", target, re.I
    )
    if plural:
        return WithdrawalTarget(TargetKind.PLURAL, target)
    ordinal = re.fullmatch(
        rf"(?:the|my) (first|second|third|last) (?:{_NOUN}|one)", target, re.I
    )
    if ordinal:
        return WithdrawalTarget(
            TargetKind.ORDINAL,
            target,
            ordinal={"first": 0, "second": 1, "third": 2, "last": -1}[
                ordinal.group(1).lower()
            ],
        )
    if re.fullmatch(r"what I (?:just )?(?:said|claimed|asserted)", target, re.I):
        return WithdrawalTarget(TargetKind.SINGULAR, target)
    # A proposition is mentioned, not asserted. Capture it only after a claim
    # noun or an explicit reference to our earlier speech.
    complement = re.fullmatch(
        rf"(?:(?:the|my|that|this)(?: previous| earlier)? {_NOUN}(?: that| about)?|what I said about)\s+(.+)",
        target,
        re.I,
    )
    if complement:
        proposition = complement.group(1).strip().strip("\"'“”‘’").rstrip(". ")
        return WithdrawalTarget(TargetKind.PROPOSITION, target, proposition=proposition)
    if re.fullmatch(
        rf"(?:that|this)(?: {_NOUN})?|(?:my|the)(?: previous| earlier)? {_NOUN}|my (?:email|send|reply|draft creation) claim",
        target,
        re.I,
    ):
        return WithdrawalTarget(TargetKind.SINGULAR, target)
    return None
