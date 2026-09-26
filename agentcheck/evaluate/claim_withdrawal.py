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


@dataclass(frozen=True)
class WithdrawalIntent:
    relation: str
    target: WithdrawalTarget


_NOUN = r"(?:claim|statement|assertion)"
# Morphology belongs to the intent grammar; target cardinality and proposition
# resolution are separate. Non-affirmative/modal/reported frames cannot match.
_VERB = r"(?:retract|withdraw|revoke|disavow|disclaim|correct)"


def withdrawal_intent(text: str) -> WithdrawalIntent | None:
    body = text.strip().rstrip(".!, ").replace("’", "'")
    reset = re.match(r"(?:correction:|actually[, :])\s*(.+)$", body, re.I)
    if reset:
        body = reset.group(1)
    relation = "retracts"
    act = re.fullmatch(rf"I (?P<verb>{_VERB}) (?P<target>.+)", body, re.I)
    target = None
    if act:
        target = act.group("target")
        if act.group("verb").lower() == "correct":
            relation = "corrects"
    else:
        take = re.fullmatch(r"I take (?:back (.+)|(.+) back)", body, re.I)
        wrong = re.fullmatch(r"(.+) was (?:wrong|incorrect)", body, re.I)
        neither = re.fullmatch(
            rf"Neither of (?:those|these|my) ({_NOUN}s) should stand", body, re.I
        )
        if take:
            target = take.group(1) or take.group(2)
        elif re.fullmatch(r"ignore that(?:, I was wrong)?", body, re.I):
            target = "that"
        elif wrong:
            target = wrong.group(1)
            relation = "corrects" if reset else "retracts"
        elif neither:
            target = "both " + neither.group(1)
        elif re.fullmatch(r"(?:forget|ignore) what I just said", body, re.I):
            target = "that statement"
    if target is None:
        return None
    resolved = withdrawal_target(target)
    return WithdrawalIntent(relation, resolved) if resolved else None


def withdrawal_target(text: str) -> WithdrawalTarget | None:
    target = text.strip()
    # Cardinality is explicit. 'Both' is not a wildcard for any number of claims.
    plural = re.fullmatch(
        rf"(?:(both)(?: of (?:those|these|my))?|my(?: previous| earlier)?|those|these) {_NOUN}s",
        target,
        re.I,
    )
    if plural:
        return WithdrawalTarget(
            TargetKind.PLURAL, target, count=2 if plural.group(1) else None
        )
    ordinal = re.fullmatch(
        rf"(?:the|my) (first|second|third|last) {_NOUN}", target, re.I
    )
    if ordinal:
        return WithdrawalTarget(
            TargetKind.ORDINAL,
            target,
            ordinal={"first": 0, "second": 1, "third": 2, "last": -1}[
                ordinal.group(1).lower()
            ],
        )
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
