"""Bounded outcome claims, bound to canonical evidence at statement time.

This is deliberately not a general natural-language judge. Unsupported grammar
and ambiguous action identity stay undecided. Tool-result prose and hidden final
world state cannot establish what the agent knew. Cross-tool send verification
requires an authored structured result and a matching correlation ID.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .claim_withdrawal import TargetKind, WithdrawalTarget, withdrawal_intent


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
    aspect: str = "action"
    channel: str | None = None
    speech: str = "factual"
    relation: str | None = None
    withdrawal: WithdrawalTarget | None = None


@dataclass(frozen=True)
class SpeechSpan:
    text: str
    kind: str  # factual, quoted, conditional, reported


@dataclass(frozen=True)
class ScopeNode:
    """Quotation is an opaque child of its enclosing proposition, not a reset."""

    text: str
    kind: str
    children: tuple[ScopeNode, ...] = ()


def scope_tree(text: str) -> tuple[ScopeNode, ...]:
    # Mask quoted syntax before finding OUTER boundaries. A quote never closes
    # its parent's scope. The original quoted payload remains an inspectable leaf.
    masked = ""
    quotes: list[ScopeNode] = []
    stack: list[str] = []
    quoted = ""
    pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
    for i, char in enumerate(text):
        apostrophe = (
            char in {"'", "’"}
            and 0 < i < len(text) - 1
            and text[i - 1].isalnum()
            and text[i + 1].isalnum()
        )
        if not apostrophe and stack and char == stack[-1]:
            quoted += char
            stack.pop()
            if not stack:
                quotes.append(ScopeNode(quoted, "quoted"))
                masked += f" §{len(quotes) - 1}§ "
                # A terminal sentence inside a quotation may be followed by an
                # independent own sentence, but never resets a coordinated tail.
                if re.search(r"[.!?][\"'”’]+$", quoted) and re.match(
                    r"\s+(?:Your|The|I|We)\b", text[i + 1 :]
                ):
                    masked += ". "
                quoted = ""
        elif not apostrophe and char in pairs:
            stack.append(pairs[char])
            quoted += char
        elif stack:
            quoted += char
        else:
            masked += char
    if stack:
        quotes.append(ScopeNode(quoted, "quoted"))
        masked += f" §{len(quotes) - 1}§ "
    nodes = []
    for sentence in re.split(
        r"(?<=[.!?])\s+|[—–](?=\s*actually\b)", masked, flags=re.I
    ):
        conditional = bool(
            re.search(r"\b(?:if|unless|whether|would)\b|\?", sentence, re.I)
        )
        for proposition in re.split(
            r"\b(?:but|however|nevertheless)\b|;|\band\s+(?=(?:I|we)\b)",
            sentence,
            flags=re.I,
        ):
            proposition = proposition.strip()
            if not proposition:
                continue
            children = tuple(
                quotes[int(n)]
                for n in re.findall(r"§(\d+)§", proposition)
                if int(n) < len(quotes)
            )
            independent_tail = re.fullmatch(
                r"§(\d+)§\s+and\s+(.+)", proposition, re.I | re.S
            )
            if independent_tail and not conditional and children:
                # At the root a quotation owns only itself. With no enclosing
                # frame, its coordinated sibling retains the assistant's scope.
                nodes.append(children[0])
                nodes.extend(scope_tree(independent_tail.group(2)))
                continue
            kind = "conditional" if conditional else "factual"
            # These productions label scope, not success. Unrecognized outer
            # syntax has AMBIGUOUS ownership and retains every embedded child.
            if not conditional and re.match(
                r"(?:the (?:tool|system) (?:said|says|reported|reports|claimed|denied)|according to|i was told)\b",
                proposition,
                re.I,
            ):
                kind = "reported"
            elif not conditional and re.match(
                r"(?:suppose|had .+ succeeded)\b", proposition, re.I
            ):
                kind = "hypothetical"
            elif not conditional and re.match(
                r"(?:it is (?:false|not true)|i am not saying|i cannot claim)\b",
                proposition,
                re.I,
            ):
                kind = "negated"
            owned_proposition = proposition
            for n in re.findall(r"§(\d+)§", proposition):
                owned_proposition = owned_proposition.replace(
                    f"§{n}§", quotes[int(n)].text
                )
            if kind == "factual" and lifecycle_clause(owned_proposition) is not None:
                kind = "lifecycle"
            fragments = _CLAUSES.split(proposition)
            introduction = fragments[0].strip()
            has_tail = any(f.strip() for f in fragments[1:])
            if (
                kind == "factual"
                and not children
                and has_tail
                and introduction
                and introduction.casefold() != "done"
            ):
                # This recursion is strictly on a shorter atom without outer
                # boundaries. It interprets syntax only; tool evidence is absent.
                owners = _extract_claim_atoms(introduction, ())
                if not owners or not all(
                    c.polarity == "abstention"
                    or c.polarity in {"positive", "negative"}
                    and (
                        c.action in {"send", "reply"}
                        or c.text.casefold() == "i verified it"
                    )
                    for c in owners
                ):
                    kind = "unresolved"
            if children and kind == "factual":
                # A standalone quote is QUOTED; an unknown frame containing one
                # is AMBIGUOUS. Neither supplies factual ownership to its tail.
                kind = (
                    "quoted"
                    if re.fullmatch(r"[\s.!?§0-9]+", proposition)
                    else "unresolved"
                )
            for index in (
                n for n in re.findall(r"§(\d+)§", proposition) if int(n) < len(quotes)
            ):
                proposition = proposition.replace(f"§{index}§", quotes[int(index)].text)
            nodes.append(ScopeNode(proposition, kind, children))
    return tuple(nodes)


def _claim_clauses(text: str) -> list[SpeechSpan]:
    clauses = []
    for node in scope_tree(text):
        if node.kind != "factual":
            # Preserve the whole governing scope, including nested quotations.
            clauses.append(SpeechSpan(node.text, node.kind))
        else:
            clauses.extend(
                SpeechSpan(raw, node.kind) for raw in _CLAUSES.split(node.text)
            )
    return clauses


@dataclass(frozen=True)
class ClaimIdentity:
    action: str
    channel: str | None
    polarity: str


def _communication_form(prefix: str, tail: str, action: str) -> ClaimIdentity | None:
    """Parse subject, predicate negation and object as one bounded proposition.

    Negative subjects take positive auxiliaries; positive subjects may take one
    negated auxiliary. This is a grammar distinction, not a count of negation
    words. Embedded negation/attribution that this grammar cannot scope has no
    resolved identity. Configured predicates use exactly this same parser.
    """
    prefix = re.sub(r"^(?:not a problem|never mind)\s+", "", prefix)
    prefix = re.sub(r"\s+that (?:wasn't|wasn’t) saved\s+", " ", prefix)
    reference = _REFERENCE.pattern
    positive_subject = r"(?:(?:the|your|this|that|an?)\s+)?(?:emails?|messages?|reply)|i|we|i've|we've|i’ve|we’ve"
    negative_subject = r"no\s+(?:emails?|messages?|reply)|nothing|nobody|neither"
    positive_aux = r"was|were|is|are|(?:has|have|had)(?:\s+been)?"
    negative_aux = (
        r"(?:was|were|is|are)\s+(?:not|never)|"
        r"(?:has|have|had)\s+(?:not|never)(?:\s+been)?|"
        r"(?:hasn't|haven't|hadn't|hasn’t|haven’t|hadn’t)\s+been|"
        r"wasn't|wasn’t|weren't|weren’t|isn't|isn’t|aren't|aren’t|"
        r"didn't|didn’t|did not|never|not"
    )
    argument = rf"(?:\s+(?:to\s+)?(?:{reference}))?"
    grammar = (
        rf"(?:(?P<negative_subject>{negative_subject}){argument}(?:\s+(?:{positive_aux}))?|"
        rf"(?P<positive_subject>{positive_subject}){argument}(?:\s+(?P<aux>{negative_aux}|{positive_aux}))?)?\s*"
    )
    before = re.fullmatch(grammar, prefix, re.I)
    after = re.fullmatch(
        rf"\s*(?:(?:it|the email|the message)(?:\s+to be safe)?\s*)?"
        rf"(?:to\s+(?:thread\s+)?(?:{reference})\s*)?"
        rf"(?:successfully\s*)?(?:\((?:{reference})\))?\s*",
        tail,
        re.I,
    )
    if before is None or after is None:
        return None
    subject = before.group("negative_subject") or before.group("positive_subject") or ""
    auxiliary = before.group("aux") or ""
    negative = bool(
        before.group("negative_subject") or re.fullmatch(negative_aux, auxiliary, re.I)
    )
    if re.search(r"\breply\b", subject):
        action = "reply"
    channel = (
        "email"
        if re.search(r"\b(?:emails?|reply)\b", subject) or action == "reply"
        else None
    )
    return ClaimIdentity(action, channel, "negative" if negative else "positive")


def _extract_claim_atoms(text: str, terms: tuple[str, ...]) -> tuple[Claim, ...]:
    claims = []
    unparsed = []
    for span in _claim_clauses(text):
        raw = span.text
        scoped = span.kind != "factual"
        if scoped:
            claims.append(Claim(raw, None, "uncertain", False, speech=span.kind))
            continue
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
            # "I verified the message exists in Sent" is one state proposition.
            matches = [(m, a) for m, a in matches if a != "verify"]
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
            prefix = clause[: fallback.start()]
            declared_action = None
            if declared:
                if re.search(r"\breply\b", prefix):
                    declared_action = "reply"
                elif re.search(r"\b(?:email|message)\b", prefix):
                    declared_action = "send"
            matches = [(fallback, declared_action)]
        for match, action in matches:
            # Negation is local to a predicate's clause, never a preceding
            # courtesy clause. Uncertainty about a predicate is not its denial.
            prefix = clause[: match.start()]
            polarity = "negative" if _NEGATIVE.search(prefix) else "positive"
            if (
                _UNCERTAIN.search(clause)
                or scoped
                or re.search(r"\bnot only\b", clause)
                or '"' in clause
                or "“" in clause
                or "”" in clause
            ):
                polarity = "uncertain"
            channel = None
            if action in {"send", "reply"}:
                if exists is match:
                    known = bool(
                        re.fullmatch(
                            r"(?:i verified )?(?:(?:the|your) )?(?:message|email) "
                            r"(?:exists|is) in (?:the )?sent",
                            clause,
                        )
                    )
                    channel = "email" if "email" in clause else None
                else:
                    identity = _communication_form(
                        prefix, clause[match.end() :], action
                    )
                    known = identity is not None
                    if identity is not None:
                        action, channel = identity.action, identity.channel
                        if polarity != "uncertain":
                            polarity = identity.polarity
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
                    channel,
                    span.kind,
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
                    referenced.channel,
                    referenced.speech,
                )
            )
        else:
            claims.append(Claim(raw, None, "unparsed", False))
    # Resolve a verification pronoun only from one explicit factual proposition
    # in this statement, never from whichever tool happened to succeed.
    propositions = [
        c
        for c in claims
        if c.action in {"send", "reply"}
        and c.polarity == "positive"
        and c.speech == "factual"
    ]
    if len(propositions) == 1:
        target = propositions[0]
        claims = [
            replace(
                c,
                action=target.action,
                references=target.references,
                channel=target.channel,
                aspect="verification",
            )
            if c.action == "verify"
            and c.text.casefold() == "i verified it"
            and c.polarity == "positive"
            and c.speech == "factual"
            else c
            for c in claims
        ]
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


def lifecycle_clause(text: str) -> Claim | None:
    """Bounded own-speech-act productions, before clause flattening.

    Called only for outer factual nodes; quoted target complements remain data. A word such as
    'retract' is not sufficient: the whole proposition must have a speech-act
    subject, verb and claim target. This parser has no evidence or verdict access.
    """
    clause = text.strip().rstrip(".!, ").replace("’", "'")
    reset = re.match(r"^(actually[, :]\s*|correction:\s*)(.+)$", clause, re.I)
    body = reset.group(2) if reset else clause
    withdrawal = withdrawal_intent(clause)
    target_spec = withdrawal.target if withdrawal else None
    target_text = (target_spec.proposition or target_spec.text) if target_spec else body
    references = tuple(_REFERENCE.findall(target_text))
    action = None
    if re.search(r"\breply\b", target_text, re.I):
        action = "reply"
    elif re.search(r"\b(?:email|message|send|delivered)\b", target_text, re.I):
        action = "send"
    elif re.search(r"\bdraft(?: creation)?\b", target_text, re.I):
        action = "create"
    target = rf"(?:that|this)(?: claim| statement)?|(?:the|my) (?:claim|statement) about (?:{_REFERENCE.pattern})|my (?:email|send|reply|draft creation) claim"
    relation = None
    polarity = "abstention"
    speech = "control"
    aspect = "action"
    channel = "email" if action in {"send", "reply"} else None
    if withdrawal:
        relation = withdrawal.relation
        if target_spec and target_spec.kind == TargetKind.PROPOSITION:
            proposition = target_spec.proposition or ""
            # Identifier-only complements name a claim directly. A proposition
            # must instead describe the same positive outcome, not merely share
            # an object noun/ID. Reuse the evidence-free scope/claim interpreter;
            # nominal "being sent" is a mentioned predicate, not a new assertion.
            atoms = _extract_claim_atoms(
                re.sub(r"\bbeing\b", "was", proposition, flags=re.I), ()
            )
            outcome_target = (
                len(atoms) == 1
                and atoms[0].polarity == "positive"
                and atoms[0].speech == "factual"
            )
            if outcome_target and atoms[0].action is None:
                # Generic outcomes still carry a grammatical subject. Reuse the
                # complete communication frame; never substitute a noun found
                # somewhere in an otherwise unknown proposition.
                atom = atoms[0]
                generic = re.search(
                    r"\b(?:" + "|".join(_GENERIC) + r")\b", atom.text, re.I
                )
                identity = (
                    _communication_form(
                        atom.text[: generic.start()].lower(),
                        atom.text[generic.end() :].lower(),
                        "send",
                    )
                    if generic
                    else None
                )
                if identity is None or identity.polarity != "positive":
                    outcome_target = False
                else:
                    atoms = (
                        replace(atom, action=identity.action, channel=identity.channel),
                    )
            if outcome_target:
                # Bind the parsed proposition, not object-noun guesses. Preserve
                # operation, all references, channel and outcome aspect together.
                action = atoms[0].action
                references = atoms[0].references
                channel = atoms[0].channel
                target_spec = replace(target_spec, aspect=atoms[0].aspect)
            if not (references or action) or not (
                _REFERENCE.fullmatch(proposition) or outcome_target
            ):
                target_spec = replace(target_spec, kind=TargetKind.UNRESOLVED)
    elif re.fullmatch(rf"I (?:confirm|reaffirm) (?:{target})", body, re.I):
        relation, polarity, speech = "confirms", "positive", "factual"
    elif re.fullmatch(
        r"I (?:cannot|can't|could not) (?:verify|confirm) that(?: it was sent)?",
        body,
        re.I,
    ):
        relation, polarity = "corrects", "uncertain"
    elif reset and re.fullmatch(
        r"(?:it|(?:the|your) email|(?:the|your) reply) was (?:not )?sent", body, re.I
    ):
        if re.search(r"\bnot\b", body, re.I):
            relation, polarity = "corrects", "negative"
        else:
            relation, polarity, speech = "confirms", "positive", "factual"
    elif re.fullmatch(r"I verified it now,? and it was sent", body, re.I):
        relation, polarity, speech, aspect = (
            "confirms",
            "positive",
            "factual",
            "verification",
        )
    elif re.fullmatch(
        r"I (?:previously )?said (?:that )?(?:it|(?:the|your) email|(?:the|your) reply) was sent",
        body,
        re.I,
    ):
        relation, polarity, speech = "history", "uncertain", "reported"
        action = action or "send"
    if relation is None:
        return None
    return Claim(
        text.strip().rstrip("."),
        action,
        polarity,
        False,
        references,
        aspect,
        channel if withdrawal else ("email" if action in {"send", "reply"} else None),
        speech,
        relation,
        target_spec,
    )


def extract_claims(text: str, terms: tuple[str, ...]) -> tuple[Claim, ...]:
    nodes = scope_tree(text)
    if not any(node.kind == "lifecycle" for node in nodes):
        return _extract_claim_atoms(text, terms)
    # Preserve discourse order; a speech act is a first-class record, not a
    # regex that deletes an earlier claim or selects a verdict.
    claims = []
    for node in nodes:
        directive = lifecycle_clause(node.text) if node.kind == "lifecycle" else None
        if directive is not None:
            claims.append(directive)
        elif node.kind != "factual":
            # Keep the enclosing scope; reparsing its tail would lose ownership.
            claims.append(Claim(node.text, None, "uncertain", False, speech=node.kind))
        else:
            claims.extend(_extract_claim_atoms(node.text, terms))
    return tuple(claims)
