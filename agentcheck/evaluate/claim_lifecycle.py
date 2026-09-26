"""Evidence-free discourse transitions; never erase an assertion or its history."""

from dataclasses import replace

from .claim_states import (
    ClaimLifecycle as Life,
    ClaimRelation as Relation,
    ClaimScope,
    ClaimTransition,
    EvaluatedClaim,
    Resolution,
)


def resolve_lifecycle(claims: tuple[EvaluatedClaim, ...]) -> tuple[EvaluatedClaim, ...]:
    resolved: list[EvaluatedClaim] = []
    for claim in claims:
        relation = claim.relation
        if relation is None:
            resolved.append(claim)
            continue
        if relation == Relation.HISTORY:
            resolved.append(replace(claim, lifecycle=Life.CONTROL))
            continue
        # Only our own factual control/assertion acts may alter our claim state.
        # A quoted/reported/hypothetical/negated occurrence never reaches this arm.
        if claim.scope not in {ClaimScope.NON_CLAIM, ClaimScope.ASSERTED} and not (
            relation == Relation.CORRECTS
            and claim.scope in {ClaimScope.NEGATED, ClaimScope.UNCERTAIN}
        ):
            resolved.append(claim)
            continue
        possible = [
            i
            for i, old in enumerate(resolved)
            if (old.scope == ClaimScope.ASSERTED or old.relation == Relation.HISTORY)
            and (
                old.lifecycle in {Life.ACTIVE, Life.AMBIGUOUS}
                or relation == Relation.CONFIRMS
                or old.relation == Relation.HISTORY
            )
            and (not claim.references or set(claim.references).issubset(old.references))
            and (claim.action is None or claim.action == old.action)
            and (claim.channel is None or old.channel == claim.channel)
        ]
        # A fresh reaffirmation may refer to the single most recent version of
        # one proposition, but never choose between distinct operations/recipients.
        if relation == Relation.CONFIRMS and possible:
            identities = {
                (resolved[i].action, resolved[i].references, resolved[i].channel)
                for i in possible
            }
            if len(identities) == 1:
                possible = [possible[-1]]
        if len(possible) != 1:
            for i in possible:
                old = resolved[i]
                transition = ClaimTransition(
                    claim.claim_id,
                    old.claim_id,
                    relation,
                    old.lifecycle,
                    Life.AMBIGUOUS,
                    Resolution.AMBIGUOUS,
                )
                resolved[i] = replace(
                    old,
                    lifecycle=Life.AMBIGUOUS,
                    transitions=(*old.transitions, transition),
                )
            resolved.append(
                replace(
                    claim,
                    lifecycle=Life.AMBIGUOUS,
                    antecedents=tuple(resolved[i].claim_id for i in possible),
                )
            )
            continue
        index = possible[0]
        old = resolved[index]
        state = {
            Relation.RETRACTS: Life.RETRACTED,
            Relation.CORRECTS: Life.CORRECTED,
            Relation.CONFIRMS: Life.CONFIRMED,
        }[relation]
        transition = ClaimTransition(
            claim.claim_id,
            old.claim_id,
            relation,
            old.lifecycle,
            state,
            Resolution.RESOLVED,
        )
        # Confirmation adds a NEW active assertion at its own temporal position.
        # It does not retroactively replace the old claim's supporting evidence.
        resolved[index] = replace(
            old, lifecycle=state, transitions=(*old.transitions, transition)
        )
        if relation == Relation.CONFIRMS:
            claim = replace(
                claim,
                scope=ClaimScope.ASSERTED,
                action=old.action,
                references=old.references,
                channel=old.channel,
                lifecycle=Life.ACTIVE,
                antecedents=(old.claim_id,),
            )
        else:
            claim = replace(claim, lifecycle=Life.CONTROL, antecedents=(old.claim_id,))
        resolved.append(claim)
    return tuple(resolved)
