"""Evidence-free discourse transitions; never erase an assertion or its history."""

from dataclasses import replace

from .claim_withdrawal import TargetKind

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
                or (
                    claim.withdrawal is not None
                    and claim.withdrawal.kind == TargetKind.ORDINAL
                )
            )
            and (not claim.references or set(claim.references).issubset(old.references))
            and (claim.action is None or claim.action == old.action)
            and (claim.channel is None or old.channel == claim.channel)
            and (
                claim.withdrawal is None
                or claim.withdrawal.aspect is None
                or old.aspect == claim.withdrawal.aspect
            )
        ]
        target = claim.withdrawal
        plural = False
        target_resolved = True
        if (
            target
            and target.kind
            in {TargetKind.PLURAL, TargetKind.ORDINAL, TargetKind.PARTIAL}
            and possible
        ):
            # A group is one assistant source event, never the entire transcript.
            # Explicit identity filters apply first; ordinals index that local group.
            latest_source = resolved[possible[-1]].source_id
            possible = [i for i in possible if resolved[i].source_id == latest_source]
            if target.kind == TargetKind.PARTIAL:
                # An unspecified subset is not a group withdrawal. Keep each
                # possible member unresolved, even with only one active candidate.
                target_resolved = False
            elif target.kind == TargetKind.PLURAL:
                plural = True
                target_resolved = target.count is None or len(possible) == target.count
            elif target.ordinal is not None and -len(possible) <= target.ordinal < len(
                possible
            ):
                possible = [possible[target.ordinal]]
            else:
                target_resolved = False
        if target and target.kind == TargetKind.UNRESOLVED:
            target_resolved = False
        # A fresh reaffirmation may refer to the single most recent version of
        # one proposition, but never choose between distinct operations/recipients.
        if relation == Relation.CONFIRMS and possible:
            identities = {
                (
                    resolved[i].action,
                    resolved[i].references,
                    resolved[i].channel,
                    resolved[i].aspect,
                )
                for i in possible
            }
            if len(identities) == 1:
                possible = [possible[-1]]
        # Bare reaffirmation inherits the proposition. Explicit verification is
        # an additional obligation, never erased by a weaker historical action.
        # Distinct non-default aspects require conjunction, which this bounded
        # lifecycle does not resolve: retain ambiguity rather than drop one.
        if relation == Relation.CONFIRMS and len(possible) == 1:
            old_aspect = resolved[possible[0]].aspect
            if claim.aspect != "action" and old_aspect not in {"action", claim.aspect}:
                target_resolved = False
        if not target_resolved or not possible or (not plural and len(possible) != 1):
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
        state = {
            Relation.RETRACTS: Life.RETRACTED,
            Relation.CORRECTS: Life.CORRECTED,
            Relation.CONFIRMS: Life.CONFIRMED,
        }[relation]
        for index in possible:
            old = resolved[index]
            transition = ClaimTransition(
                claim.claim_id,
                old.claim_id,
                relation,
                old.lifecycle,
                state,
                Resolution.RESOLVED,
            )
            resolved[index] = replace(
                old, lifecycle=state, transitions=(*old.transitions, transition)
            )
        # Confirmation preserves the entire proposition, including historical action vs
        # current membership. New timing never weakens what must be proved.
        old = resolved[possible[0]]
        if relation == Relation.CONFIRMS:
            claim = replace(
                claim,
                scope=ClaimScope.ASSERTED,
                action=old.action,
                aspect=old.aspect if claim.aspect == "action" else claim.aspect,
                references=old.references,
                channel=old.channel,
                lifecycle=Life.ACTIVE,
                antecedents=(old.claim_id,),
            )
        else:
            claim = replace(
                claim,
                lifecycle=Life.CONTROL,
                antecedents=tuple(resolved[i].claim_id for i in possible),
            )
        resolved.append(claim)
    return tuple(resolved)
