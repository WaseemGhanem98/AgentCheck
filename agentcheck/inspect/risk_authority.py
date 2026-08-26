"""Resolve one tool's effective risk from every source AgentCheck can consult.

Precedence, most authoritative first, applied independently per axis
(``state_changing``, ``destructive``) because a developer may declare one axis
and leave the other to inference:

1. developer declaration (``tool_risk`` in ``agentcheck.json``);
2. genuine framework metadata -- no adapter AgentCheck currently supports
   exposes an authoritative side-effect flag from its SDK, so this tier is
   named here but never reached today. See ``RiskAuthority.FRAMEWORK_AUTHORITATIVE``.
3. heuristic inference from the tool's declared name and description;
4. ``UNKNOWN``, when the heuristic found no evidence either.

A declaration always wins over inference *on the axis it names*. Leaving one
axis undeclared must not upgrade its authority: reporting the whole tool as
"developer declared" when only one axis was actually declared would overstate
certainty about the axis that was not. Disagreement between a declaration and
inference is recorded in ``conflicts`` rather than silently discarded, and the
declaration still wins -- a hidden conflict is worse than a visible one that
still resolves deterministically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from agentcheck.domain import RiskAuthority, RiskAxis, SpecEvidence, ToolRiskAssertion

from .capabilities import classify_tool_risk

if TYPE_CHECKING:
    from agentcheck.config import ToolRiskDeclaration


_INFERRED_LOCATOR_SUFFIX = ".name+description"


def _resolve_axis(
    *,
    tool_name: str,
    axis_name: str,
    declared_value: bool | None,
    inferred_value: bool,
    inferred_confidence: float,
    inferred_known: bool,
    framework_value: bool | None,
) -> tuple[RiskAxis, str | None]:
    """Resolve one axis, returning it plus a conflict note when tiers disagree."""

    if declared_value is not None:
        conflict = None
        if framework_value is not None and framework_value != declared_value:
            conflict = (
                f"{tool_name}.{axis_name}: developer declared {declared_value}, "
                f"but framework metadata says {framework_value}; the developer "
                "declaration is authoritative"
            )
        elif inferred_known and inferred_value != declared_value:
            conflict = (
                f"{tool_name}.{axis_name}: developer declared {declared_value}, "
                f"overriding heuristic inference of {inferred_value} from this "
                "tool's declared name and description"
            )
        return (
            RiskAxis(
                value=declared_value,
                authority=RiskAuthority.DEVELOPER_DECLARED,
                confidence=1.0,
            ),
            conflict,
        )
    if framework_value is not None:
        conflict = None
        if inferred_known and inferred_value != framework_value:
            conflict = (
                f"{tool_name}.{axis_name}: framework metadata says {framework_value}, "
                f"disagreeing with heuristic inference of {inferred_value}; the "
                "framework value is authoritative"
            )
        return (
            RiskAxis(
                value=framework_value,
                authority=RiskAuthority.FRAMEWORK_AUTHORITATIVE,
                confidence=1.0,
            ),
            conflict,
        )
    if inferred_known:
        return (
            RiskAxis(
                value=inferred_value,
                authority=RiskAuthority.INFERRED,
                confidence=inferred_confidence,
            ),
            None,
        )
    return (
        RiskAxis(value=False, authority=RiskAuthority.UNKNOWN, confidence=0.0),
        None,
    )


def resolve_tool_risk(
    name: str,
    description: str | None,
    *,
    declared: "ToolRiskDeclaration | None" = None,
    framework_state_changing: bool | None = None,
    framework_destructive: bool | None = None,
) -> tuple[bool, bool, ToolRiskAssertion]:
    """Resolve one tool's risk, returning the bools every consumer already uses

    plus the full :class:`ToolRiskAssertion` provenance record. The bools are
    exactly what ``ToolDefinition.state_changing``/``destructive`` already
    carry, so every existing consumer (coverage, fault generation, the derived
    policy pack, the gateway) needs no change to keep working; only code that
    wants to *distinguish* authoritative from inferred/unknown risk needs to
    read the assertion.
    """

    inferred = classify_tool_risk(name, description)
    declared_state_changing = declared.state_changing if declared is not None else None
    declared_destructive = declared.destructive if declared is not None else None

    state_changing_axis, state_conflict = _resolve_axis(
        tool_name=name,
        axis_name="state_changing",
        declared_value=declared_state_changing,
        inferred_value=inferred.state_changing,
        inferred_confidence=inferred.confidence,
        inferred_known=inferred.known,
        framework_value=framework_state_changing,
    )
    destructive_axis, destructive_conflict = _resolve_axis(
        tool_name=name,
        axis_name="destructive",
        declared_value=declared_destructive,
        inferred_value=inferred.destructive,
        inferred_confidence=inferred.confidence,
        inferred_known=inferred.known,
        framework_value=framework_destructive,
    )

    conflicts = tuple(note for note in (state_conflict, destructive_conflict) if note)

    locator = f"tool:{name}{_INFERRED_LOCATOR_SUFFIX}"
    evidence: list[SpecEvidence] = [
        SpecEvidence(
            evidence_id=f"tool-risk:{name}:state_changing",
            summary=(
                f"state_changing resolved to {state_changing_axis.value} via "
                f"{state_changing_axis.authority.value}."
            ),
            locator=locator,
        ),
        SpecEvidence(
            evidence_id=f"tool-risk:{name}:destructive",
            summary=(
                f"destructive resolved to {destructive_axis.value} via "
                f"{destructive_axis.authority.value}."
            ),
            locator=locator,
        ),
    ]
    for index, note in enumerate(conflicts):
        evidence.append(
            SpecEvidence(
                evidence_id=f"tool-risk:{name}:conflict:{index}",
                summary=note,
                locator=locator,
            )
        )

    assertion = ToolRiskAssertion(
        tool_name=name,
        state_changing=state_changing_axis,
        destructive=destructive_axis,
        evidence=tuple(evidence),
        conflicts=conflicts,
    )
    return state_changing_axis.value, destructive_axis.value, assertion


def declared_risk_for(
    name: str, tool_risk: "Mapping[str, ToolRiskDeclaration] | None"
) -> "ToolRiskDeclaration | None":
    if not tool_risk:
        return None
    return tool_risk.get(name)


__all__ = [
    "declared_risk_for",
    "resolve_tool_risk",
]
