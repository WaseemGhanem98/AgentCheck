"""Budgeted hierarchical 1-deletion over replayable scenarios.

The search never weakens oracles, never executes invalid or secret-shaped
candidates, and never claims global minimality. Dimensions are searched in
``REDUCTION_ORDER`` and the pass repeats until a full sweep accepts no
deletion or the candidate/round budget is exhausted. ``locally_minimal``
means a complete unsuccessful sweep ran: no single remaining removable item
in any searched dimension can be deleted while preserving the failure
signature. That is 1-minimality within the searched dimensions, not a
globally smallest scenario.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import ValidationError

from agentcheck.domain import AgentSpec, CaseEvaluation, Scenario, Verdict
from agentcheck.generate.lint import lint_scenario
from agentcheck.replay.manifest import secret_shaped_reason

from .candidates import (
    REDUCTION_ORDER,
    ReductionDimension,
    dimension_items,
    minimum_keep,
    reconstruct_scenario,
    removable_indices,
)
from .complexity import ScenarioComplexity, is_strictly_smaller, measure_complexity
from .signature import (
    FailureSignature,
    extract_failure_signature,
    signatures_match,
    unsupported_failure_reason,
)


ExecuteScenario = Callable[[Scenario], CaseEvaluation]


@dataclass(frozen=True, slots=True)
class ShrinkSearchOutcome:
    scenario: Scenario
    complexity: ScenarioComplexity
    signature: FailureSignature
    candidate_executions: int
    accepted_reductions: int
    rejected_candidates: int
    skipped_invalid: int
    rejected_by_reason: tuple[tuple[str, int], ...]
    budget_exhausted: bool
    minimality: Literal["locally_minimal", "budget_exhausted"]
    rounds_completed: int


@dataclass
class _Budget:
    max_candidates: int
    max_rounds: int
    executions: int = 0
    accepted: int = 0
    rejected: int = 0
    skipped: int = 0
    rounds: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    exhausted: bool = False


def shrink_scenario(
    original: Scenario,
    original_evaluation: CaseEvaluation,
    *,
    spec: AgentSpec,
    execute: ExecuteScenario,
    max_candidates: int,
    max_rounds: int,
) -> ShrinkSearchOutcome:
    """Reduce ``original`` while preserving its failure signature."""

    signature = extract_failure_signature(original_evaluation)
    current = original
    current_complexity = measure_complexity(current)
    budget = _Budget(max_candidates=max_candidates, max_rounds=max_rounds)

    while not budget.exhausted:
        progress = False
        for dimension in REDUCTION_ORDER:
            if budget.exhausted:
                break
            removable = removable_indices(current, dimension, signature)
            if len(removable) <= minimum_keep(dimension):
                continue
            if budget.rounds >= budget.max_rounds:
                budget.exhausted = True
                break
            budget.rounds += 1
            reduced, reduced_complexity = _reduce_dimension(
                current,
                dimension,
                spec=spec,
                signature=signature,
                execute=execute,
                budget=budget,
                current_complexity=current_complexity,
            )
            if reduced.fingerprint != current.fingerprint:
                progress = True
                current = reduced
                current_complexity = reduced_complexity
        if not progress:
            break

    minimality: Literal["locally_minimal", "budget_exhausted"] = (
        "budget_exhausted" if budget.exhausted else "locally_minimal"
    )
    return ShrinkSearchOutcome(
        scenario=current,
        complexity=current_complexity,
        signature=signature,
        candidate_executions=budget.executions,
        accepted_reductions=budget.accepted,
        rejected_candidates=budget.rejected,
        skipped_invalid=budget.skipped,
        rejected_by_reason=tuple(sorted(budget.reasons.items())),
        budget_exhausted=budget.exhausted,
        minimality=minimality,
        rounds_completed=budget.rounds,
    )


def _reduce_dimension(
    current: Scenario,
    dimension: ReductionDimension,
    *,
    spec: AgentSpec,
    signature: FailureSignature,
    execute: ExecuteScenario,
    budget: _Budget,
    current_complexity: ScenarioComplexity,
) -> tuple[Scenario, ScenarioComplexity]:
    removable = removable_indices(current, dimension, signature)
    kept_removable = _minimize_by_deletion(
        removable,
        min_keep=minimum_keep(dimension),
        test=lambda kept: _try_keep(
            current,
            dimension,
            kept,
            spec=spec,
            signature=signature,
            execute=execute,
            budget=budget,
            current_complexity=current_complexity,
        ),
        budget=budget,
    )
    if kept_removable == removable:
        return current, current_complexity
    keep = _merge_keep(current, dimension, signature, kept_removable)
    try:
        candidate = reconstruct_scenario(current, dimension, keep)
    except (ValidationError, ValueError):
        return current, current_complexity
    return candidate, measure_complexity(candidate)


def _try_keep(
    current: Scenario,
    dimension: ReductionDimension,
    kept_removable: tuple[int, ...],
    *,
    spec: AgentSpec,
    signature: FailureSignature,
    execute: ExecuteScenario,
    budget: _Budget,
    current_complexity: ScenarioComplexity,
) -> bool | None:
    """True accept, False reject, None stop because the execution budget is gone."""

    if budget.exhausted:
        return None
    keep = _merge_keep(current, dimension, signature, kept_removable)
    try:
        candidate = reconstruct_scenario(current, dimension, keep)
    except (ValidationError, ValueError):
        budget.skipped += 1
        budget.reasons["invalid_scenario"] += 1
        return False
    if not is_strictly_smaller(measure_complexity(candidate), current_complexity):
        budget.skipped += 1
        budget.reasons["not_smaller"] += 1
        return False
    issues = lint_scenario(candidate, spec)
    if issues:
        budget.skipped += 1
        budget.reasons["lint"] += 1
        return False
    if secret_shaped_reason(candidate) is not None:
        budget.skipped += 1
        budget.reasons["secret_shaped"] += 1
        return False
    if budget.executions >= budget.max_candidates:
        budget.exhausted = True
        return None
    budget.executions += 1
    evaluation = execute(candidate)
    if evaluation.verdict == Verdict.PASS:
        budget.rejected += 1
        budget.reasons["pass"] += 1
        return False
    if evaluation.verdict == Verdict.INCONCLUSIVE:
        budget.rejected += 1
        budget.reasons["inconclusive"] += 1
        return False
    if evaluation.verdict == Verdict.INFRA_ERROR:
        budget.rejected += 1
        budget.reasons["infra_error"] += 1
        return False
    if unsupported_failure_reason(evaluation) is not None:
        budget.rejected += 1
        budget.reasons["unsupported_signature"] += 1
        return False
    try:
        candidate_signature = extract_failure_signature(evaluation)
    except ValueError:
        budget.rejected += 1
        budget.reasons["unsupported_signature"] += 1
        return False
    if not signatures_match(signature, candidate_signature):
        budget.rejected += 1
        budget.reasons["signature_mismatch"] += 1
        return False
    budget.accepted += 1
    return True


def _merge_keep(
    scenario: Scenario,
    dimension: ReductionDimension,
    signature: FailureSignature,
    kept_removable: tuple[int, ...],
) -> tuple[int, ...]:
    removable = set(kept_removable)
    all_removable = set(removable_indices(scenario, dimension, signature))
    kept: list[int] = []
    for index in range(len(dimension_items(scenario, dimension))):
        if index in all_removable and index not in removable:
            continue
        kept.append(index)
    return tuple(kept)


def _minimize_by_deletion(
    items: tuple[int, ...],
    *,
    min_keep: int,
    test: Callable[[tuple[int, ...]], bool | None],
    budget: _Budget,
) -> tuple[int, ...]:
    """Delete one remaining original index at a time, left to right.

    After an accepted deletion the pass restarts so later items cannot mask an
    earlier now-removable index. Completing a dimension is 1-minimal for that
    dimension, not a proof that the whole scenario is globally smallest.
    """

    current = items
    progress = True
    while progress and not budget.exhausted and len(current) > min_keep:
        progress = False
        for index in current:
            if budget.exhausted:
                return current
            candidate = tuple(item for item in current if item != index)
            if len(candidate) < min_keep:
                continue
            accepted = test(candidate)
            if accepted is True:
                current = candidate
                progress = True
                break
            if accepted is None:
                return current
    return current


__all__ = ["ExecuteScenario", "ShrinkSearchOutcome", "shrink_scenario"]
