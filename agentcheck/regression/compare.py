"""Compare two stored AgentCheck runs without executing a target.

Scenarios are matched by structural fingerprint, then by display ID for a
scenario whose content changed. Array position is never an identity: two runs
of the same suite can order, select, or lint cases differently.
"""

from __future__ import annotations

from agentcheck.baseline.build import baseline_from_loaded
from agentcheck.baseline.compare import compare_baselines
from agentcheck.baseline.contract import ComparisonItem, EvaluationBaseline
from agentcheck.errors import ConfigurationError
from agentcheck.report.load import LoadedRun

from .contract import (
    BLOCKING_CATEGORIES,
    MAX_COMPARISON_ITEMS,
    UNCERTIFIABLE_CATEGORIES,
    Comparability,
    ComparabilityCaveat,
    IncomparableReason,
    RunComparison,
    RunComparisonCategory,
    RunComparisonItem,
)


_CATEGORY_ORDER = (
    "new_regression",
    "changed_failure",
    "uncertifiable_failure",
    "unchanged_failure",
    "resolved_failure",
    "changed_scenario",
    "new_scenario",
    "removed_scenario",
    "deselected_scenario",
    "inconclusive_change",
    "infra_change",
    "unchanged",
)


def compare_runs(base: LoadedRun, head: LoadedRun) -> RunComparison:
    """Classify behavioral differences between two stored runs of one target."""

    if base.root != head.root:
        raise ConfigurationError(
            "run comparison requires two runs of the same target directory"
        )
    base_snapshot = baseline_from_loaded(base)
    head_snapshot = baseline_from_loaded(head)
    shared = _shared_scenario_count(base_snapshot, head_snapshot)

    incomparable_reason = _incomparable_reason(
        base, head, base_snapshot, head_snapshot, shared=shared
    )
    caveats = _caveats(base, head, base_snapshot, head_snapshot)
    if incomparable_reason is not None:
        return RunComparison(
            base_run_id=base.run_id,
            head_run_id=head.run_id,
            comparability=Comparability.INCOMPARABLE,
            incomparable_reason=incomparable_reason,
            caveats=caveats,
            base_scenario_count=len(base_snapshot.cases),
            head_scenario_count=len(head_snapshot.cases),
            shared_scenario_count=shared,
            base_failure_count=_failure_count(base_snapshot),
            head_failure_count=_failure_count(head_snapshot),
            new_regression_count=0,
            changed_failure_count=0,
            resolved_count=0,
            unchanged_failure_count=0,
            uncertifiable_count=0,
        )

    classified = compare_baselines(base_snapshot, head_snapshot)
    deselected = _deselected_ids(head)
    items = tuple(
        sorted(
            (_item(entry, deselected) for entry in classified.items),
            key=lambda item: (
                _CATEGORY_ORDER.index(item.category),
                item.fingerprint,
                item.scenario_id,
            ),
        )
    )
    if any(item.category == "changed_scenario" for item in items):
        caveats = (*caveats, ComparabilityCaveat.SCENARIO_IDENTITY_CHANGED)
    visible = items[:MAX_COMPARISON_ITEMS]
    return RunComparison(
        base_run_id=base.run_id,
        head_run_id=head.run_id,
        comparability=Comparability.COMPARABLE,
        caveats=caveats,
        base_scenario_count=len(base_snapshot.cases),
        head_scenario_count=len(head_snapshot.cases),
        shared_scenario_count=shared,
        base_failure_count=_failure_count(base_snapshot),
        head_failure_count=_failure_count(head_snapshot),
        new_regression_count=_count(items, "new_regression"),
        changed_failure_count=_count(items, "changed_failure"),
        resolved_count=_count(items, "resolved_failure"),
        unchanged_failure_count=_count(items, "unchanged_failure"),
        uncertifiable_count=sum(
            _count(items, category) for category in UNCERTIFIABLE_CATEGORIES
        ),
        items=visible,
        omitted_items=len(items) - len(visible),
    )


def exit_code(comparison: RunComparison) -> int:
    """CI status for a completed comparison.

    Mirrors ``agentcheck test``: 2 is an outcome AgentCheck could not certify,
    1 is an attributable behavioral regression, 3 is weak or absent evidence,
    and 0 is no detected regression. A comparison that cannot be made is never
    reported as a clean result.
    """

    if comparison.comparability is Comparability.INCOMPARABLE:
        return 3
    if comparison.uncertifiable_count:
        return 2
    if any(item.blocking for item in comparison.items):
        return 1
    return 0


def _incomparable_reason(
    base: LoadedRun,
    head: LoadedRun,
    base_snapshot: EvaluationBaseline,
    head_snapshot: EvaluationBaseline,
    *,
    shared: int,
) -> IncomparableReason | None:
    if base.run_id == head.run_id:
        return IncomparableReason.SAME_RUN
    if shared:
        return None
    shared_ids = {item.scenario_id for item in base_snapshot.cases}.intersection(
        item.scenario_id for item in head_snapshot.cases
    )
    if shared_ids:
        return None
    # Nothing links the two runs, so every classification would describe the
    # identity change rather than a behavior.
    return IncomparableReason.NO_SHARED_SCENARIOS


def _caveats(
    base: LoadedRun,
    head: LoadedRun,
    base_snapshot: EvaluationBaseline,
    head_snapshot: EvaluationBaseline,
) -> tuple[ComparabilityCaveat, ...]:
    caveats: list[ComparabilityCaveat] = []
    # Every binding below comes from what each run recorded for itself. The
    # configured spec and frozen suite files can postdate either run, so they
    # are never comparison inputs.
    if base_snapshot.spec_id != head_snapshot.spec_id or (
        base.behavioral_coverage.spec_digest != head.behavioral_coverage.spec_digest
    ):
        caveats.append(ComparabilityCaveat.SPEC_CHANGED)
    # The scenario digest is always recorded and hashes the evaluated
    # (scenario_id, fingerprint) multiset, so it answers "same scenario set?"
    # even for a run that used no frozen suite.
    scenario_set_changed = (
        base.behavioral_coverage.scenario_digest
        != head.behavioral_coverage.scenario_digest
    )
    if scenario_set_changed:
        caveats.append(ComparabilityCaveat.SCENARIO_SET_CHANGED)
    base_suite = base.behavioral_coverage.suite_fingerprint
    head_suite = head.behavioral_coverage.suite_fingerprint
    if base_suite is not None and head_suite is not None and base_suite != head_suite:
        caveats.append(ComparabilityCaveat.SUITE_FINGERPRINT_CHANGED)
    if base.seed != head.seed:
        caveats.append(ComparabilityCaveat.SEED_CHANGED)
    if scenario_set_changed and (
        base.selection is not None or head.selection is not None
    ):
        # Only meaningful when a scenario is actually absent from one side.
        caveats.append(ComparabilityCaveat.SELECTION_ACTIVE)
    if base.git_revision is None or head.git_revision is None:
        caveats.append(ComparabilityCaveat.SOURCE_REVISION_UNRECORDED)
    elif base.git_revision == head.git_revision:
        caveats.append(ComparabilityCaveat.SOURCE_REVISION_UNCHANGED)
    return tuple(caveats)


def _deselected_ids(head: LoadedRun) -> frozenset[str]:
    if head.selection is None:
        return frozenset()
    return frozenset(head.selection.excluded_ids)


def _item(entry: ComparisonItem, deselected: frozenset[str]) -> RunComparisonItem:
    category: RunComparisonCategory = entry.category
    blocking = entry.blocking
    if category == "removed_scenario" and entry.scenario_id in deselected:
        # This scenario was excluded before execution, so the head run holds no
        # evidence about it either way. Reporting it as removed would claim a
        # suite change that did not happen.
        category = "deselected_scenario"
        blocking = False
    return RunComparisonItem(
        category=category,
        scenario_id=entry.scenario_id,
        fingerprint=entry.fingerprint,
        base_verdict=entry.baseline_verdict,
        head_verdict=entry.current_verdict,
        base_failure_fingerprint=entry.baseline_failure_fingerprint,
        head_failure_fingerprint=entry.current_failure_fingerprint,
        blocking=blocking and category in BLOCKING_CATEGORIES,
    )


def _shared_scenario_count(
    base: EvaluationBaseline, head: EvaluationBaseline
) -> int:
    base_fingerprints = {item.fingerprint for item in base.cases}
    head_fingerprints = {item.fingerprint for item in head.cases}
    return len(base_fingerprints.intersection(head_fingerprints))


def _failure_count(snapshot: EvaluationBaseline) -> int:
    return sum(1 for item in snapshot.cases if item.verdict == "FAIL")


def _count(items: tuple[RunComparisonItem, ...], category: str) -> int:
    return sum(1 for item in items if item.category == category)


__all__ = ["compare_runs", "exit_code"]
