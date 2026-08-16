"""Compare two evaluation baselines without executing a target."""

from __future__ import annotations

from .contract import (
    BLOCKING_CATEGORIES,
    CERTIFICATION_FAILURE_CATEGORIES,
    BaselineCase,
    BaselineComparison,
    ComparisonCategory,
    ComparisonItem,
    EvaluationBaseline,
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
    "inconclusive_change",
    "infra_change",
    "unchanged",
)


def compare_baselines(
    baseline: EvaluationBaseline,
    current: EvaluationBaseline,
) -> BaselineComparison:
    """Classify scenario-level differences. Human reviews are ignored."""

    items = _comparison_items(baseline, current)
    return BaselineComparison(
        baseline_id=baseline.baseline_id,
        current_run_id=current.created_from_run_id,
        baseline_scenario_count=len(baseline.cases),
        current_scenario_count=len(current.cases),
        baseline_failure_count=_failure_count(baseline),
        current_failure_count=_failure_count(current),
        new_regression_count=_count(items, "new_regression", blocking_only=True),
        resolved_count=_count(items, "resolved_failure"),
        unchanged_failure_count=_count(items, "unchanged_failure"),
        changed_failure_count=_count(items, "changed_failure", blocking_only=True),
        items=items,
        blocking_categories=BLOCKING_CATEGORIES,
    )


def gate_exit_code(comparison: BaselineComparison) -> int:
    """CI status for a completed comparison. Distinct from ``agentcheck test``."""

    if any(item.category in CERTIFICATION_FAILURE_CATEGORIES for item in comparison.items):
        return 2
    if any(item.blocking for item in comparison.items):
        return 1
    return 0


def format_comparison(comparison: BaselineComparison) -> str:
    removed_known_failures = sum(
        1
        for item in comparison.items
        if item.category == "removed_scenario" and item.baseline_verdict == "FAIL"
    )
    fail_to_inconclusive = sum(
        1
        for item in comparison.items
        if item.baseline_verdict == "FAIL" and item.current_verdict == "INCONCLUSIVE"
    )
    uncertifiable = _count(comparison.items, "uncertifiable_failure")
    lines = [
        "Baseline:",
        f"{comparison.baseline_scenario_count} scenarios",
        f"{comparison.baseline_failure_count} known failures",
        "",
        "Current:",
        f"{comparison.current_scenario_count} scenarios",
        f"{comparison.current_failure_count} failures",
        "",
        "New regressions:",
        str(comparison.new_regression_count),
        "",
        "Resolved:",
        str(comparison.resolved_count),
        "",
        "UNCHANGED:",
        str(comparison.unchanged_failure_count),
        "",
        "Removed known failures:",
        str(removed_known_failures),
        "",
        "FAIL -> INCONCLUSIVE:",
        str(fail_to_inconclusive),
    ]
    if uncertifiable:
        lines.extend(["", "Uncertifiable failures:", str(uncertifiable)])
    regressions = [
        item
        for item in comparison.items
        if item.blocking and item.category in BLOCKING_CATEGORIES
    ]
    if regressions:
        lines.append("")
        lines.append("New regression details:")
        for item in regressions:
            transition = _transition(item)
            lines.append(
                f"- {item.scenario_id} ({item.fingerprint}) {transition} [{item.category}]"
            )
    uncertifiable_items = [
        item
        for item in comparison.items
        if item.category == "uncertifiable_failure"
    ]
    if uncertifiable_items:
        lines.append("")
        lines.append("Uncertifiable failure details:")
        for item in uncertifiable_items:
            transition = _transition(item)
            lines.append(
                f"- {item.scenario_id} ({item.fingerprint}) {transition} [{item.category}]"
            )
    if removed_known_failures:
        lines.append("")
        lines.append(
            "Removed known failures do not block by default; they are not "
            "equivalent to a clean stable suite."
        )
    if fail_to_inconclusive:
        lines.append("")
        lines.append(
            "FAIL -> INCONCLUSIVE does not block; weak evidence is not a PASS."
        )
    return "\n".join(lines) + "\n"


def _comparison_items(
    baseline: EvaluationBaseline,
    current: EvaluationBaseline,
) -> tuple[ComparisonItem, ...]:
    baseline_by_fp = {item.fingerprint: item for item in baseline.cases}
    current_by_fp = {item.fingerprint: item for item in current.cases}
    used_baseline: set[str] = set()
    used_current: set[str] = set()
    items: list[ComparisonItem] = []

    for fingerprint in sorted(set(baseline_by_fp).intersection(current_by_fp)):
        left = baseline_by_fp[fingerprint]
        right = current_by_fp[fingerprint]
        used_baseline.add(fingerprint)
        used_current.add(fingerprint)
        items.append(_item(left, right, fingerprint_changed=False))

    leftover_baseline = {
        item.scenario_id: item
        for item in baseline.cases
        if item.fingerprint not in used_baseline
    }
    leftover_current = {
        item.scenario_id: item
        for item in current.cases
        if item.fingerprint not in used_current
    }
    for scenario_id in sorted(set(leftover_baseline).intersection(leftover_current)):
        left = leftover_baseline[scenario_id]
        right = leftover_current[scenario_id]
        used_baseline.add(left.fingerprint)
        used_current.add(right.fingerprint)
        items.append(_item(left, right, fingerprint_changed=True))

    for item in current.cases:
        if item.fingerprint in used_current:
            continue
        items.append(_item(None, item, fingerprint_changed=False))
    for item in baseline.cases:
        if item.fingerprint in used_baseline:
            continue
        items.append(_item(item, None, fingerprint_changed=False))

    items.sort(
        key=lambda item: (
            _CATEGORY_ORDER.index(item.category),
            item.fingerprint,
            item.scenario_id,
        )
    )
    return tuple(items)


def _item(
    baseline: BaselineCase | None,
    current: BaselineCase | None,
    *,
    fingerprint_changed: bool,
) -> ComparisonItem:
    category, blocking = _classify(
        baseline, current, fingerprint_changed=fingerprint_changed
    )
    fingerprint = (
        current.fingerprint
        if current is not None
        else baseline.fingerprint  # type: ignore[union-attr]
    )
    scenario_id = (
        current.scenario_id
        if current is not None
        else baseline.scenario_id  # type: ignore[union-attr]
    )
    return ComparisonItem(
        category=category,
        scenario_id=scenario_id,
        fingerprint=fingerprint,
        baseline_verdict=None if baseline is None else baseline.verdict,
        current_verdict=None if current is None else current.verdict,
        baseline_failure_fingerprint=_signature_fingerprint(baseline),
        current_failure_fingerprint=_signature_fingerprint(current),
        blocking=blocking,
    )


def _classify(
    baseline: BaselineCase | None,
    current: BaselineCase | None,
    *,
    fingerprint_changed: bool,
) -> tuple[ComparisonCategory, bool]:
    if current is not None and current.verdict == "INFRA_ERROR":
        return "infra_change", False
    if baseline is not None and baseline.verdict == "INFRA_ERROR" and current is not None:
        return "infra_change", False
    if current is None:
        return "removed_scenario", False

    current_sig = _signature_fingerprint(current)
    baseline_sig = _signature_fingerprint(baseline)
    if current.verdict == "FAIL":
        if baseline is None or baseline.verdict != "FAIL":
            return "new_regression", True
        if current_sig is not None and baseline_sig is not None:
            if baseline_sig == current_sig:
                category: ComparisonCategory = (
                    "changed_scenario" if fingerprint_changed else "unchanged_failure"
                )
                return category, False
            return "changed_failure", True
        return "uncertifiable_failure", False
    if baseline is not None and baseline.verdict == "FAIL" and current.verdict == "PASS":
        return "resolved_failure", False
    if current.verdict == "INCONCLUSIVE" or (
        baseline is not None and baseline.verdict == "INCONCLUSIVE"
    ):
        if baseline is None or current.verdict != baseline.verdict:
            return "inconclusive_change", False
    if baseline is None:
        return "new_scenario", False
    if fingerprint_changed:
        return "changed_scenario", False
    if current.verdict == baseline.verdict:
        return "unchanged", False
    return "inconclusive_change", False


def _signature_fingerprint(case: BaselineCase | None) -> str | None:
    if case is None or case.failure_signature is None:
        return None
    return case.failure_signature.fingerprint


def _failure_count(baseline: EvaluationBaseline) -> int:
    return sum(1 for item in baseline.cases if item.verdict == "FAIL")


def _count(
    items: tuple[ComparisonItem, ...],
    category: ComparisonCategory,
    *,
    blocking_only: bool = False,
) -> int:
    return sum(
        1
        for item in items
        if item.category == category and (not blocking_only or item.blocking)
    )


def _transition(item: ComparisonItem) -> str:
    left = item.baseline_verdict or "absent"
    right = item.current_verdict or "absent"
    return f"{left} -> {right}"


__all__ = [
    "compare_baselines",
    "format_comparison",
    "gate_exit_code",
]
