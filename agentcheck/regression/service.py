"""Load two stored runs and report their behavioral differences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentcheck.errors import ConfigurationError
from agentcheck.report.load import load_stored_run

from .compare import compare_runs, exit_code
from .contract import (
    ComparabilityCaveat,
    IncomparableReason,
    RunComparison,
)


# One sentence per disclosed binding change. Each says what the reader may no
# longer conclude, not merely that something differed.
_CAVEAT_TEXT: dict[ComparabilityCaveat, str] = {
    ComparabilityCaveat.SPEC_CHANGED: (
        "The inspected agent specification changed between these runs, so a "
        "verdict difference may describe a different agent rather than a "
        "behavioral regression in the same one."
    ),
    ComparabilityCaveat.SCENARIO_SET_CHANGED: (
        "The two runs evaluated different scenario sets, so added and removed "
        "scenarios are expected and are not by themselves evidence of lost "
        "coverage."
    ),
    ComparabilityCaveat.SUITE_FINGERPRINT_CHANGED: (
        "Both runs recorded a frozen suite fingerprint and they differ, so the "
        "suite itself changed rather than only which cases ran."
    ),
    ComparabilityCaveat.SEED_CHANGED: (
        "The generation seed changed, so scenario identities may differ for "
        "reasons unrelated to the agent."
    ),
    ComparabilityCaveat.SELECTION_ACTIVE: (
        "Coverage selection pruned at least one run, so an absent scenario may "
        "have been excluded before execution rather than removed from the suite."
    ),
    ComparabilityCaveat.SOURCE_REVISION_UNRECORDED: (
        "At least one run recorded no source revision, so a difference cannot "
        "be attributed to a source change."
    ),
    ComparabilityCaveat.SOURCE_REVISION_UNCHANGED: (
        "Both runs recorded the same source revision, so any difference is "
        "run-to-run variation of a stochastic execution, not a source change."
    ),
    ComparabilityCaveat.SCENARIO_IDENTITY_CHANGED: (
        "At least one scenario matched by display ID with a different "
        "structural fingerprint, so its stimulus or oracle changed and its "
        "verdicts describe different questions."
    ),
}

_INCOMPARABLE_TEXT: dict[IncomparableReason, str] = {
    IncomparableReason.SAME_RUN: (
        "Both arguments resolved to the same stored run, so there is nothing "
        "to compare."
    ),
    IncomparableReason.NO_SHARED_SCENARIOS: (
        "The two runs share no scenario identity, so every classification "
        "would describe the identity change rather than a behavior."
    ),
}

# Printed on every comparison, not only on a suspicious one.
_STOCHASTIC_NOTE = (
    "A verdict pair is evidence about these two executions. AgentCheck "
    "reproduces inputs and harness behavior, not model output, so an unchanged "
    "verdict is not proof that a behavior is stable and a changed verdict is "
    "not proof that the source caused it. Repeat runs to establish stability."
)


@dataclass(frozen=True, slots=True)
class CheckedRunComparison:
    comparison: RunComparison
    exit_code: int
    summary: str


def compare_stored_runs(
    target: str | Path,
    *,
    base_run_id: str,
    head_run_id: str | None = None,
    latest: bool = False,
) -> CheckedRunComparison:
    """Compare two stored runs of one target without executing anything."""

    if head_run_id and latest:
        raise ConfigurationError("pass only one of --head and --latest")
    if not head_run_id and not latest:
        raise ConfigurationError("pass one of --head or --latest")
    base = load_stored_run(target, run_id=base_run_id)
    head = load_stored_run(target, run_id=head_run_id, latest=latest)
    comparison = compare_runs(base, head)
    return CheckedRunComparison(
        comparison=comparison,
        exit_code=exit_code(comparison),
        summary=format_run_comparison(comparison),
    )


def encode_run_comparison(comparison: RunComparison) -> str:
    return json.dumps(
        comparison.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )


def format_run_comparison(comparison: RunComparison) -> str:
    lines = [
        f"Base run: {comparison.base_run_id}",
        f"Head run: {comparison.head_run_id}",
        "",
        f"Base: {comparison.base_scenario_count} scenarios, "
        f"{comparison.base_failure_count} failures",
        f"Head: {comparison.head_scenario_count} scenarios, "
        f"{comparison.head_failure_count} failures",
        f"Shared scenario identities: {comparison.shared_scenario_count}",
    ]
    reason = comparison.incomparable_reason
    if reason is not None:
        lines.extend(
            [
                "",
                "Incomparable: no scenario was classified.",
                _INCOMPARABLE_TEXT[reason],
            ]
        )
        lines.extend(_caveat_lines(comparison))
        lines.extend(["", _STOCHASTIC_NOTE])
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "",
            f"New regressions:     {comparison.new_regression_count}",
            f"Changed failures:    {comparison.changed_failure_count}",
            f"Resolved:            {comparison.resolved_count}",
            f"Unchanged failures:  {comparison.unchanged_failure_count}",
            f"Uncertifiable:       {comparison.uncertifiable_count}",
        ]
    )
    blocking = [item for item in comparison.items if item.blocking]
    if blocking:
        lines.extend(["", "Attributable regressions:"])
        lines.extend(
            f"- {item.scenario_id} ({item.fingerprint}) "
            f"{item.base_verdict or 'absent'} -> {item.head_verdict or 'absent'} "
            f"[{item.category}]"
            for item in blocking
        )
    uncertifiable = [
        item
        for item in comparison.items
        if item.category in {"infra_change", "uncertifiable_failure"}
    ]
    if uncertifiable:
        lines.extend(
            [
                "",
                "Outcomes AgentCheck could not certify:",
            ]
        )
        lines.extend(
            f"- {item.scenario_id} ({item.fingerprint}) "
            f"{item.base_verdict or 'absent'} -> {item.head_verdict or 'absent'} "
            f"[{item.category}]"
            for item in uncertifiable
        )
    deselected = [
        item for item in comparison.items if item.category == "deselected_scenario"
    ]
    if deselected:
        lines.extend(
            [
                "",
                f"Excluded before execution in the head run: {len(deselected)}",
                "These scenarios were never evaluated, so this run holds no "
                "evidence about them either way.",
            ]
        )
    if comparison.omitted_items:
        lines.extend(
            [
                "",
                f"{comparison.omitted_items} further scenario detail(s) omitted "
                "from the bounded comparison record; the counts above are complete.",
            ]
        )
    lines.extend(_caveat_lines(comparison))
    lines.extend(["", _STOCHASTIC_NOTE])
    return "\n".join(lines) + "\n"


def _caveat_lines(comparison: RunComparison) -> list[str]:
    if not comparison.caveats:
        return []
    lines = ["", "Comparability caveats:"]
    lines.extend(
        f"- {caveat.value}: {_CAVEAT_TEXT[caveat]}" for caveat in comparison.caveats
    )
    return lines


__all__ = [
    "CheckedRunComparison",
    "compare_stored_runs",
    "encode_run_comparison",
    "format_run_comparison",
]
