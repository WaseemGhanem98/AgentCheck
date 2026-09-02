"""One command for the release question: does this change block the build?

`test` answers "what did the agent do". `baseline check` answers "is any of that
new". A release gate needs both, plus one thing neither says on its own: whether
the run is even certifiable. A suite that could not execute is not a passing
suite, and it is not a behavioural regression either -- reporting it as either
one is how a broken harness becomes a green build.

This orchestrates the existing operations and does not reimplement them. It runs
the frozen suite through the same `execute_suite`, compares against a trusted
baseline through the same `check_baseline`, and maps the result onto the exit
codes the CLI already documents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentcheck.application import SuiteExecution, execute_suite
from agentcheck.baseline.contract import DEFAULT_BASELINE_FILENAME
from agentcheck.baseline.service import check_baseline
from agentcheck.config import contained_path
from agentcheck.coverage import (
    BehavioralCoverage,
    BehavioralCoverageStatus,
    BehavioralDimension,
    risk_obligations_for_spec,
)
from agentcheck.domain import AgentSpec
from agentcheck.domain import Verdict


# The exit contract `test` already publishes, reused rather than reinvented.
EXIT_PASS = 0
EXIT_BEHAVIORAL_FAILURE = 1
EXIT_NOT_CERTIFIABLE = 2
EXIT_INCONCLUSIVE = 3


# The behavioural dimensions that exist *because* a tool carries risk. A
# declaration on the tool is what creates the obligation, so these are the only
# dimensions the evidence floor can speak for. `success_path`,
# `failure_handling` and `timeout_handling` are seeded for every declared tool
# regardless of risk; including them would make any uncovered tool a gate
# failure, which the accepted decision explicitly rejects.
_RISK_OBLIGATION_DIMENSIONS = frozenset(
    {
        BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
        BehavioralDimension.DUPLICATE_ACTION,
        BehavioralDimension.AMBIGUOUS_OUTCOME,
        BehavioralDimension.RETRY_CONTROL,
    }
)

# Keep one blocked gate readable in a terminal; the report holds the full set.
MAX_REPORTED_OBLIGATIONS = 10


@dataclass(frozen=True)
class UnmetRiskObligation:
    """One required behavioural evidence obligation the suite did not meet.

    ``subject`` is the coverage subject (``tool:<name>``), ``dimension`` the
    risk-scoped behaviour that a declared risk made mandatory, and
    ``reason_code`` the coverage analyzer's own account of what is absent.
    """

    subject: str
    dimension: str
    reason_code: str

    @property
    def tool_name(self) -> str:
        """The tool this obligation belongs to.

        Obligations are built from ``tool:<name>`` subjects the spec produced,
        so the prefix is always present; anything else is returned unchanged
        rather than mangled into a name that does not exist.
        """

        prefix, separator, name = self.subject.partition(":")
        return name if separator and prefix == "tool" else self.subject


TRUNCATED_DETAIL_REASON = "coverage_detail_unavailable"


def find_unmet_risk_obligations(
    spec: AgentSpec, coverage: BehavioralCoverage
) -> tuple[UnmetRiskObligation, ...]:
    """Obligations created by declared risk that the suite did not meet.

    Authority comes from ``spec.tool_risk``, never from a coverage status. A
    status cannot stand in for authority: ``_seed_from_scenarios`` can raise a
    risk dimension to APPLICABLE from scenario constraints alone, so a tool
    whose risk is only *inferred* can reach MISSING. Blocking on that would
    make inference authoritative and would tell the developer to correct a
    declaration they never wrote.

    Coverage supplies only the status of an obligation the spec already
    established. ``BehavioralCoverageFamily.requirements`` is a bounded detail
    view -- capped, and stripped of subjects that collide after redaction --
    while the counts beside it are complete. An obligation whose status is not
    in that view is therefore *unknown*, not satisfied, and is reported as
    unmet rather than silently dropped; treating an absent row as "fine" is the
    same false green this floor exists to close.
    """

    obligations = risk_obligations_for_spec(spec)
    if not obligations:
        return ()

    unmet: list[UnmetRiskObligation] = []
    for family in coverage.families:
        if family.dimension not in _RISK_OBLIGATION_DIMENSIONS:
            continue
        visible = {item.subject: item for item in family.requirements}
        for tool_name, dimensions in obligations.items():
            if family.dimension not in dimensions:
                continue
            subject = f"tool:{tool_name}"
            requirement = visible.get(subject)
            if requirement is None:
                # The spec says this is required and the report cannot show its
                # status. Refusing to certify is the only honest answer.
                unmet.append(
                    UnmetRiskObligation(
                        subject=subject,
                        dimension=family.dimension.value,
                        reason_code=TRUNCATED_DETAIL_REASON,
                    )
                )
            elif requirement.status is BehavioralCoverageStatus.MISSING:
                unmet.append(
                    UnmetRiskObligation(
                        subject=subject,
                        dimension=family.dimension.value,
                        reason_code=requirement.reason_code,
                    )
                )
    return tuple(sorted(unmet, key=lambda item: (item.subject, item.dimension)))


def _obligation_detail(
    obligations: tuple[UnmetRiskObligation, ...],
) -> tuple[str, ...]:
    """Say which tool, which requirement, why it blocks, and what to do next.

    Changing the exit code alone would tell a developer their build stopped
    without telling them what evidence is missing or how to supply it, so the
    detail lines carry all four.
    """

    tools = sorted({item.tool_name for item in obligations})
    lines = [
        f"{len(obligations)} required behavioural evidence obligation(s) are "
        f"unmet across {len(tools)} tool(s)",
    ]
    for item in obligations[:MAX_REPORTED_OBLIGATIONS]:
        lines.append(
            f"  {item.tool_name}: no {item.dimension} evidence "
            f"({item.reason_code})"
        )
    remaining = len(obligations) - MAX_REPORTED_OBLIGATIONS
    if remaining > 0:
        lines.append(f"  ... and {remaining} more; the report lists all of them")
    if any(item.reason_code == TRUNCATED_DETAIL_REASON for item in obligations):
        lines.append(
            "  (some statuses could not be read from the bounded coverage "
            "detail, so they are reported as unmet rather than assumed met)"
        )
    lines.append(
        "these are required because the tool's risk is declared, not inferred; "
        "a declaration is authoritative and cannot be satisfied by absence"
    )
    lines.append(
        "add cases that exercise the listed behaviours for those tools, or "
        "correct the tool_risk declaration in agentcheck.json if the tool is "
        "not actually state-changing or destructive"
    )
    return tuple(lines)


class GateDecision(str):
    """A gate outcome, readable in a log and stable in JSON."""

    PASS = "pass"
    BLOCK = "block"


@dataclass(frozen=True)
class GateResult:
    decision: str
    exit_code: int
    reason: str
    detail: tuple[str, ...]
    summary_block: str
    counts: dict[str, int]
    run_id: str
    suite_fingerprint: str | None
    baseline_path: str | None
    baseline_compared: bool
    report_path: str
    unmet_risk_obligations: tuple[UnmetRiskObligation, ...] = ()

    def to_json(self) -> str:
        return json.dumps(
            {
                "decision": self.decision,
                "exit_code": self.exit_code,
                "reason": self.reason,
                "detail": list(self.detail),
                "baseline_summary": self.summary_block,
                "counts": self.counts,
                "run_id": self.run_id,
                "suite_fingerprint": self.suite_fingerprint,
                "baseline": self.baseline_path,
                "baseline_compared": self.baseline_compared,
                "report": self.report_path,
                "unmet_risk_obligations": [
                    {
                        "tool": item.tool_name,
                        "subject": item.subject,
                        "dimension": item.dimension,
                        "reason_code": item.reason_code,
                    }
                    for item in self.unmet_risk_obligations
                ],
            },
            indent=2,
            sort_keys=True,
        )

    def render(self) -> str:
        head = "BLOCK" if self.decision == GateDecision.BLOCK else "PASS"
        lines = [f"{head}: {self.reason}", ""]
        counts = ", ".join(
            f"{name} {self.counts.get(name, 0)}"
            for name in ("pass", "fail", "inconclusive", "infra_error")
        )
        lines.append(f"  verdicts   {counts}")
        lines.append(f"  run        {self.run_id}")
        if self.suite_fingerprint:
            lines.append(f"  suite      {self.suite_fingerprint}")
        if self.unmet_risk_obligations:
            # A one-line summary only; the per-obligation rows live in `detail`,
            # which is rendered below, so they are not repeated here.
            tools = sorted({item.tool_name for item in self.unmet_risk_obligations})
            lines.append(
                f"  evidence   {len(self.unmet_risk_obligations)} required "
                f"obligation(s) unmet across {len(tools)} tool(s)"
            )
        lines.append(
            "  baseline   "
            + (
                f"{self.baseline_path} (compared)"
                if self.baseline_compared
                else f"{self.baseline_path or DEFAULT_BASELINE_FILENAME} (none recorded)"
            )
        )
        lines.append(f"  report     {self.report_path}")
        if self.detail:
            lines.append("")
            lines.extend(f"  - {item}" for item in self.detail)
        if self.summary_block:
            lines.append("")
            lines.extend(
                f"  {line}" for line in self.summary_block.rstrip().splitlines()
            )
        return "\n".join(lines) + "\n"


def gate_error_result(message: str) -> GateResult:
    """A gate answer when execution produced no certifiable result.

    The failure may occur before workers or after they finish but before the
    required evidence is complete. Either way there is no trusted run or
    report. The strongest answer available is "not certifiable", and a caller
    reading `--json` needs it as a document rather than as empty stdout.

    `counts` is deliberately empty rather than zeroed. Zeros would read as
    "nothing failed", which is a claim this run is in no position to make.
    """

    return GateResult(
        decision=GateDecision.BLOCK,
        exit_code=EXIT_NOT_CERTIFIABLE,
        reason=message,
        detail=(
            "no certifiable result was produced, so no behavioural claim is made",
            "this is not recorded as a behavioural regression",
        ),
        summary_block="",
        counts={},
        run_id="",
        suite_fingerprint=None,
        baseline_path=None,
        baseline_compared=False,
        report_path="",
    )


def _counts(execution: SuiteExecution) -> dict[str, int]:
    counter = execution.counts
    return {verdict.value.lower(): counter[verdict] for verdict in Verdict}


def _baseline_file(root: Path, baseline: str | None) -> Path | None:
    try:
        candidate = contained_path(root, baseline or DEFAULT_BASELINE_FILENAME)
    except Exception:
        return None
    return candidate if candidate.is_file() else None


def run_gate(
    target: str | Path,
    *,
    baseline: str | None = None,
    seed: int | None = None,
    run_id: str | None = None,
    persist_store: bool = True,
    python_executable: str | None = None,
    progress: Any | None = None,
    on_inspected: Any | None = None,
    on_prepared: Any | None = None,
) -> GateResult:
    """Execute the trusted suite, compare it to the baseline, and decide."""

    execution = execute_suite(
        target,
        seed=seed,
        run_id=run_id,
        persist_store=persist_store,
        python_executable=python_executable,
        progress=progress,
        on_inspected=on_inspected,
        on_prepared=on_prepared,
    )
    counts = _counts(execution)
    root = execution.target_root
    suite_fingerprint = (
        execution.frozen_suite.fingerprint if execution.frozen_suite else None
    )
    report = str(execution.report_path)
    # Evaluated once here, consulted only on the paths that would otherwise
    # return PASS. Certifiability and a real behavioural failure keep their own
    # more specific answers; this floor never overrides them.
    obligations = find_unmet_risk_obligations(
        execution.spec, execution.behavioral_coverage
    )

    # Certifiability first. An infrastructure error means the suite did not get
    # to say anything, so it is neither a pass nor a regression, and answering
    # the release question from it would be answering it from nothing.
    if counts.get("infra_error"):
        return GateResult(
            decision=GateDecision.BLOCK,
            exit_code=EXIT_NOT_CERTIFIABLE,
            reason="the run is not certifiable, so no behavioural claim is made",
            detail=(
                f"{counts['infra_error']} case(s) stopped on infrastructure, "
                "not on agent behaviour",
                "this is not recorded as a behavioural regression",
                "fix the harness or the fixtures, then re-run the gate",
            ),
            summary_block="",
            counts=counts,
            run_id=execution.run_id,
            suite_fingerprint=suite_fingerprint,
            baseline_path=None,
            baseline_compared=False,
            report_path=report,
            unmet_risk_obligations=obligations,
        )

    baseline_file = _baseline_file(root, baseline)
    if baseline_file is None:
        # Without a trusted baseline there is nothing to call a *regression*, so
        # the gate answers the weaker question it can actually answer.
        detail: list[str] = [
            "no trusted baseline was found, so nothing was compared",
            f"record one with: agentcheck baseline create {root}"
            " --latest --out " + DEFAULT_BASELINE_FILENAME,
        ]
        if counts.get("fail"):
            return GateResult(
                decision=GateDecision.BLOCK,
                exit_code=EXIT_BEHAVIORAL_FAILURE,
                reason="the suite recorded a behavioural failure",
                detail=tuple([f"{counts['fail']} failing case(s)", *detail]),
                summary_block="",
                counts=counts,
                run_id=execution.run_id,
                suite_fingerprint=suite_fingerprint,
                baseline_path=None,
                baseline_compared=False,
                report_path=report,
            )
        if counts.get("inconclusive"):
            return GateResult(
                decision=GateDecision.BLOCK,
                exit_code=EXIT_INCONCLUSIVE,
                reason="the suite could not decide, which is not a pass",
                detail=tuple(
                    [f"{counts['inconclusive']} inconclusive case(s)", *detail]
                ),
                summary_block="",
                counts=counts,
                run_id=execution.run_id,
                suite_fingerprint=suite_fingerprint,
                baseline_path=None,
                baseline_compared=False,
                report_path=report,
            )
        if obligations:
            return GateResult(
                decision=GateDecision.BLOCK,
                exit_code=EXIT_INCONCLUSIVE,
                reason=(
                    "required behavioural evidence is missing, which is not a pass"
                ),
                detail=tuple([*_obligation_detail(obligations), *detail]),
                summary_block="",
                counts=counts,
                run_id=execution.run_id,
                suite_fingerprint=suite_fingerprint,
                baseline_path=None,
                baseline_compared=False,
                report_path=report,
                unmet_risk_obligations=obligations,
            )
        return GateResult(
            decision=GateDecision.PASS,
            exit_code=EXIT_PASS,
            reason="every case passed",
            detail=tuple(detail),
            summary_block="",
            counts=counts,
            run_id=execution.run_id,
            suite_fingerprint=suite_fingerprint,
            baseline_path=None,
            baseline_compared=False,
            report_path=report,
            unmet_risk_obligations=obligations,
        )

    relative = str(baseline_file.relative_to(root))
    checked = check_baseline(
        root, baseline_path=relative, run_id=execution.run_id, latest=False
    )
    if checked.exit_code == EXIT_PASS and counts.get("inconclusive"):
        # ``baseline check`` asks only whether an authoritative failure is new,
        # so INCONCLUSIVE is deliberately non-blocking there.  The one-command
        # release gate has a stronger contract: missing evidence can never be
        # upgraded to PASS merely because a baseline had nothing to compare.
        return GateResult(
            decision=GateDecision.BLOCK,
            exit_code=EXIT_INCONCLUSIVE,
            reason="the suite could not decide, which is not a pass",
            detail=(
                f"{counts['inconclusive']} inconclusive case(s)",
                "the baseline found no new authoritative failure, but it cannot "
                "supply evidence the current run did not observe",
            ),
            summary_block=checked.summary,
            counts=counts,
            run_id=execution.run_id,
            suite_fingerprint=suite_fingerprint,
            baseline_path=relative,
            baseline_compared=True,
            report_path=report,
            unmet_risk_obligations=obligations,
        )
    if checked.exit_code == EXIT_PASS and obligations:
        # A baseline can accept known failures; it cannot supply evidence the
        # suite never gathered. Same principle the INCONCLUSIVE branch above
        # already applies, extended to a requirement that received no case.
        return GateResult(
            decision=GateDecision.BLOCK,
            exit_code=EXIT_INCONCLUSIVE,
            reason="required behavioural evidence is missing, which is not a pass",
            detail=_obligation_detail(obligations),
            summary_block=checked.summary,
            counts=counts,
            run_id=execution.run_id,
            suite_fingerprint=suite_fingerprint,
            baseline_path=relative,
            baseline_compared=True,
            report_path=report,
            unmet_risk_obligations=obligations,
        )
    if checked.exit_code == EXIT_PASS:
        return GateResult(
            decision=GateDecision.PASS,
            exit_code=EXIT_PASS,
            reason="no new behavioural failure against the trusted baseline",
            detail=(
                f"{counts.get('fail', 0)} known failure(s) accepted by the baseline",
            ),
            summary_block="",
            counts=counts,
            run_id=execution.run_id,
            suite_fingerprint=suite_fingerprint,
            baseline_path=relative,
            baseline_compared=True,
            report_path=report,
            unmet_risk_obligations=obligations,
        )

    reason = {
        EXIT_BEHAVIORAL_FAILURE: "a behavioural failure is new against the trusted baseline",
        EXIT_NOT_CERTIFIABLE: "the comparison could not be trusted",
        EXIT_INCONCLUSIVE: "the comparison could not decide, which is not a pass",
    }.get(checked.exit_code, "the baseline check refused this run")
    return GateResult(
        decision=GateDecision.BLOCK,
        exit_code=checked.exit_code,
        reason=reason,
        detail=(),
        summary_block=checked.summary,
        counts=counts,
        run_id=execution.run_id,
        suite_fingerprint=suite_fingerprint,
        baseline_path=relative,
        baseline_compared=True,
        report_path=report,
        unmet_risk_obligations=obligations,
    )


__all__ = [
    "EXIT_BEHAVIORAL_FAILURE",
    "UnmetRiskObligation",
    "EXIT_INCONCLUSIVE",
    "EXIT_NOT_CERTIFIABLE",
    "EXIT_PASS",
    "GateDecision",
    "GateResult",
    "TRUNCATED_DETAIL_REASON",
    "find_unmet_risk_obligations",
    "gate_error_result",
    "run_gate",
]
