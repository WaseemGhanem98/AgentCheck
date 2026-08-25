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
from agentcheck.domain import Verdict


# The exit contract `test` already publishes, reused rather than reinvented.
EXIT_PASS = 0
EXIT_BEHAVIORAL_FAILURE = 1
EXIT_NOT_CERTIFIABLE = 2
EXIT_INCONCLUSIVE = 3


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
    """A gate answer for a run that never started.

    A suite that failed to load or a target that no longer matches it stops
    before any scenario executes, so there is no run and no report. That is
    still an answer to the release question -- the strongest one available is
    "not certifiable" -- and a caller reading `--json` needs it as a document
    rather than as an empty stdout it has to parse.

    `counts` is deliberately empty rather than zeroed. Zeros would read as
    "nothing failed", which is a claim this run is in no position to make.
    """

    return GateResult(
        decision=GateDecision.BLOCK,
        exit_code=EXIT_NOT_CERTIFIABLE,
        reason=message,
        detail=(
            "the suite did not execute, so no behavioural claim is made",
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
        )

    relative = str(baseline_file.relative_to(root))
    checked = check_baseline(
        root, baseline_path=relative, run_id=execution.run_id, latest=False
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
    )


__all__ = [
    "EXIT_BEHAVIORAL_FAILURE",
    "EXIT_INCONCLUSIVE",
    "EXIT_NOT_CERTIFIABLE",
    "EXIT_PASS",
    "GateDecision",
    "GateResult",
    "gate_error_result",
    "run_gate",
]
