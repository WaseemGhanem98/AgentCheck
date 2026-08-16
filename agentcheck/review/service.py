"""Record human finding reviews without executing the target."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from agentcheck import __version__
from agentcheck.config import AgentCheckConfig
from agentcheck.domain import CaseEvaluation, Finding, Verdict, utc_now
from agentcheck.errors import ConfigurationError
from agentcheck.privacy import redact_log_text
from agentcheck.report.load import load_stored_run
from agentcheck.store import StoreError, open_evaluation_store

from .contract import (
    HumanDecision,
    HumanReview,
    ReviewSourceBinding,
    finding_fingerprint,
)
from .store import (
    findings_relative_path,
    hash_findings_file,
    screen_review_note,
    screen_reviewer_label,
    write_human_review,
)


@dataclass(frozen=True, slots=True)
class RecordedReview:
    review: HumanReview
    path: Path
    finding: Finding


def record_finding_review(
    target: str | Path,
    *,
    run_id: str,
    finding_id: str,
    decision: HumanDecision,
    note: str = "",
    reviewer: str | None = None,
) -> RecordedReview:
    """Append a human review bound to the current findings artifact."""

    loaded = load_stored_run(target, run_id=run_id)
    finding = _require_finding(loaded.findings, finding_id)
    _assert_finding_is_automated_fail(loaded.evaluations, finding)
    note = screen_review_note(note)
    reviewer = screen_reviewer_label(reviewer)
    findings_path = findings_relative_path(loaded.config, loaded.run_id)
    findings_file = loaded.run_directory / "findings.json"
    review = HumanReview(
        run_id=loaded.run_id,
        finding_id=finding.finding_id,
        finding_fingerprint=finding_fingerprint(finding),
        failure_signature=finding.failure_signature,
        automated_verdict="FAIL",
        decision=decision,
        note=note,
        reviewer=reviewer,
        recorded_at=utc_now(),
        source=ReviewSourceBinding(
            findings_path=findings_path,
            findings_digest=hash_findings_file(findings_file),
        ),
        agentcheck_version=__version__,
    )
    path = write_human_review(loaded.root, loaded.config, review)
    _index_review(loaded.root, loaded.config, review, path)
    return RecordedReview(review=review, path=path, finding=finding)


def _require_finding(findings: tuple[Finding, ...], finding_id: str) -> Finding:
    matches = [item for item in findings if item.finding_id == finding_id]
    if not matches:
        raise ConfigurationError(f"unknown finding {finding_id}")
    if len(matches) != 1:
        raise ConfigurationError(f"finding {finding_id} is not unique in findings.json")
    return matches[0]


def _assert_finding_is_automated_fail(
    evaluations: tuple[CaseEvaluation, ...], finding: Finding
) -> None:
    by_id = {item.scenario_id: item for item in evaluations}
    for scenario_id in finding.affected_scenario_ids:
        evaluation = by_id.get(scenario_id)
        if evaluation is None:
            raise ConfigurationError(
                f"finding {finding.finding_id} refers to missing evaluation {scenario_id}"
            )
        if evaluation.verdict != Verdict.FAIL:
            raise ConfigurationError(
                f"finding {finding.finding_id} is bound to a non-FAIL evaluation"
            )


def _index_review(
    root: Path,
    config: AgentCheckConfig,
    review: HumanReview,
    path: Path,
) -> None:
    try:
        store = open_evaluation_store(root, config)
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        store.record_review(
            review_id=review.review_id,
            run_id=review.run_id,
            finding_id=review.finding_id,
            finding_fingerprint=review.finding_fingerprint,
            decision=review.decision,
            recorded_at=review.recorded_at.isoformat(),
            artifact_path=relative,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except (StoreError, OSError, ValueError) as exc:
        print(
            "AgentCheck warning: evaluation store failed: "
            + redact_log_text(str(exc)),
            file=sys.stderr,
        )


__all__ = [
    "RecordedReview",
    "record_finding_review",
]
