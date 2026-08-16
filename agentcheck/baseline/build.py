"""Build an evaluation baseline from stored JSON/JSONL artifacts."""

from __future__ import annotations

from agentcheck import __version__
from agentcheck.domain import CaseEvaluation, Finding, Verdict
from agentcheck.errors import ConfigurationError
from agentcheck.report.load import LoadedRun
from agentcheck.shrink.signature import extract_failure_signature

from .contract import BaselineCase, BaselineFinding, EvaluationBaseline


def baseline_from_loaded(loaded: LoadedRun) -> EvaluationBaseline:
    """Snapshot automated results. Human reviews are excluded."""

    evaluation_by_id = _evaluations_by_scenario(loaded)
    cases: list[BaselineCase] = []
    for scenario in loaded.scenarios:
        evaluation = evaluation_by_id.get(scenario.scenario_id)
        if evaluation is None:
            raise ConfigurationError(
                f"stored run is missing an evaluation for scenario {scenario.scenario_id!r}"
            )
        cases.append(
            _case_from_evaluation(
                scenario.scenario_id, scenario.fingerprint, evaluation
            )
        )
    cases.sort(key=lambda item: (item.fingerprint, item.scenario_id))
    policy_pack_ids = _policy_pack_ids(loaded)
    selection_algorithm, selected_fingerprints = _selection(loaded)
    suite_id, suite_fingerprint = _suite_identity(loaded)
    return EvaluationBaseline(
        created_from_run_id=loaded.run_id,
        spec_id=loaded.spec.spec_id,
        suite_id=suite_id,
        suite_fingerprint=suite_fingerprint,
        seed=loaded.seed,
        policy_pack_ids=policy_pack_ids,
        selection_algorithm=selection_algorithm,
        selected_fingerprints=selected_fingerprints,
        cases=tuple(cases),
        findings=_findings(loaded.findings),
        agentcheck_version=__version__,
    )


def _evaluations_by_scenario(loaded: LoadedRun) -> dict[str, CaseEvaluation]:
    by_id: dict[str, CaseEvaluation] = {}
    for evaluation in loaded.evaluations:
        previous = by_id.get(evaluation.scenario_id)
        if previous is not None:
            raise ConfigurationError(
                f"stored run has duplicate evaluations for {evaluation.scenario_id!r}"
            )
        by_id[evaluation.scenario_id] = evaluation
    expected = {scenario.scenario_id for scenario in loaded.scenarios}
    extra = set(by_id).difference(expected)
    if extra:
        raise ConfigurationError(
            "stored run has evaluations for scenarios that are not in suite.json: "
            + ", ".join(sorted(extra))
        )
    return by_id


def _case_from_evaluation(
    scenario_id: str,
    fingerprint: str,
    evaluation: CaseEvaluation,
) -> BaselineCase:
    # Shrink identity is optional on a baseline case. Absence must never make
    # an authoritative FAIL look like a non-regression in comparison.
    signature = None
    if evaluation.verdict == Verdict.FAIL:
        try:
            signature = extract_failure_signature(evaluation)
        except ValueError:
            signature = None
    return BaselineCase(
        scenario_id=scenario_id,
        fingerprint=fingerprint,
        verdict=evaluation.verdict.value,
        failure_signature=signature,
    )


def _findings(findings: tuple[Finding, ...]) -> tuple[BaselineFinding, ...]:
    items = [
        BaselineFinding(
            finding_id=item.finding_id,
            finding_fingerprint=item.content_hash(),
            failure_signature=item.failure_signature,
        )
        for item in findings
    ]
    items.sort(key=lambda item: item.finding_id)
    return tuple(items)


def _policy_pack_ids(loaded: LoadedRun) -> tuple[str, ...]:
    if loaded.frozen_suite is not None:
        return tuple(sorted(set(loaded.frozen_suite.provenance.policy_packs)))
    return tuple(
        sorted({item.value.policy_id for item in loaded.spec.policies.items})
    )


def _suite_identity(loaded: LoadedRun) -> tuple[str | None, str | None]:
    frozen = loaded.frozen_suite
    if frozen is None:
        return None, None
    return frozen.suite_id, frozen.fingerprint


def _selection(loaded: LoadedRun) -> tuple[str | None, tuple[str, ...]]:
    if loaded.selection is None:
        return None, ()
    fingerprints = tuple(sorted(item.fingerprint for item in loaded.scenarios))
    return loaded.selection.algorithm, fingerprints


__all__ = ["baseline_from_loaded"]
