from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pytest

from agentcheck.artifacts import ArtifactStore
from agentcheck.baseline.service import create_baseline
from agentcheck.domain import (
    AgentProperty,
    AgentSpec,
    CapabilitiesSpec,
    CanonicalRun,
    CaseEvaluation,
    IdentitySpec,
    InspectionProvenance,
    InstructionsSpec,
    InterfaceSpec,
    ObservabilitySpec,
    RunTermination,
    RuntimeSpec,
    Scenario,
    SourceKind,
    SourceReference,
    SpecEvidence,
    ToolsSpec,
    Verdict,
    utc_now,
)
from agentcheck.errors import ConfigurationError
from agentcheck.evaluate import evaluate_run, infrastructure_evaluation
from agentcheck.generate.templates import build_account_support_suite
from agentcheck.report import load_stored_run, render_stored_run
from agentcheck.report.load import _validate_execution_bindings


T = TypeVar("T")
SEED = 1729
RUN_ID = "stored-binding"


def _property(value: T) -> AgentProperty[T]:
    return AgentProperty(
        value=value,
        source=SourceReference(kind=SourceKind.RUNTIME_INTROSPECTION, locator="test"),
        confidence=1,
        evidence=(SpecEvidence(evidence_id="e", summary="test"),),
    )


def _spec() -> AgentSpec:
    return AgentSpec(
        spec_id="portable-spec",
        identity=IdentitySpec(
            name=_property("Stored Binding Agent"),
            framework=_property("custom"),
            framework_version=_property("1"),
            provider=_property(None),
            model=_property(None),
        ),
        interface=InterfaceSpec(
            entrypoint=_property("agent.py:agent"),
            input_modalities=_property(("text",)),
            output_modalities=_property(("text",)),
            input_schema=_property(None),
            output_schema=_property(None),
            interactive=_property(True),
        ),
        instructions=InstructionsSpec(
            system=_property("test"), developer=_property(None)
        ),
        capabilities=CapabilitiesSpec(),
        tools=ToolsSpec(),
        runtime=RuntimeSpec(
            max_model_turns=_property(None),
            max_tool_calls=_property(None),
            timeout_seconds=_property(None),
            token_budget=_property(None),
            cost_budget_usd=_property(None),
        ),
        observability=ObservabilitySpec(
            supported_event_types=_property(("final_output",)),
            usage_metrics=_property(()),
            provider_request_ids=_property(False),
            source_event_links=_property(True),
        ),
        provenance=InspectionProvenance(
            inspector="test",
            inspector_version="1",
            inspected_at=utc_now(),
            target="test",
            sources=(
                SourceReference(kind=SourceKind.RUNTIME_INTROSPECTION, locator="test"),
            ),
        ),
    )


@dataclass(frozen=True)
class StoredArtifacts:
    root: Path
    directory: Path
    spec: AgentSpec
    scenario: Scenario
    run: CanonicalRun
    evaluation: CaseEvaluation


def _write_stored_artifacts(tmp_path: Path) -> StoredArtifacts:
    root = tmp_path / "target"
    root.mkdir()
    spec = _spec()
    scenario_data = build_account_support_suite(seed=SEED)[0].model_dump(mode="json")
    for field in (
        "tool_fixtures",
        "injected_faults",
        "expected_postconditions",
        "required_tool_behavior",
        "allowed_tool_behavior",
        "forbidden_tool_behavior",
        "trajectory_constraints",
        "output_criteria",
    ):
        scenario_data[field] = []
    scenario_data["initial_world_state"] = {}
    scenario_data["fingerprint"] = ""
    scenario = Scenario.model_validate_json(json.dumps(scenario_data))
    now = utc_now()
    run = CanonicalRun(
        run_id=f"{RUN_ID}-case-001",
        scenario_id=scenario.scenario_id,
        target_id=spec.spec_id,
        started_at=now,
        ended_at=now,
        termination=RunTermination.COMPLETED,
        initial_world_state=scenario.initial_world_state,
        final_world_state=scenario.initial_world_state,
        final_output="No action was required.",
        latency_ms=0,
    )
    evaluation = evaluate_run(scenario, run)
    assert evaluation.verdict is Verdict.PASS
    assert all(
        assertion.supporting_evidence_ids
        for assertion in evaluation.assertions
        if assertion.required and assertion.result is Verdict.PASS
    )
    artifacts = ArtifactStore(root, ".agentcheck", RUN_ID)
    artifacts.write_json("agent-spec.json", spec)
    artifacts.write_json(
        "suite.json",
        {
            "schema_version": "agentcheck.suite.v1",
            "run_id": RUN_ID,
            "seed": SEED,
            "scenarios": (scenario,),
        },
    )
    artifacts.write_jsonl("runs.jsonl", (run,))
    artifacts.write_jsonl("evaluations.jsonl", (evaluation,))
    artifacts.write_json("findings.json", ())
    artifacts.write_json(
        "summary.json",
        {
            "schema_version": "agentcheck.summary.v1",
            "run_id": RUN_ID,
            "seed": SEED,
            "invalid_scenarios": 0,
        },
    )
    return StoredArtifacts(root, artifacts.root, spec, scenario, run, evaluation)


def _replace_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _replace_jsonl(path: Path, values: tuple[object, ...]) -> None:
    lines = [
        item.model_dump_json() if hasattr(item, "model_dump_json") else json.dumps(item)
        for item in values
    ]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _replace_suite(stored: StoredArtifacts, scenarios: tuple[Scenario, ...]) -> None:
    _replace_json(
        stored.directory / "suite.json",
        {
            "schema_version": "agentcheck.suite.v1",
            "run_id": RUN_ID,
            "seed": SEED,
            "scenarios": [scenario.model_dump(mode="json") for scenario in scenarios],
        },
    )


def test_empty_stored_execution_cannot_produce_loaded_run_report_or_baseline(
    tmp_path: Path,
) -> None:
    stored = _write_stored_artifacts(tmp_path)
    _replace_suite(stored, ())
    _replace_jsonl(stored.directory / "runs.jsonl", ())
    _replace_jsonl(stored.directory / "evaluations.jsonl", ())

    message = "suite.json must contain at least one scenario"
    with pytest.raises(ConfigurationError, match=message):
        load_stored_run(stored.root, run_id=RUN_ID)
    with pytest.raises(ConfigurationError, match=message):
        render_stored_run(stored.root, run_id=RUN_ID)
    with pytest.raises(ConfigurationError, match=message):
        create_baseline(stored.root, run_id=RUN_ID, out="baseline.json")

    assert not (stored.directory / "report.html").exists()
    assert not (stored.root / "baseline.json").exists()


def test_duplicate_suite_scenario_ids_are_rejected(tmp_path: Path) -> None:
    stored = _write_stored_artifacts(tmp_path)
    _replace_suite(stored, (stored.scenario, stored.scenario))

    with pytest.raises(ConfigurationError, match="scenario IDs must be unique"):
        load_stored_run(stored.root, run_id=RUN_ID)


def test_non_infrastructure_evaluation_requires_a_canonical_run(
    tmp_path: Path,
) -> None:
    stored = _write_stored_artifacts(tmp_path)
    _replace_jsonl(stored.directory / "runs.jsonl", ())

    with pytest.raises(ConfigurationError, match="no matching canonical run"):
        load_stored_run(stored.root, run_id=RUN_ID)


def test_baseline_cannot_trust_a_pass_without_a_canonical_run(
    tmp_path: Path,
) -> None:
    stored = _write_stored_artifacts(tmp_path)
    _replace_jsonl(stored.directory / "runs.jsonl", ())

    with pytest.raises(ConfigurationError, match="no matching canonical run"):
        create_baseline(stored.root, run_id=RUN_ID, out="baseline.json")

    assert not (stored.root / "baseline.json").exists()


def test_duplicate_evaluations_cannot_inflate_a_regenerated_report(
    tmp_path: Path,
) -> None:
    stored = _write_stored_artifacts(tmp_path)
    duplicate = stored.evaluation.model_copy(update={"evaluation_id": "eval-duplicate"})
    _replace_jsonl(
        stored.directory / "evaluations.jsonl", (stored.evaluation, duplicate)
    )

    with pytest.raises(ConfigurationError, match="more than one evaluation"):
        render_stored_run(stored.root, run_id=RUN_ID)


def test_duplicate_evaluation_ids_are_rejected(tmp_path: Path) -> None:
    stored = _write_stored_artifacts(tmp_path)
    _replace_jsonl(
        stored.directory / "evaluations.jsonl",
        (stored.evaluation, stored.evaluation),
    )

    with pytest.raises(ConfigurationError, match="evaluation IDs must be unique"):
        load_stored_run(stored.root, run_id=RUN_ID)


def test_suite_scenarios_cannot_disappear_from_evaluations(tmp_path: Path) -> None:
    stored = _write_stored_artifacts(tmp_path)
    _replace_jsonl(stored.directory / "evaluations.jsonl", ())

    with pytest.raises(ConfigurationError, match="missing evaluations"):
        load_stored_run(stored.root, run_id=RUN_ID)


def test_evaluations_cannot_add_scenarios_outside_the_suite(tmp_path: Path) -> None:
    stored = _write_stored_artifacts(tmp_path)
    extra = stored.evaluation.model_copy(
        update={
            "evaluation_id": "eval-extra",
            "scenario_id": "outside-suite",
            "run_id": "outside-run",
        }
    )
    _replace_jsonl(
        stored.directory / "evaluations.jsonl",
        (stored.evaluation, extra),
    )

    with pytest.raises(ConfigurationError, match="outside suite.json"):
        load_stored_run(stored.root, run_id=RUN_ID)


def test_evaluation_run_ids_are_unique(tmp_path: Path) -> None:
    stored = _write_stored_artifacts(tmp_path)
    duplicate_run_reference = stored.evaluation.model_copy(
        update={"evaluation_id": "eval-extra", "scenario_id": "outside-suite"}
    )
    _replace_jsonl(
        stored.directory / "evaluations.jsonl",
        (stored.evaluation, duplicate_run_reference),
    )

    with pytest.raises(ConfigurationError, match="evaluation run IDs must be unique"):
        load_stored_run(stored.root, run_id=RUN_ID)


def test_duplicate_canonical_run_ids_are_rejected(tmp_path: Path) -> None:
    stored = _write_stored_artifacts(tmp_path)
    _replace_jsonl(stored.directory / "runs.jsonl", (stored.run, stored.run))

    with pytest.raises(ConfigurationError, match="canonical run IDs must be unique"):
        load_stored_run(stored.root, run_id=RUN_ID)


def test_more_than_one_run_for_a_scenario_is_rejected(tmp_path: Path) -> None:
    stored = _write_stored_artifacts(tmp_path)
    duplicate = stored.run.model_copy(update={"run_id": f"{RUN_ID}-case-002"})
    _replace_jsonl(stored.directory / "runs.jsonl", (stored.run, duplicate))

    with pytest.raises(ConfigurationError, match="more than one canonical run"):
        load_stored_run(stored.root, run_id=RUN_ID)


def test_runs_cannot_add_scenarios_outside_the_suite(tmp_path: Path) -> None:
    stored = _write_stored_artifacts(tmp_path)
    outside = stored.run.model_copy(update={"scenario_id": "outside-suite"})
    _replace_jsonl(stored.directory / "runs.jsonl", (outside,))

    with pytest.raises(ConfigurationError, match="outside suite.json"):
        load_stored_run(stored.root, run_id=RUN_ID)


def test_evaluation_run_id_must_match_the_scenario_run(tmp_path: Path) -> None:
    stored = _write_stored_artifacts(tmp_path)
    mismatched = stored.evaluation.model_copy(update={"run_id": "missing-case-run"})
    _replace_jsonl(stored.directory / "evaluations.jsonl", (mismatched,))

    with pytest.raises(ConfigurationError, match="no matching canonical run"):
        load_stored_run(stored.root, run_id=RUN_ID)


def test_evaluation_run_ids_cannot_be_swapped_between_scenarios(
    tmp_path: Path,
) -> None:
    stored = _write_stored_artifacts(tmp_path)
    other_scenario = stored.scenario.model_copy(
        update={"scenario_id": "second-scenario"}
    )
    other_run = stored.run.model_copy(
        update={
            "run_id": f"{RUN_ID}-case-002",
            "scenario_id": other_scenario.scenario_id,
        }
    )
    first_evaluation = stored.evaluation.model_copy(update={"run_id": other_run.run_id})
    other_evaluation = stored.evaluation.model_copy(
        update={
            "evaluation_id": "eval-second-scenario",
            "scenario_id": other_scenario.scenario_id,
            "run_id": stored.run.run_id,
        }
    )

    with pytest.raises(ConfigurationError, match="references a different scenario"):
        _validate_execution_bindings(
            spec=stored.spec,
            scenarios=(stored.scenario, other_scenario),
            runs=(stored.run, other_run),
            evaluations=(first_evaluation, other_evaluation),
        )


def test_run_target_must_match_the_stored_spec(tmp_path: Path) -> None:
    stored = _write_stored_artifacts(tmp_path)
    mismatched = stored.run.model_copy(update={"target_id": "another-spec"})
    _replace_jsonl(stored.directory / "runs.jsonl", (mismatched,))

    with pytest.raises(ConfigurationError, match="target_id does not match"):
        load_stored_run(stored.root, run_id=RUN_ID)


@pytest.mark.parametrize("keep_run", [False, True])
def test_infrastructure_evaluation_may_have_zero_or_one_matching_run(
    tmp_path: Path, keep_run: bool
) -> None:
    stored = _write_stored_artifacts(tmp_path)
    evaluation = infrastructure_evaluation(
        stored.scenario,
        code="worker_error",
        message="No behavioral evidence was produced.",
        phase="execution",
        run_id=stored.run.run_id,
    )
    _replace_jsonl(stored.directory / "evaluations.jsonl", (evaluation,))
    if not keep_run:
        _replace_jsonl(stored.directory / "runs.jsonl", ())

    loaded = load_stored_run(stored.root, run_id=RUN_ID)

    assert loaded.evaluations == (evaluation,)
    assert loaded.runs == ((stored.run,) if keep_run else ())


def test_exact_legacy_target_identity_remains_a_valid_binding(tmp_path: Path) -> None:
    stored = _write_stored_artifacts(tmp_path)
    legacy_spec = stored.spec.model_copy(
        update={"spec_id": "legacy-spec", "legacy_spec_id": None}
    )
    legacy_run = stored.run.model_copy(update={"target_id": "legacy-spec"})
    _replace_json(
        stored.directory / "agent-spec.json",
        legacy_spec.model_dump(mode="json"),
    )
    _replace_jsonl(stored.directory / "runs.jsonl", (legacy_run,))

    loaded = load_stored_run(stored.root, run_id=RUN_ID)

    assert loaded.runs == (legacy_run,)


def test_hybrid_legacy_target_identity_does_not_rescue_a_mismatched_run(
    tmp_path: Path,
) -> None:
    stored = _write_stored_artifacts(tmp_path)
    hybrid_spec = stored.spec.model_copy(update={"legacy_spec_id": "legacy-spec"})
    hybrid_run = stored.run.model_copy(update={"target_id": "legacy-spec"})
    _replace_json(
        stored.directory / "agent-spec.json",
        hybrid_spec.model_dump(mode="json"),
    )
    _replace_jsonl(stored.directory / "runs.jsonl", (hybrid_run,))

    with pytest.raises(ConfigurationError, match="target_id does not match"):
        load_stored_run(stored.root, run_id=RUN_ID)
