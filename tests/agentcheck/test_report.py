from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import TypeVar

import pytest

from agentcheck.artifacts import ArtifactStore
from agentcheck.coverage import (
    BehavioralCoverage,
    BehavioralCoverageReferenceScope,
    analyze_behavioral_coverage,
)
from agentcheck.cli import main
from agentcheck.domain import (
    AgentProperty,
    AgentSpec,
    AssertionResult,
    CapabilitiesSpec,
    CanonicalRun,
    CaseEvaluation,
    ConversationRole,
    ConversationTurn,
    IdentitySpec,
    InspectionProvenance,
    InstructionsSpec,
    InterfaceSpec,
    ObservabilitySpec,
    OracleProvenance,
    OracleStrength,
    Policy,
    PoliciesSpec,
    RunTermination,
    RuntimeSpec,
    Scenario,
    SourceKind,
    SourceReference,
    SpecEvidence,
    ToolsSpec,
    ToolBehaviorConstraint,
    UsageMetrics,
    Verdict,
    utc_now,
)
from agentcheck.errors import ConfigurationError
from agentcheck.generate import (
    CaseLineage,
    CaseOrigin,
    FrozenCase,
    FrozenSuite,
    build_account_support_suite,
)
from agentcheck.report.render import _behavioral_coverage_html
from agentcheck.generate.suite import GeneratorProvenance, SuiteCoverage
from agentcheck.generate.selection import SelectionDecision, SelectionPlan
from agentcheck.report import load_stored_run, render_report, render_stored_run
from agentcheck.store import SqliteEvaluationStore, StoredRun


T = TypeVar("T")
REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"
SEED = 1729


def _property(value: T) -> AgentProperty[T]:
    return AgentProperty(value=value, source=SourceReference(kind=SourceKind.RUNTIME_INTROSPECTION, locator="test"), confidence=1, evidence=(SpecEvidence(evidence_id="e", summary="test"),))


def _spec(hostile: str, *, system_prompt: str = "TOP SECRET SYSTEM PROMPT") -> AgentSpec:
    return AgentSpec(
        spec_id="spec",
        identity=IdentitySpec(name=_property(hostile), framework=_property("OpenAI Agents SDK"), framework_version=_property("0.20.0"), provider=_property(None), model=_property(None)),
        interface=InterfaceSpec(entrypoint=_property("agent.py:agent"), input_modalities=_property(("text",)), output_modalities=_property(("text",)), input_schema=_property(None), output_schema=_property(None), interactive=_property(True)),
        instructions=InstructionsSpec(system=_property(system_prompt), developer=_property(None)),
        capabilities=CapabilitiesSpec(),
        tools=ToolsSpec(),
        runtime=RuntimeSpec(max_model_turns=_property(None), max_tool_calls=_property(None), timeout_seconds=_property(None), token_budget=_property(None), cost_budget_usd=_property(None)),
        observability=ObservabilitySpec(supported_event_types=_property(("final_output",)), usage_metrics=_property(()), provider_request_ids=_property(False), source_event_links=_property(True)),
        provenance=InspectionProvenance(inspector="test", inspector_version="1", inspected_at=utc_now(), target="test", sources=(SourceReference(kind=SourceKind.RUNTIME_INTROSPECTION, locator="test"),)),
    )


def test_report_escapes_xss_and_hides_system_prompt() -> None:
    hostile = '</pre><script>alert("x")</script><img src=x onerror=alert(1)>'

    report = render_report(run_id="run", target=hostile, git_revision=None, spec=_spec(hostile), scenarios=(), runs=(), evaluations=(), findings=())

    assert hostile not in report
    assert "&lt;script&gt;" in report
    assert "TOP SECRET SYSTEM PROMPT" not in report
    assert "<script" not in report.casefold()
    assert "https://" not in report
    assert "Content-Security-Policy" in report


def test_declared_behavioral_coverage_suppresses_all_zero_families() -> None:
    report = render_report(
        run_id="run",
        target="target",
        git_revision=None,
        spec=_spec("agent"),
        scenarios=(),
        runs=(),
        evaluations=(),
        findings=(),
    )

    assert "Declared behavioral coverage" in report
    assert "No declared behavioral coverage requirements were derived." in report
    assert "<h4>Success path</h4>" not in report
    assert "Unknown and unsupported requirements remain explicit." in report


def test_declared_behavioral_coverage_is_escaped_and_avoids_a_percentage() -> None:
    hostile = '<script>coverage</script><img src=x onerror="alert(1)">'
    spec = _spec("agent")
    payload = analyze_behavioral_coverage(spec, ()).model_dump(mode="json")
    payload["families"] = [
        {
            "dimension": "failure_handling",
            "missing": 1,
            "requirements": [
                {
                    "subject": hostile,
                    "status": "missing",
                    "reason_code": hostile,
                }
            ],
        }
    ]
    payload.pop("fingerprint")
    coverage = BehavioralCoverage.model_validate_json(json.dumps(payload))

    subsection = _behavioral_coverage_html(coverage)
    assert hostile not in subsection
    assert "&lt;script&gt;coverage&lt;/script&gt;" in subsection
    assert "Applicable: 1" in subsection
    assert "Missing: 1" in subsection
    assert "%" not in subsection
    assert "does not prove that the agent exercised an action path" in subsection
    assert "complete coverage of real-world behavioral risks" in subsection


def test_render_report_rejects_coverage_bound_to_another_spec() -> None:
    spec = _spec("agent")
    coverage = analyze_behavioral_coverage(spec, ()).model_copy(
        update={"spec_id": "different-spec"}
    )

    with pytest.raises(ValueError):
        render_report(
            run_id="run",
            target="target",
            git_revision=None,
            spec=spec,
            scenarios=(),
            runs=(),
            evaluations=(),
            findings=(),
            behavioral_coverage=coverage,
        )


def test_report_redacts_secrets_in_instructions_and_structured_state() -> None:
    scenario = build_account_support_suite()[0].model_copy(
        update={
            "initial_world_state": {
                "password": "world-secret-value",
                "message": "Bearer report-secret-token",
            }
        }
    )
    report = render_report(
        run_id="run",
        target="api_key=sk-targetsecret12345",
        git_revision=None,
        spec=_spec(
            "safe",
            system_prompt="api_key=sk-systemsecret12345",
        ),
        scenarios=(scenario,),
        runs=(),
        evaluations=(),
        findings=(),
        include_instructions=True,
    )

    assert "targetsecret" not in report
    assert "systemsecret" not in report
    assert "world-secret-value" not in report
    assert "report-secret-token" not in report
    assert "[REDACTED]" in report


def test_report_does_not_present_partial_usage_as_a_complete_total() -> None:
    now = utc_now()
    known = CanonicalRun(
        run_id="known",
        scenario_id="case-1",
        target_id="spec",
        started_at=now,
        ended_at=now,
        termination=RunTermination.COMPLETED,
        usage=UsageMetrics(total_tokens=12, cost_usd=0.25),
    )
    unknown = CanonicalRun(
        run_id="unknown",
        scenario_id="case-2",
        target_id="spec",
        started_at=now,
        ended_at=now,
        termination=RunTermination.COMPLETED,
    )

    report = render_report(
        run_id="run",
        target="target",
        git_revision=None,
        spec=_spec("agent"),
        scenarios=(),
        runs=(known, unknown),
        evaluations=(),
        findings=(),
    )

    assert report.count("Unknown (1/2 runs reported)") == 2
    assert ">12<" not in report
    assert ">0.25 USD<" not in report


def _copy_example(tmp_path: Path) -> Path:
    target = tmp_path / "account_agent"
    shutil.copytree(
        EXAMPLE,
        target,
        ignore=shutil.ignore_patterns(".agentcheck", "__pycache__"),
    )
    return target


def _pass_evaluation(scenario_id: str, run_id: str) -> CaseEvaluation:
    now = utc_now()
    return CaseEvaluation(
        evaluation_id=f"eval-{scenario_id}",
        scenario_id=scenario_id,
        run_id=run_id,
        verdict=Verdict.PASS,
        assertions=(
            AssertionResult(
                assertion_id="a1",
                criterion="completed",
                result=Verdict.PASS,
                oracle_ids=("oracle-1",),
                rationale="The stored fixture passed.",
            ),
        ),
        started_at=now,
        completed_at=now,
        summary="passed",
    )


def _write_run(
    root: Path,
    run_id: str,
    *,
    spec: AgentSpec | None = None,
    scenario: object | None = None,
    seed: int = SEED,
) -> Path:
    spec = spec or _spec("Account Support Agent")
    chosen = scenario or build_account_support_suite(seed=seed)[0]
    run = CanonicalRun(
        run_id=f"{run_id}-case-001",
        scenario_id=chosen.scenario_id,
        target_id=spec.spec_id,
        started_at=utc_now(),
        ended_at=utc_now(),
        termination=RunTermination.COMPLETED,
    )
    evaluation = _pass_evaluation(chosen.scenario_id, run.run_id)
    artifacts = ArtifactStore(root, ".agentcheck", run_id)
    artifacts.write_json("agent-spec.json", spec)
    artifacts.write_json(
        "suite.json",
        {
            "schema_version": "agentcheck.suite.v1",
            "run_id": run_id,
            "seed": seed,
            "scenarios": [chosen],
        },
    )
    artifacts.write_json(
        "invalid-scenarios.json",
        {"schema_version": "agentcheck.invalid_scenarios.v1", "items": []},
    )
    artifacts.write_jsonl("runs.jsonl", (run,))
    artifacts.write_jsonl("evaluations.jsonl", (evaluation,))
    artifacts.write_json("findings.json", ())
    artifacts.write_json(
        "summary.json",
        {
            "schema_version": "agentcheck.summary.v1",
            "run_id": run_id,
            "target": str(root),
            "git_revision": None,
            "suite_size": 1,
            "invalid_scenarios": 0,
            "observed_suite_pass_rate": 1.0,
            "counts": {
                "PASS": 1,
                "FAIL": 0,
                "INCONCLUSIVE": 0,
                "INFRA_ERROR": 0,
            },
            "finding_count": 0,
            "seed": seed,
        },
    )
    artifacts.write_text("report.html", "<html><body>placeholder</body></html>")
    return artifacts.root


def _selection_plan(selected_id: str, excluded_id: str) -> SelectionPlan:
    return SelectionPlan(
        selected_ids=(selected_id,),
        excluded_ids=(excluded_id,),
        decisions=(
            SelectionDecision(
                scenario_id=selected_id,
                selected=True,
                reason="selected for coverage",
            ),
            SelectionDecision(
                scenario_id=excluded_id,
                selected=False,
                reason="excluded by coverage selection",
            ),
        ),
    )


def test_old_summary_derives_behavioral_coverage_from_stored_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "target"
    root.mkdir()
    _write_run(root, "run-old-summary")

    loaded = load_stored_run(root, run_id="run-old-summary")

    assert loaded.behavioral_coverage == analyze_behavioral_coverage(
        loaded.spec, loaded.scenarios
    )
    generated = render_stored_run(root, run_id="run-old-summary")
    report = generated.report_path.read_text(encoding="utf-8")
    assert "Declared behavioral coverage" in report
    assert "structural suite-design analysis" in report


def test_old_selected_summary_marks_available_denominator_incomplete(
    tmp_path: Path,
) -> None:
    root = tmp_path / "target"
    root.mkdir()
    reference = build_account_support_suite(seed=SEED)[:2]
    directory = _write_run(
        root,
        "run-old-selected-summary",
        scenario=reference[0],
    )
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["selection"] = _selection_plan(
        reference[0].scenario_id, reference[1].scenario_id
    ).model_dump(mode="json")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    loaded = load_stored_run(root, run_id="run-old-selected-summary")

    assert (
        loaded.behavioral_coverage.reference_scope
        is BehavioralCoverageReferenceScope.AVAILABLE_SCENARIOS_ONLY
    )
    generated = render_stored_run(root, run_id="run-old-selected-summary")
    report = generated.report_path.read_text(encoding="utf-8")
    assert "Incomplete denominator" in report
    assert "reference scope is available scenarios only" in report
    assert "may undercount missing risks" in report

    summary["selection"] = _selection_plan(
        reference[1].scenario_id, reference[0].scenario_id
    ).model_dump(mode="json")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="selection binding"):
        load_stored_run(root, run_id="run-old-selected-summary")


def test_stored_complete_coverage_is_not_downgraded_by_selection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "target"
    root.mkdir()
    spec = _spec("Account Support Agent")
    reference = build_account_support_suite(seed=SEED)[:2]
    selected = (reference[0],)
    directory = _write_run(
        root,
        "run-complete-selected-summary",
        spec=spec,
        scenario=selected[0],
    )
    coverage = analyze_behavioral_coverage(
        spec,
        selected,
        reference_scenarios=reference,
    )
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["selection"] = _selection_plan(
        reference[0].scenario_id, reference[1].scenario_id
    ).model_dump(mode="json")
    summary["behavioral_coverage"] = coverage.model_dump(mode="json")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    loaded = load_stored_run(root, run_id="run-complete-selected-summary")

    assert (
        loaded.behavioral_coverage.reference_scope
        is BehavioralCoverageReferenceScope.COMPLETE
    )
    generated = render_stored_run(root, run_id="run-complete-selected-summary")
    report = generated.report_path.read_text(encoding="utf-8")
    assert "Incomplete denominator" not in report
    assert "tool:update_email" in report

    coverage_document = summary["behavioral_coverage"]
    coverage_document["reference_scenario_count"] = 1
    coverage_document["reference_scenario_digest"] = coverage_document[
        "scenario_digest"
    ]
    coverage_document.pop("fingerprint")
    rebound = BehavioralCoverage.model_validate_json(json.dumps(coverage_document))
    summary["behavioral_coverage"] = rebound.model_dump(mode="json")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="behavioral coverage|selection binding"):
        load_stored_run(root, run_id="run-complete-selected-summary")


def test_stored_summary_rejects_explicit_null_behavioral_coverage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "target"
    root.mkdir()
    directory = _write_run(root, "run-null-coverage")
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["behavioral_coverage"] = None
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="behavioral coverage"):
        load_stored_run(root, run_id="run-null-coverage")


def test_stored_summary_loads_bound_behavioral_coverage(tmp_path: Path) -> None:
    root = tmp_path / "target"
    root.mkdir()
    directory = _write_run(root, "run-with-coverage")
    legacy = load_stored_run(root, run_id="run-with-coverage")
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["behavioral_coverage"] = legacy.behavioral_coverage.model_dump(mode="json")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    loaded = load_stored_run(root, run_id="run-with-coverage")

    assert loaded.behavioral_coverage == legacy.behavioral_coverage


def test_stored_summary_rejects_behavioral_coverage_bound_to_another_spec(
    tmp_path: Path,
) -> None:
    root = tmp_path / "target"
    root.mkdir()
    directory = _write_run(root, "run-mismatched-coverage")
    legacy = load_stored_run(root, run_id="run-mismatched-coverage")
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["behavioral_coverage"] = legacy.behavioral_coverage.model_dump(mode="json")
    summary["behavioral_coverage"]["spec_id"] = "different-spec"
    summary["behavioral_coverage"].pop("fingerprint")
    rebound = BehavioralCoverage.model_validate_json(
        json.dumps(summary["behavioral_coverage"])
    )
    summary["behavioral_coverage"] = rebound.model_dump(mode="json")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="behavioral coverage"):
        load_stored_run(root, run_id="run-mismatched-coverage")


def test_stored_summary_rejects_behavioral_coverage_bound_to_other_scenarios(
    tmp_path: Path,
) -> None:
    root = tmp_path / "target"
    root.mkdir()
    directory = _write_run(root, "run-mismatched-scenarios")
    legacy = load_stored_run(root, run_id="run-mismatched-scenarios")
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["behavioral_coverage"] = legacy.behavioral_coverage.model_dump(mode="json")
    summary["behavioral_coverage"]["scenario_digest"] = "sha256:different"
    summary["behavioral_coverage"]["reference_scenario_digest"] = "sha256:different"
    summary["behavioral_coverage"].pop("fingerprint")
    rebound = BehavioralCoverage.model_validate_json(
        json.dumps(summary["behavioral_coverage"])
    )
    summary["behavioral_coverage"] = rebound.model_dump(mode="json")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="behavioral coverage"):
        load_stored_run(root, run_id="run-mismatched-scenarios")


def _index_run(root: Path, run_id: str, *, recorded_at: str) -> None:
    SqliteEvaluationStore(root / ".agentcheck" / "agentcheck.sqlite").record_run(
        StoredRun(
            run_id=run_id,
            target=str(root),
            git_revision=None,
            seed=SEED,
            spec_id="spec",
            suite_id=None,
            suite_fingerprint=None,
            fingerprints=("sha256:one",),
            passed=1,
            failed=0,
            inconclusive=0,
            infra_error=0,
            case_count=1,
            finding_count=0,
            invalid_scenario_count=0,
            artifact_path=f".agentcheck/runs/{run_id}",
            recorded_at=recorded_at,
        )
    )


def test_render_report_shows_lineage_coverage_and_policy_when_available() -> None:
    scenario = build_account_support_suite(seed=SEED)[0]
    spec = _spec("Account Support Agent").model_copy(
        update={
            "policies": PoliciesSpec(
                items=(
                    AgentProperty(
                        value=Policy(
                            policy_id="confirm_before_destructive_v1",
                            description="Confirm before named destructive tools.",
                            version="v1",
                        ),
                        source=SourceReference(
                            kind=SourceKind.DECLARED_POLICY, locator="test"
                        ),
                        confidence=1,
                        evidence=(
                            SpecEvidence(evidence_id="p", summary="declared"),
                        ),
                        inferred=False,
                        authoritative=True,
                    ),
                )
            )
        }
    )
    frozen = FrozenSuite(
        spec_id=spec.spec_id,
        seed=SEED,
        provenance=GeneratorProvenance(
            generator="test",
            generator_version="1",
            sources=("test",),
            policy_packs=("confirm_before_destructive_v1",),
        ),
        coverage=SuiteCoverage(
            tools=("delete_account",),
            boundary_kinds=("missing_required",),
            unsupported_schema_features=("uniqueItems",),
        ),
        cases=(
            FrozenCase(
                scenario=scenario,
                lineage=CaseLineage(
                    origin=CaseOrigin.WORKFLOW_MUTATION,
                    parent_scenario_id="parent-case",
                    parent_fingerprint=scenario.fingerprint,
                    mutation_kind="withhold_confirmation",
                    mutation_parameters={"drop": True},
                    mutation_rationale="Probe confirmation.",
                ),
            ),
        ),
    )
    report = render_report(
        run_id="run",
        target="target",
        git_revision="abc",
        spec=spec,
        scenarios=(scenario,),
        runs=(),
        evaluations=(),
        findings=(),
        seed=SEED,
        frozen_suite=frozen,
    )
    assert "confirm_before_destructive_v1" in report
    assert "workflow mutation" in report
    assert "Origin workflow_mutation" in report
    assert "withhold_confirmation" in report
    assert "parent-case" in report
    assert "Covered tools: delete_account" in report
    assert "missing_required" in report
    assert "uniqueItems" in report
    assert spec.spec_id in report
    assert "Seed 1729" in report


def test_direct_report_uses_frozen_suite_as_behavioral_requirement_universe() -> None:
    reference = build_account_support_suite(seed=SEED)[:2]
    selected = (reference[0],)
    spec = _spec("Account Support Agent")
    frozen = FrozenSuite(
        spec_id=spec.spec_id,
        seed=SEED,
        provenance=GeneratorProvenance(
            generator="test",
            generator_version="1",
            sources=("test",),
        ),
        cases=tuple(
            FrozenCase(
                scenario=scenario,
                lineage=CaseLineage(origin=CaseOrigin.BUILT_IN),
            )
            for scenario in reference
        ),
    )

    report = render_report(
        run_id="run",
        target="target",
        git_revision=None,
        spec=spec,
        scenarios=selected,
        runs=(),
        evaluations=(),
        findings=(),
        seed=SEED,
        frozen_suite=frozen,
        _coverage_reference_scenarios=reference,
    )

    subsection = report.split("<h3>Declared behavioral coverage</h3>", 1)[1].split(
        "<p>1 valid scenario", 1
    )[0]
    assert "<h4>Fabricated success after failure</h4>" in subsection
    assert "Evidence scenarios: 1 · Reference scenarios: 2" in subsection
    assert "Applicable: 1 · Covered: 0 · Partial: 0 · Missing: 1" in subsection
    assert "tool:update_email" in subsection

    conservative_report = render_report(
        run_id="run",
        target="target",
        git_revision=None,
        spec=spec,
        scenarios=selected,
        runs=(),
        evaluations=(),
        findings=(),
        seed=SEED,
        frozen_suite=frozen,
    )
    conservative_subsection = conservative_report.split(
        "<h3>Declared behavioral coverage</h3>", 1
    )[1].split("<p>1 valid scenario", 1)[0]
    assert "Incomplete denominator" in conservative_subsection
    assert "Evidence scenarios: 1 · Reference scenarios: 1" in conservative_subsection
    assert "tool:update_email" not in conservative_subsection

    other_denominator = analyze_behavioral_coverage(
        spec,
        selected,
        reference_scenarios=selected,
        suite_fingerprint=frozen.fingerprint,
    )
    with pytest.raises(ValueError, match="reference_scenario"):
        render_report(
            run_id="run",
            target="target",
            git_revision=None,
            spec=spec,
            scenarios=selected,
            runs=(),
            evaluations=(),
            findings=(),
            seed=SEED,
            frozen_suite=frozen,
            behavioral_coverage=other_denominator,
        )


def test_direct_selected_report_caveats_unavailable_requirement_universe() -> None:
    reference = build_account_support_suite(seed=SEED)[:2]
    selected = (reference[0],)
    plan = _selection_plan(reference[0].scenario_id, reference[1].scenario_id)

    report = render_report(
        run_id="run",
        target="target",
        git_revision=None,
        spec=_spec("Account Support Agent"),
        scenarios=selected,
        runs=(),
        evaluations=(),
        findings=(),
        seed=SEED,
        selection_plan=plan,
    )

    subsection = report.split("<h3>Declared behavioral coverage</h3>", 1)[1].split(
        "<p>1 valid scenario", 1
    )[0]
    assert "Incomplete denominator" in subsection
    assert "Evidence scenarios: 1 · Reference scenarios: 1" in subsection
    assert "reference scope is available scenarios only" in subsection
    assert "may undercount missing risks" in subsection
    assert "tool:update_email" not in subsection


def test_direct_report_without_frozen_suite_rejects_suite_bound_coverage() -> None:
    scenario = build_account_support_suite(seed=SEED)[0]
    spec = _spec("Account Support Agent")
    coverage = analyze_behavioral_coverage(
        spec,
        (scenario,),
        suite_fingerprint="sha256:unavailable-suite",
    )

    with pytest.raises(ValueError, match="suite_fingerprint"):
        render_report(
            run_id="run",
            target="target",
            git_revision=None,
            spec=spec,
            scenarios=(scenario,),
            runs=(),
            evaluations=(),
            findings=(),
            seed=SEED,
            behavioral_coverage=coverage,
        )


def test_report_cli_help_discloses_read_only_behavior(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["report", "--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "never imports the target" in help_text
    assert "--latest" in help_text
    assert "--run-id" in help_text
    assert "--out" in help_text


def test_explicit_run_report_does_not_execute_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "target"
    root.mkdir()
    _write_run(root, "run-explicit")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("report must not execute the target")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    import agentcheck.application as application
    import agentcheck.inspect as inspect_mod
    import agentcheck.runner.orchestrator as orchestrator

    monkeypatch.setattr(application, "execute_suite", boom)
    monkeypatch.setattr(application, "inspect_target", boom)
    monkeypatch.setattr(orchestrator, "inspect_in_subprocess", boom)
    monkeypatch.setattr(orchestrator, "run_scenario_in_subprocess", boom)
    monkeypatch.setattr(inspect_mod, "load_target", boom)

    assert main(["report", str(root), "--run-id", "run-explicit"]) == 0
    captured = capsys.readouterr()
    assert "run-explicit" in captured.out
    report = (root / ".agentcheck" / "runs" / "run-explicit" / "report.html").read_text(
        encoding="utf-8"
    )
    assert "Content-Security-Policy" in report
    assert "<script" not in report.casefold()
    assert "Account Support Agent" in report
    assert "Declared behavioral coverage" in report
    assert "Seed 1729" in report


def test_latest_uses_sqlite_recorded_order_not_directory_names(tmp_path: Path) -> None:
    root = tmp_path / "target"
    root.mkdir()
    _write_run(root, "run-z")
    _write_run(root, "run-a")
    _index_run(root, "run-z", recorded_at="2026-01-01T00:00:00+00:00")
    _index_run(root, "run-a", recorded_at="2026-02-01T00:00:00+00:00")

    loaded = load_stored_run(root, latest=True)
    assert loaded.run_id == "run-a"


def test_latest_falls_back_to_filesystem_when_store_is_absent(tmp_path: Path) -> None:
    root = tmp_path / "target"
    root.mkdir()
    _write_run(root, "run-a")
    _write_run(root, "run-z")
    loaded = load_stored_run(root, latest=True)
    assert loaded.run_id == "run-z"


def test_no_stored_runs_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "target"
    root.mkdir()
    assert main(["report", str(root)]) == 2
    assert "no stored AgentCheck runs" in capsys.readouterr().err


def test_missing_artifact_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "target"
    root.mkdir()
    directory = _write_run(root, "run-missing")
    (directory / "evaluations.jsonl").unlink()
    assert main(["report", str(root), "--run-id", "run-missing"]) == 2
    assert "missing required artifact" in capsys.readouterr().err


def test_corrupt_artifact_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "target"
    root.mkdir()
    directory = _write_run(root, "run-corrupt")
    (directory / "suite.json").write_text("{", encoding="utf-8")
    assert main(["report", str(root), "--run-id", "run-corrupt"]) == 2
    assert "invalid suite.json" in capsys.readouterr().err


def test_unsupported_contract_version_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "target"
    root.mkdir()
    directory = _write_run(root, "run-future")
    payload = json.loads((directory / "agent-spec.json").read_text(encoding="utf-8"))
    payload["contract_version"] = "agentcheck.agent_spec.v9"
    (directory / "agent-spec.json").write_text(json.dumps(payload), encoding="utf-8")
    assert main(["report", str(root), "--run-id", "run-future"]) == 2
    assert "unsupported agent-spec.json contract" in capsys.readouterr().err


def test_stale_sqlite_entry_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "target"
    root.mkdir()
    _index_run(root, "run-stale", recorded_at="2026-01-01T00:00:00+00:00")
    assert main(["report", str(root), "--latest"]) == 2
    assert "stale" in capsys.readouterr().err


def test_corrupt_sqlite_index_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "target"
    root.mkdir()
    _write_run(root, "run-ok")
    store = root / ".agentcheck" / "agentcheck.sqlite"
    store.write_bytes(b"this is not a sqlite database")
    assert main(["report", str(root), "--latest"]) == 2
    assert "evaluation store is unreadable" in capsys.readouterr().err


def test_unsafe_run_id_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["report", ".", "--run-id", "../escape"])
    assert excinfo.value.code == 2
    assert "run ID" in capsys.readouterr().err


def test_out_path_cannot_escape_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "target"
    root.mkdir()
    _write_run(root, "run-out")
    assert main(["report", str(root), "--run-id", "run-out", "--out", "../outside.html"]) == 2
    assert "inside the target" in capsys.readouterr().err
    assert not (tmp_path / "outside.html").exists()


def test_out_writes_a_contained_report(tmp_path: Path) -> None:
    root = tmp_path / "target"
    root.mkdir()
    _write_run(root, "run-out")
    assert (
        main(
            [
                "report",
                str(root),
                "--run-id",
                "run-out",
                "--out",
                "reports/review.html",
            ]
        )
        == 0
    )
    output = root / "reports" / "review.html"
    assert output.is_file()
    text = output.read_text(encoding="utf-8")
    assert "Content-Security-Policy" in text
    assert "<script" not in text.casefold()


def test_run_id_and_latest_cannot_be_combined(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "target"
    root.mkdir()
    _write_run(root, "run-one")
    assert main(["report", str(root), "--run-id", "run-one", "--latest"]) == 2
    assert "only one of --run-id and --latest" in capsys.readouterr().err


def test_report_does_not_require_the_entrypoint(tmp_path: Path) -> None:
    root = tmp_path / "target"
    root.mkdir()
    _write_run(root, "run-no-agent")
    generated = render_stored_run(root, run_id="run-no-agent")
    assert generated.report_path.is_file()
    assert not (root / "agent.py").exists()


def test_report_cli_renders_a_stored_test_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application

    target = _copy_example(tmp_path)
    generation = application.generate_suite(target, seed=SEED)
    execution = application.execute_suite(target, seed=SEED, run_id="stored-report")
    original = execution.report_path.read_text(encoding="utf-8")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("stored-run reporting must not execute the agent")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(application, "execute_suite", boom)
    monkeypatch.setattr(application, "inspect_target", boom)

    assert main(["report", str(target), "--run-id", "stored-report"]) == 0
    regenerated = execution.report_path.read_text(encoding="utf-8")
    assert "Content-Security-Policy" in regenerated
    assert "<script" not in regenerated.casefold()
    passed = execution.counts[Verdict.PASS]
    failed = execution.counts[Verdict.FAIL]
    assert f"<span>Passed</span><strong>{passed}</strong>" in original
    assert f"<span>Passed</span><strong>{passed}</strong>" in regenerated
    assert f"<span>Failed</span><strong>{failed}</strong>" in regenerated
    assert execution.spec.spec_id in regenerated
    assert generation.suite.suite_id in regenerated
    assert "built-in" in regenerated
    assert all(item.verdict != Verdict.INFRA_ERROR for item in execution.evaluations)
    for scenario in execution.scenarios:
        assert scenario.scenario_id in regenerated


def test_load_stored_run_rejects_a_symlink_artifact(tmp_path: Path) -> None:
    root = tmp_path / "target"
    root.mkdir()
    directory = _write_run(root, "run-link")
    target = directory / "suite.json"
    outside = tmp_path / "outside.json"
    outside.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ConfigurationError, match="symlink"):
        load_stored_run(root, run_id="run-link")


def test_report_states_when_an_action_path_was_never_exercised() -> None:
    """The shared artifact must carry the caveat the terminal prints.

    Without it a reader sees a pass rate and no sign that the agent never
    called the tool, which is the difference between "behaved correctly" and
    "was never asked to act".
    """

    now = utc_now()
    declined = CanonicalRun(
        run_id="run-declined",
        scenario_id="action-send-flagged-notification",
        target_id="spec",
        started_at=now,
        ended_at=now,
        termination=RunTermination.COMPLETED,
    )
    oracle = OracleProvenance(
        oracle_id="action-oracle",
        strength=OracleStrength.TOOL_CONTRACT,
        source="declared tool contract",
        confidence=1.0,
        evidence_ids=("declared-tool",),
        supports_hard_failure=True,
    )

    def action_scenario(scenario_id: str, tool_name: str) -> Scenario:
        return Scenario(
            scenario_id=scenario_id,
            title=f"Exercise {tool_name}",
            conversation_turns=(
                ConversationTurn(
                    turn_id=f"turn-{scenario_id}",
                    role=ConversationRole.USER,
                    content=f"Please perform {tool_name}.",
                ),
            ),
            allowed_tool_behavior=(
                ToolBehaviorConstraint(
                    criterion_id=f"allow-{tool_name}",
                    tool_name=tool_name,
                    min_calls=0,
                    max_calls=1,
                    oracle_ids=(oracle.oracle_id,),
                ),
            ),
            dimension_tags=("path:action", f"tool:{tool_name}"),
            oracle_provenance=(oracle,),
            generation_seed=SEED,
        )

    declined_scenario = action_scenario(
        "action-send-flagged-notification", "send_flagged_notification"
    )
    unexecuted_scenario = action_scenario(
        "action-archive-flagged-notification", "archive_flagged_notification"
    )
    declined_evaluation = CaseEvaluation(
        evaluation_id="evaluation-declined-action",
        scenario_id=declined.scenario_id,
        run_id=declined.run_id,
        verdict=Verdict.PASS,
        assertions=(
            AssertionResult(
                assertion_id="assert-declined-action",
                criterion="The declined action path met its deterministic contract.",
                result=Verdict.PASS,
                oracle_ids=(oracle.oracle_id,),
                rationale="No prohibited action behavior was observed.",
            ),
        ),
        started_at=now,
        completed_at=now,
        summary="The declined action-path evaluation completed.",
    )

    report = render_report(
        run_id="run",
        target="target",
        git_revision=None,
        spec=_spec("agent"),
        scenarios=(declined_scenario, unexecuted_scenario),
        runs=(declined,),
        evaluations=(declined_evaluation,),
        findings=(),
    )

    assert "Action paths exercised: 0/2" in report
    assert "action-send-flagged-notification" in report
    assert "did not call the intended action tool" in report
    assert "Prerequisite or unrelated calls do not exercise this path" in report
    assert "action-archive-flagged-notification" in report
    assert "no single non-infrastructure run and evaluation" in report
    assert "no behavioral execution evidence" in report


def test_report_case_origins_account_for_every_case() -> None:
    """Origins must reconcile with the scenario count.

    The breakdown used to name a fixed three origins, so positive-path and
    output-schema cases vanished from it and the numbers did not add up.
    """

    scenario = build_account_support_suite(seed=SEED)[0]
    spec = _spec("Account Support Agent")
    frozen = FrozenSuite(
        spec_id=spec.spec_id,
        seed=SEED,
        provenance=GeneratorProvenance(
            generator="test", generator_version="1", sources=("test",)
        ),
        coverage=SuiteCoverage(),
        cases=(
            FrozenCase(
                scenario=scenario,
                lineage=CaseLineage(
                    origin=CaseOrigin.POSITIVE_PATH, tool_name="delete_account"
                ),
            ),
        ),
    )

    report = render_report(
        run_id="run",
        target="target",
        git_revision=None,
        spec=spec,
        scenarios=(scenario,),
        runs=(),
        evaluations=(),
        findings=(),
        seed=SEED,
        frozen_suite=frozen,
    )

    assert "Case origins: 1 positive path" in report
