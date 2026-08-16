from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import TypeVar

import pytest

from agentcheck.artifacts import ArtifactStore
from agentcheck.cli import main
from agentcheck.domain import (
    AgentProperty,
    AgentSpec,
    AssertionResult,
    CapabilitiesSpec,
    CanonicalRun,
    CaseEvaluation,
    IdentitySpec,
    InspectionProvenance,
    InstructionsSpec,
    InterfaceSpec,
    ObservabilitySpec,
    Policy,
    PoliciesSpec,
    RunTermination,
    RuntimeSpec,
    SourceKind,
    SourceReference,
    SpecEvidence,
    ToolsSpec,
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
from agentcheck.generate.suite import GeneratorProvenance, SuiteCoverage
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
