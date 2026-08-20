from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentcheck.artifacts import ArtifactStore
from agentcheck.cli import main
from agentcheck.config import AgentCheckConfig
from agentcheck.domain import (
    AgentProperty,
    AgentSpec,
    AssertionResult,
    CapabilitiesSpec,
    CanonicalRun,
    CaseEvaluation,
    Evidence,
    EvidenceKind,
    Finding,
    FixTarget,
    IdentitySpec,
    InspectionProvenance,
    InstructionsSpec,
    InterfaceSpec,
    ObservabilitySpec,
    RootCauseLayer,
    RunTermination,
    RuntimeSpec,
    Severity,
    SourceKind,
    SourceReference,
    SpecEvidence,
    SuggestedFix,
    ToolsSpec,
    Verdict,
    utc_now,
)
from agentcheck.errors import ConfigurationError
from agentcheck.generate.templates import build_account_support_suite
from agentcheck.report import render_stored_run
from agentcheck.review import (
    HUMAN_REVIEW_CONTRACT_VERSION,
    MAX_NOTE_CHARS,
    HumanReview,
    finding_fingerprint,
    load_reviews_for_run,
)
from agentcheck.review.service import record_finding_review
from agentcheck.review.contract import ReviewSourceBinding
from agentcheck.store import CURRENT_SCHEMA_VERSION, SqliteEvaluationStore


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"
SEED = 1729
FINDING_ID = "finding:duplicate_side_effect"


def _property(value: object) -> AgentProperty[object]:
    return AgentProperty(
        value=value,
        source=SourceReference(kind=SourceKind.RUNTIME_INTROSPECTION, locator="test"),
        confidence=1,
        evidence=(SpecEvidence(evidence_id="e", summary="test"),),
    )


def _spec() -> AgentSpec:
    return AgentSpec(
        spec_id="spec",
        identity=IdentitySpec(
            name=_property("Account Support Agent"),  # type: ignore[arg-type]
            framework=_property("OpenAI Agents SDK"),  # type: ignore[arg-type]
            framework_version=_property("0.20.0"),  # type: ignore[arg-type]
            provider=_property(None),  # type: ignore[arg-type]
            model=_property(None),  # type: ignore[arg-type]
        ),
        interface=InterfaceSpec(
            entrypoint=_property("agent.py:agent"),  # type: ignore[arg-type]
            input_modalities=_property(("text",)),  # type: ignore[arg-type]
            output_modalities=_property(("text",)),  # type: ignore[arg-type]
            input_schema=_property(None),  # type: ignore[arg-type]
            output_schema=_property(None),  # type: ignore[arg-type]
            interactive=_property(True),  # type: ignore[arg-type]
        ),
        instructions=InstructionsSpec(
            system=_property("support"),  # type: ignore[arg-type]
            developer=_property(None),  # type: ignore[arg-type]
        ),
        capabilities=CapabilitiesSpec(),
        tools=ToolsSpec(),
        runtime=RuntimeSpec(
            max_model_turns=_property(None),  # type: ignore[arg-type]
            max_tool_calls=_property(None),  # type: ignore[arg-type]
            timeout_seconds=_property(None),  # type: ignore[arg-type]
            token_budget=_property(None),  # type: ignore[arg-type]
            cost_budget_usd=_property(None),  # type: ignore[arg-type]
        ),
        observability=ObservabilitySpec(
            supported_event_types=_property(("final_output",)),  # type: ignore[arg-type]
            usage_metrics=_property(()),  # type: ignore[arg-type]
            provider_request_ids=_property(False),  # type: ignore[arg-type]
            source_event_links=_property(True),  # type: ignore[arg-type]
        ),
        provenance=InspectionProvenance(
            inspector="test",
            inspector_version="1",
            inspected_at=utc_now(),
            target="test",
            sources=(SourceReference(kind=SourceKind.RUNTIME_INTROSPECTION, locator="test"),),
        ),
    )


def _finding(scenario_id: str) -> Finding:
    return Finding(
        finding_id=FINDING_ID,
        failure_signature="duplicate_side_effect",
        title="Duplicate side-effect call",
        description="The same state-changing tool and arguments were executed more than once.",
        severity=Severity.HIGH,
        confidence=0.95,
        affected_scenario_ids=(scenario_id,),
        evidence_ids=("ev-1",),
        root_cause_layer=RootCauseLayer.WORKFLOW_LOGIC,
        likely_cause="The workflow lacks an execution/idempotency guard.",
        suggested_fixes=(
            SuggestedFix(
                fix_id="fix:duplicate_side_effect",
                target=FixTarget.WORKFLOW_LOGIC,
                summary="Record successful side effects.",
                rationale="Supported by one deterministic failing case.",
                proposed_change="Record successful side effects.",
                confidence=0.9,
                evidence_ids=("ev-1",),
            ),
        ),
    )


def _fail_evaluation(scenario_id: str, run_id: str) -> CaseEvaluation:
    now = utc_now()
    return CaseEvaluation(
        evaluation_id=f"eval-{scenario_id}",
        scenario_id=scenario_id,
        run_id=run_id,
        verdict=Verdict.FAIL,
        assertions=(
            AssertionResult(
                assertion_id="a-fail",
                criterion="must not duplicate a side effect",
                result=Verdict.FAIL,
                oracle_ids=("oracle-1",),
                supporting_evidence_ids=("ev-1",),
                rationale="Duplicate tool arguments were observed.",
                confidence=0.95,
            ),
        ),
        evidence=(
            Evidence(
                evidence_id="ev-1",
                kind=EvidenceKind.TOOL_ATTEMPT,
                summary="duplicate tool call",
                source_ids=("event-1",),
            ),
        ),
        started_at=now,
        completed_at=now,
        summary="failed",
    )


def _write_run(root: Path, run_id: str = "review-run") -> tuple[Path, Finding]:
    scenario = build_account_support_suite(seed=SEED)[0]
    spec = _spec()
    finding = _finding(scenario.scenario_id)
    evaluation = _fail_evaluation(scenario.scenario_id, f"{run_id}-case-001")
    run = CanonicalRun(
        run_id=evaluation.run_id or f"{run_id}-case-001",
        scenario_id=scenario.scenario_id,
        target_id=spec.spec_id,
        started_at=utc_now(),
        ended_at=utc_now(),
        termination=RunTermination.COMPLETED,
    )
    artifacts = ArtifactStore(root, ".agentcheck", run_id)
    artifacts.write_json("agent-spec.json", spec)
    artifacts.write_json(
        "suite.json",
        {
            "schema_version": "agentcheck.suite.v1",
            "run_id": run_id,
            "seed": SEED,
            "scenarios": [scenario],
        },
    )
    artifacts.write_json(
        "invalid-scenarios.json",
        {"schema_version": "agentcheck.invalid_scenarios.v1", "items": []},
    )
    artifacts.write_jsonl("runs.jsonl", (run,))
    artifacts.write_jsonl("evaluations.jsonl", (evaluation,))
    artifacts.write_json("findings.json", (finding,))
    artifacts.write_json(
        "summary.json",
        {
            "schema_version": "agentcheck.summary.v1",
            "run_id": run_id,
            "target": str(root),
            "git_revision": None,
            "suite_size": 1,
            "invalid_scenarios": 0,
            "observed_suite_pass_rate": 0.0,
            "counts": {
                "PASS": 0,
                "FAIL": 1,
                "INCONCLUSIVE": 0,
                "INFRA_ERROR": 0,
            },
            "finding_count": 1,
            "seed": SEED,
        },
    )
    artifacts.write_text("report.html", "<html><body>placeholder</body></html>")
    return artifacts.root, finding


def _target(tmp_path: Path) -> Path:
    root = tmp_path / "account_agent"
    shutil.copytree(
        EXAMPLE,
        root,
        ignore=shutil.ignore_patterns(".agentcheck", "__pycache__"),
    )
    return root


def _patch_execution_tripwires(monkeypatch: pytest.MonkeyPatch) -> None:
    import agentcheck.application as application
    import agentcheck.runner.orchestrator as orchestrator
    from agentcheck.runner.tool_gateway import ToolGateway

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("review must not inspect, run a worker, or invoke tools")

    monkeypatch.setattr(application, "inspect_in_subprocess", explode)
    monkeypatch.setattr(application, "run_scenario_in_subprocess", explode)
    monkeypatch.setattr(orchestrator, "inspect_in_subprocess", explode)
    monkeypatch.setattr(orchestrator, "run_scenario_in_subprocess", explode)
    monkeypatch.setattr(ToolGateway, "__init__", explode)


def test_review_decisions_are_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)
    _write_run(root)
    _patch_execution_tripwires(monkeypatch)
    for decision in ("accepted", "rejected", "needs_followup"):
        recorded = record_finding_review(
            root,
            run_id="review-run",
            finding_id=FINDING_ID,
            decision=decision,  # type: ignore[arg-type]
            note=f"Decision {decision}",
            reviewer="alice",
        )
        assert recorded.review.decision == decision
        assert recorded.review.automated_verdict == "FAIL"
        assert recorded.review.schema_version == HUMAN_REVIEW_CONTRACT_VERSION


def test_invalid_decision_is_rejected(tmp_path: Path) -> None:
    root = _target(tmp_path)
    _write_run(root)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "review",
                str(root),
                "--run-id",
                "review-run",
                "--finding-id",
                FINDING_ID,
                "--decision",
                "blessed",
            ]
        )
    assert excinfo.value.code == 2


def test_unknown_run_and_finding(tmp_path: Path) -> None:
    root = _target(tmp_path)
    _write_run(root)
    with pytest.raises(ConfigurationError, match="run artifacts were not found"):
        record_finding_review(
            root,
            run_id="missing-run",
            finding_id=FINDING_ID,
            decision="accepted",
        )
    with pytest.raises(ConfigurationError, match="unknown finding"):
        record_finding_review(
            root,
            run_id="review-run",
            finding_id="finding:not-real",
            decision="accepted",
        )


def test_note_size_and_secret_are_refused(tmp_path: Path) -> None:
    root = _target(tmp_path)
    _write_run(root)
    with pytest.raises(ConfigurationError, match="character bound"):
        record_finding_review(
            root,
            run_id="review-run",
            finding_id=FINDING_ID,
            decision="accepted",
            note="x" * (MAX_NOTE_CHARS + 1),
        )
    with pytest.raises(ConfigurationError, match="credential-shaped"):
        record_finding_review(
            root,
            run_id="review-run",
            finding_id=FINDING_ID,
            decision="accepted",
            note="sk-thisisafakesecretvalue12",
        )


def test_append_only_latest_and_unchanged_automation(tmp_path: Path) -> None:
    root = _target(tmp_path)
    run_dir, _written = _write_run(root)
    findings_before = (run_dir / "findings.json").read_bytes()
    evaluations_before = (run_dir / "evaluations.jsonl").read_bytes()
    stored_finding = Finding.model_validate_json(
        json.dumps(json.loads(findings_before.decode("utf-8"))[0])
    )
    fingerprint_before = finding_fingerprint(stored_finding)

    first = record_finding_review(
        root,
        run_id="review-run",
        finding_id=FINDING_ID,
        decision="needs_followup",
        note="Need more context",
    )
    first_bytes = first.path.read_bytes()
    second = record_finding_review(
        root,
        run_id="review-run",
        finding_id=FINDING_ID,
        decision="accepted",
        note="Confirmed regression",
    )
    assert first.path.read_bytes() == first_bytes
    assert second.path != first.path
    loaded = load_reviews_for_run(root, AgentCheckConfig(), "review-run")
    assert [item.decision for item in loaded] == ["needs_followup", "accepted"]
    assert (run_dir / "findings.json").read_bytes() == findings_before
    assert (run_dir / "evaluations.jsonl").read_bytes() == evaluations_before
    reloaded = Finding.model_validate_json(
        json.dumps(json.loads((run_dir / "findings.json").read_text(encoding="utf-8"))[0])
    )
    assert finding_fingerprint(reloaded) == fingerprint_before
    assert reloaded.failure_signature == stored_finding.failure_signature


def test_finding_identity_mismatch_is_visible_in_report(tmp_path: Path) -> None:
    root = _target(tmp_path)
    run_dir, finding = _write_run(root)
    record_finding_review(
        root,
        run_id="review-run",
        finding_id=FINDING_ID,
        decision="accepted",
        note="Confirmed",
    )
    payload = json.loads((run_dir / "findings.json").read_text(encoding="utf-8"))
    payload[0]["title"] = "Tampered title"
    (run_dir / "findings.json").write_text(json.dumps(payload), encoding="utf-8")
    generated = render_stored_run(root, run_id="review-run")
    html = generated.report_path.read_text(encoding="utf-8")
    assert "Tampered title" in html
    assert "Automated verdict" in html
    assert "FAIL" in html
    assert "finding identity mismatch" in html
    assert "Confirmed" in html
    assert "default-src 'none'" in html


def test_html_escapes_review_note(tmp_path: Path) -> None:
    root = _target(tmp_path)
    _write_run(root)
    record_finding_review(
        root,
        run_id="review-run",
        finding_id=FINDING_ID,
        decision="rejected",
        note='</pre><script>alert("x")</script>',
    )
    generated = render_stored_run(root, run_id="review-run")
    html = generated.report_path.read_text(encoding="utf-8")
    assert "<script" not in html.casefold()
    assert "&lt;script&gt;" in html
    assert "Automated verdict" in html
    assert "default-src 'none'" in html


def test_corrupt_and_unsupported_review_artifacts(tmp_path: Path) -> None:
    root = _target(tmp_path)
    _write_run(root)
    reviews = root / ".agentcheck" / "reviews" / "review-run"
    reviews.mkdir(parents=True)
    (reviews / "review-aaaaaaaaaaaaaaaaaaaaaaaa.json").write_text("{", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid human review"):
        load_reviews_for_run(root, AgentCheckConfig(), "review-run")
    (reviews / "review-aaaaaaaaaaaaaaaaaaaaaaaa.json").write_text(
        json.dumps({"schema_version": "agentcheck.human_review.v0"}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="unsupported human review"):
        load_reviews_for_run(root, AgentCheckConfig(), "review-run")


def test_corrupt_findings_are_refused(tmp_path: Path) -> None:
    root = _target(tmp_path)
    run_dir, _finding_obj = _write_run(root)
    (run_dir / "findings.json").write_text("{", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        record_finding_review(
            root,
            run_id="review-run",
            finding_id=FINDING_ID,
            decision="accepted",
        )


def test_review_contract_rejects_extra_fields_and_versions() -> None:
    document = HumanReview(
        run_id="review-run",
        finding_id=FINDING_ID,
        finding_fingerprint="sha256:" + "a" * 64,
        failure_signature="duplicate_side_effect",
        automated_verdict="FAIL",
        decision="accepted",
        recorded_at=utc_now(),
        source=ReviewSourceBinding(
            findings_path=".agentcheck/runs/review-run/findings.json",
            findings_digest="sha256:" + "b" * 64,
        ),
        agentcheck_version="0.1.0",
    ).model_dump(mode="json")
    document["unexpected"] = True
    with pytest.raises(ValidationError):
        HumanReview.model_validate_json(json.dumps(document))
    document = HumanReview(
        run_id="review-run",
        finding_id=FINDING_ID,
        finding_fingerprint="sha256:" + "a" * 64,
        failure_signature="duplicate_side_effect",
        automated_verdict="FAIL",
        decision="accepted",
        recorded_at=utc_now(),
        source=ReviewSourceBinding(
            findings_path=".agentcheck/runs/review-run/findings.json",
            findings_digest="sha256:" + "b" * 64,
        ),
        agentcheck_version="0.1.0",
    ).model_dump(mode="json")
    document["schema_version"] = "agentcheck.human_review.v0"
    with pytest.raises(ValidationError):
        HumanReview.model_validate_json(json.dumps(document))


def test_cli_review_does_not_execute_and_indexes_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _target(tmp_path)
    _write_run(root)
    _patch_execution_tripwires(monkeypatch)
    assert (
        main(
            [
                "review",
                str(root),
                "--run-id",
                "review-run",
                "--finding-id",
                FINDING_ID,
                "--decision",
                "accepted",
                "--note",
                "Confirmed",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "Automated verdict: FAIL" in captured.out
    assert "Human decision: accepted" in captured.out
    store = SqliteEvaluationStore(root / ".agentcheck" / "agentcheck.sqlite")
    connection = store._connect()
    try:
        count = connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
        version = connection.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0]
    finally:
        connection.close()
    assert count == 1
    assert int(version) == CURRENT_SCHEMA_VERSION


def test_review_does_not_invoke_replay_or_shrink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application

    root = _target(tmp_path)
    _write_run(root)

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("review must not execute replay, shrink, or workers")

    monkeypatch.setattr(application, "replay_suite", explode)
    monkeypatch.setattr(application, "shrink_suite", explode)
    monkeypatch.setattr(application, "execute_suite", explode)
    _patch_execution_tripwires(monkeypatch)
    record_finding_review(
        root,
        run_id="review-run",
        finding_id=FINDING_ID,
        decision="rejected",
        note="Not a regression",
    )
