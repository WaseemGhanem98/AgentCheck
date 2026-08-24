from __future__ import annotations

import io
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

import agentcheck.cli as cli
from agentcheck.coverage import (
    BehavioralCoverage,
    BehavioralCoverageFamily,
    BehavioralCoverageReferenceScope,
    BehavioralCoverageRequirement,
    BehavioralCoverageStatus,
    BehavioralDimension,
)
from agentcheck.domain import Scenario, Verdict
from agentcheck.evaluate import infrastructure_evaluation
from agentcheck.errors import ConfigurationError
from agentcheck.generate import build_account_support_suite
from agentcheck.initialize import write_initial_config


def _empty_behavioral_coverage() -> BehavioralCoverage:
    return BehavioralCoverage(
        spec_id="test-spec",
        spec_digest="sha256:test-spec",
        scenario_count=1,
        scenario_digest="sha256:test-scenarios",
        reference_scenario_count=1,
        reference_scenario_digest="sha256:test-scenarios",
    )


def _infra_execution(
    scenario: Scenario,
    *,
    code: str,
    message: str,
) -> SimpleNamespace:
    evaluation = infrastructure_evaluation(
        scenario,
        code=code,
        message=message,
        phase="run",
    )
    return SimpleNamespace(
        scenarios=(scenario,),
        evaluations=(evaluation,),
        spec=object(),
        frozen_suite=None,
        invalid_scenarios=(),
        selection=None,
        counts=Counter({Verdict.INFRA_ERROR: 1}),
        observed_pass_rate=None,
        runs=(),
        findings=(),
        report_path=Path("report.html"),
        behavioral_coverage=_empty_behavioral_coverage(),
        replay_manifest_path=None,
    )


def _execution_for_verdict(
    scenario: Scenario,
    verdict: Verdict,
) -> SimpleNamespace:
    if verdict == Verdict.INFRA_ERROR:
        return _infra_execution(
            scenario,
            code="fixture_not_found",
            message="No fixture declared for tool 'lookup_customer'.",
        )
    evaluation = SimpleNamespace(
        scenario_id=scenario.scenario_id,
        verdict=verdict,
        infrastructure_error=None,
    )
    return SimpleNamespace(
        scenarios=(scenario,),
        evaluations=(evaluation,),
        spec=object(),
        frozen_suite=None,
        invalid_scenarios=(),
        selection=None,
        counts=Counter({verdict: 1}),
        observed_pass_rate=1.0 if verdict == Verdict.PASS else 0.0,
        runs=(),
        findings=(),
        report_path=Path("report.html"),
        behavioral_coverage=_empty_behavioral_coverage(),
        replay_manifest_path=None,
    )


def _execute_with_progress(execution: SimpleNamespace):
    def execute(*_args: object, **kwargs: object) -> SimpleNamespace:
        on_inspected = kwargs["on_inspected"]
        on_prepared = kwargs["on_prepared"]
        progress = kwargs["progress"]
        assert callable(on_inspected)
        assert callable(on_prepared)
        assert callable(progress)
        on_inspected(execution.spec)
        excluded = (
            len(execution.selection.excluded_ids)
            if execution.selection is not None
            else 0
        )
        on_prepared(
            execution.frozen_suite,
            len(execution.scenarios),
            len(execution.invalid_scenarios),
            excluded,
        )
        for completed, (scenario, evaluation) in enumerate(
            zip(execution.scenarios, execution.evaluations),
            start=1,
        ):
            progress(
                completed,
                len(execution.scenarios),
                scenario,
                evaluation,
            )
        return execution

    return execute


def test_declared_behavioral_coverage_prints_counts_and_exact_open_subjects(
    capsys: pytest.CaptureFixture[str],
) -> None:
    requirements = tuple(
        BehavioralCoverageRequirement(
            subject=subject,
            status=status,
            reason_code=f"reason_{status.value}",
        )
        for subject, status in (
            ("tool:covered", BehavioralCoverageStatus.COVERED),
            ("tool:partial", BehavioralCoverageStatus.PARTIAL),
            ("tool:missing", BehavioralCoverageStatus.MISSING),
            ("tool:unknown", BehavioralCoverageStatus.UNKNOWN),
            ("tool:unsupported", BehavioralCoverageStatus.UNSUPPORTED),
        )
    )
    coverage = BehavioralCoverage(
        spec_id="test-spec",
        spec_digest="sha256:test-spec",
        scenario_count=5,
        scenario_digest="sha256:test-scenarios",
        reference_scenario_count=7,
        reference_scenario_digest="sha256:test-reference-scenarios",
        families=(
            BehavioralCoverageFamily(dimension=BehavioralDimension.SUCCESS_PATH),
            BehavioralCoverageFamily(
                dimension=BehavioralDimension.FAILURE_HANDLING,
                covered=1,
                partial=1,
                missing=1,
                unknown=1,
                unsupported=1,
                requirements=requirements,
            ),
        ),
    )

    cli._print_declared_behavioral_coverage(coverage)

    output = capsys.readouterr().out
    assert "Declared behavioral coverage:" in output
    assert "Evidence scenarios: 5 / 7 reference scenarios" in output
    assert (
        "Failure handling: 1 covered / 3 applicable; partial 1; missing 1; "
        "unknown 1; unsupported 1"
    ) in output
    assert "partial: tool:partial [reason_partial]" in output
    assert "missing: tool:missing [reason_missing]" in output
    assert "unknown: tool:unknown [reason_unknown]" in output
    assert "unsupported: tool:unsupported [reason_unsupported]" in output
    assert "tool:covered" not in output
    assert "Success path:" not in output
    assert "%" not in output
    assert "complete coverage" not in output.casefold()


def test_declared_behavioral_coverage_redacts_bounded_requirement_identifiers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_subject_secret = "sk-secretvalue123"
    secret_subject = f"tool:{raw_subject_secret}"
    secret_reason = "sk-reasonsecret123"
    coverage = BehavioralCoverage(
        spec_id="test-spec",
        spec_digest="sha256:test-spec",
        scenario_count=1,
        scenario_digest="sha256:test-scenarios",
        reference_scenario_count=1,
        reference_scenario_digest="sha256:test-scenarios",
        families=(
            BehavioralCoverageFamily(
                dimension=BehavioralDimension.RETRY_CONTROL,
                unknown=1,
                requirements=(
                    BehavioralCoverageRequirement(
                        subject=secret_subject,
                        status=BehavioralCoverageStatus.UNKNOWN,
                        reason_code=secret_reason,
                    ),
                ),
            ),
        ),
    )

    cli._print_declared_behavioral_coverage(coverage)

    output = capsys.readouterr().out
    assert secret_subject not in output
    assert raw_subject_secret not in output
    assert secret_reason not in output
    assert "[REDACTED]" in output


def test_declared_behavioral_coverage_collapses_all_empty_families(
    capsys: pytest.CaptureFixture[str],
) -> None:
    coverage = BehavioralCoverage(
        spec_id="test-spec",
        spec_digest="sha256:test-spec",
        scenario_count=0,
        scenario_digest="sha256:test-scenarios",
        reference_scenario_count=0,
        reference_scenario_digest="sha256:test-scenarios",
        families=(
            BehavioralCoverageFamily(dimension=BehavioralDimension.SUCCESS_PATH),
            BehavioralCoverageFamily(dimension=BehavioralDimension.ORDERING),
        ),
    )

    cli._print_declared_behavioral_coverage(coverage)

    output = capsys.readouterr().out
    assert "No declared behavioral requirements were identified." in output
    assert "Success path:" not in output
    assert "Ordering:" not in output


def test_declared_behavioral_coverage_warns_when_reference_scope_is_incomplete(
    capsys: pytest.CaptureFixture[str],
) -> None:
    coverage = BehavioralCoverage(
        spec_id="test-spec",
        spec_digest="sha256:test-spec",
        scenario_count=1,
        scenario_digest="sha256:test-scenarios",
        reference_scenario_count=1,
        reference_scenario_digest="sha256:test-scenarios",
        reference_scope=BehavioralCoverageReferenceScope.AVAILABLE_SCENARIOS_ONLY,
    )

    cli._print_declared_behavioral_coverage(coverage)

    output = capsys.readouterr().out
    assert "Evidence scenarios: 1 / 1 reference scenarios" in output
    assert "reference scope contains available scenarios only" in output
    assert "unavailable scenario contracts may add behavioral requirements" in output
    assert "%" not in output


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (
            "fixture_not_found",
            "No fixture declared for tool 'lookup_customer'.",
        ),
        (
            "unknown_tool",
            "Tool 'send_email' is not declared by the inspected agent.",
        ),
        (
            "fixture_not_found",
            "Prerequisite fixture for tool 'authenticate_user' is missing.",
        ),
    ],
)
def test_test_command_surfaces_actionable_infrastructure_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    code: str,
    message: str,
) -> None:
    scenario = build_account_support_suite()[0]
    execution = _infra_execution(scenario, code=code, message=message)
    monkeypatch.setattr(cli, "execute_suite", _execute_with_progress(execution))
    monkeypatch.setattr(cli, "_print_inspection", lambda *_args, **_kwargs: None)

    assert cli.main(["test", "."]) == 2

    output = capsys.readouterr().out
    assert f"INFRA_ERROR — {code}: {message}" in output
    assert scenario.title in output
    assert "Traceback" not in output


def test_test_command_reports_stages_and_scenario_progress_in_order(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scenario = build_account_support_suite()[0]
    execution = _execution_for_verdict(scenario, Verdict.PASS)
    monkeypatch.setattr(cli, "execute_suite", _execute_with_progress(execution))
    monkeypatch.setattr(cli, "_print_inspection", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli,
        "_print_action_path_exercise",
        lambda *_args, **_kwargs: None,
    )

    assert cli.main(["test", "."]) == 0

    output = capsys.readouterr().out
    expected = (
        "Inspecting agent...",
        "Inspection complete. ✓",
        "Building deterministic test suite... ✓ 1 scenario",
        "Running 1 scenario in isolated workers...",
        "[1/1]",
        "Finalizing report...",
    )
    positions = [output.index(item) for item in expected]
    assert positions == sorted(positions)
    assert scenario.title in output
    assert "[1/1]" in output
    assert output.count(" PASS") >= 1


@pytest.mark.parametrize(
    ("verdict", "exit_code"),
    [
        (Verdict.PASS, 0),
        (Verdict.FAIL, 1),
        (Verdict.INCONCLUSIVE, 3),
        (Verdict.INFRA_ERROR, 2),
    ],
)
def test_test_progress_renders_each_verdict_without_changing_exit_semantics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verdict: Verdict,
    exit_code: int,
) -> None:
    scenario = build_account_support_suite()[0]
    execution = _execution_for_verdict(scenario, verdict)
    monkeypatch.setattr(cli, "execute_suite", _execute_with_progress(execution))
    monkeypatch.setattr(cli, "_print_inspection", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli,
        "_print_action_path_exercise",
        lambda *_args, **_kwargs: None,
    )

    assert cli.main(["test", "."]) == exit_code

    output = capsys.readouterr().out
    result_line = next(line for line in output.splitlines() if line.startswith("[1/1]"))
    assert result_line.endswith(verdict.value)
    assert "Declared behavioral coverage:" in output


def test_test_progress_heartbeat_is_readable_without_a_tty() -> None:
    stream = io.StringIO()
    reporter = cli._TestProgressReporter(
        stream=stream,
        heartbeat_seconds=60.0,
    )
    reporter.start()
    reporter.inspection_complete()
    reporter.suite_ready(None, 2, 0, 0)
    scenario = build_account_support_suite()[0]
    evaluation = _execution_for_verdict(scenario, Verdict.PASS).evaluations[0]
    reporter.case_completed(1, 2, scenario, evaluation)
    reporter.emit_heartbeat()
    reporter.close()

    output = stream.getvalue()
    result_position = output.index("[1/2]")
    heartbeat_position = output.index("1 scenario still running (1/2 complete)")
    assert result_position < heartbeat_position
    assert "[1/2] Still running..." not in output
    assert "\r" not in output
    assert "\x1b" not in output
    assert "\b" not in output


def test_infrastructure_diagnostic_is_redacted_single_line_and_bounded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    scenario = build_account_support_suite()[0]
    code = "fixture_" + ("c" * 100)
    message = (
        "Authorization: Bearer test-secret-token-value\n"
        "\x1b[31munsafe-control "
        + ("x" * 500)
    )
    evaluation = infrastructure_evaluation(
        scenario,
        code=code,
        message=message,
        phase="run",
    )

    cli._print_case_evaluation(evaluation, title=scenario.title)

    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "INFRA_ERROR —" in line
    ]
    assert len(lines) == 1
    diagnostic = lines[0]
    assert "[REDACTED]" in diagnostic
    assert "test-secret-token-value" not in diagnostic
    assert "\x1b" not in diagnostic
    assert "[TRUNCATED]" in diagnostic
    assert len(diagnostic) <= 430


def test_test_help_states_the_exact_declared_tool_safety_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["test", "--help"])

    assert exc_info.value.code == 0
    normalized = " ".join(capsys.readouterr().out.split())
    assert "Declared tool calls are simulated" in normalized
    assert "never reach their original handlers" in normalized
    assert "direct filesystem writes" in normalized
    assert "subprocess execution" in normalized
    assert "local database access" in normalized
    assert "outside the declared-tool guarantee" in normalized
    assert "cannot cause external side effects" not in normalized


def test_init_help_says_the_target_directory_must_already_exist(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["init", "--help"])

    assert exc_info.value.code == 0
    normalized = " ".join(capsys.readouterr().out.split())
    assert "TARGET must already be an existing directory" in normalized
    assert "init does not create it" in normalized


def test_missing_init_directory_error_tells_the_developer_what_to_do(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigurationError, match="create the directory first"):
        write_initial_config(tmp_path / "absent")


def test_custom_init_explains_that_the_starter_may_initially_fail(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["init", str(tmp_path), "--adapter", "custom"]) == 0

    output = capsys.readouterr().out
    assert "integration shape, not a finished policy" in output
    assert "first generated evaluation may contain behavioral FAIL results" in output
    assert "representative fixtures" in output


def test_public_safety_docs_keep_the_declared_tool_boundary_narrow() -> None:
    root = Path(__file__).parents[2]
    readme = (root / "README.md").read_text(encoding="utf-8")
    security = (root / "SECURITY.md").read_text(encoding="utf-8")
    custom = (root / "docs" / "custom-agents.md").read_text(encoding="utf-8")

    assert "Declared real tool handlers never execute" in readme
    assert "outside the declared-tool guarantee" in readme
    assert "Network denial is not a general operating-system sandbox" in readme

    assert "declared-tool guarantee is not a general Python sandbox" in security
    assert "Direct filesystem writes, subprocess execution" in security
    assert "direct local-database access" in " ".join(security.split())
    assert "outside the guarantee" in " ".join(security.split())

    assert "Declared real tool handlers are never supplied" in custom
    assert "os.remove(...)" in custom
    assert "subprocess.run(...)" in custom
    assert "local database" in custom
    assert "outside the declared-tool guarantee" in custom
    assert "Network denial is not a general operating-system sandbox" in custom


def test_allowed_public_alpha_polish_does_not_drift() -> None:
    root = Path(__file__).parents[2]
    readme = (root / "README.md").read_text(encoding="utf-8")
    ci_docs = (root / "docs" / "ci-trust-model.md").read_text(encoding="utf-8")
    custom = (root / "docs" / "custom-agents.md").read_text(encoding="utf-8")
    handoff = (
        root / "examples" / "evaluation" / "handoff_router" / "README.md"
    ).read_text(encoding="utf-8")

    assert "python -m pytest tests -q -n 2" in readme
    assert "964 tests" not in ci_docs
    assert "381 of 964" not in ci_docs
    assert "integration shape, not a finished agent policy" in " ".join(custom.split())
    assert "first generated evaluation can contain behavioral `FAIL`" in custom
    assert "not a CLI demonstration of the three planted handoff defects" in " ".join(handoff.split())
    assert "may report 100% with zero action paths exercised" in " ".join(handoff.split())
