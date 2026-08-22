from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

import agentcheck.cli as cli
from agentcheck.domain import Scenario, Verdict
from agentcheck.evaluate import infrastructure_evaluation
from agentcheck.errors import ConfigurationError
from agentcheck.generate import build_account_support_suite
from agentcheck.initialize import write_initial_config


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
        replay_manifest_path=None,
    )


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
    monkeypatch.setattr(cli, "execute_suite", lambda *_args, **_kwargs: execution)
    monkeypatch.setattr(cli, "_print_inspection", lambda *_args, **_kwargs: None)

    assert cli.main(["test", "."]) == 2

    output = capsys.readouterr().out
    assert f"INFRA_ERROR — {code}: {message}" in output
    assert scenario.title in output
    assert "Traceback" not in output


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
