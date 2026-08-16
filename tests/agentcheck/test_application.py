from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import agentcheck.application as application
import agentcheck.cli as cli
from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.domain import Scenario
from agentcheck.errors import ScenarioValidationError
from agentcheck.generate import build_account_support_suite
from agentcheck.inspect import load_target
from agentcheck.runner.orchestrator import ProcessResult


EXAMPLE = Path(__file__).parents[2] / "examples" / "evaluation" / "account_agent"


def _replace(scenario: Scenario, **updates: object) -> Scenario:
    data = scenario.model_dump(mode="python")
    data.update(updates)
    data["fingerprint"] = ""
    return Scenario.model_validate(data)


def test_execute_suite_rejects_an_all_invalid_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, source = load_target(EXAMPLE)
    spec = OpenAIAgentsAdapter().inspect(target, source=source)
    base = build_account_support_suite()[0]
    invalid = _replace(
        base,
        required_tool_behavior=(),
        output_criteria=(),
        allowed_tool_behavior=base.required_tool_behavior,
    )
    inspection = ProcessResult(value=spec, infrastructure_error=None)
    monkeypatch.setattr(
        application,
        "inspect_in_subprocess",
        lambda *_args, **_kwargs: inspection,
    )
    monkeypatch.setattr(application, "_suite", lambda *_args, **_kwargs: (invalid,))

    with pytest.raises(ScenarioValidationError, match="No valid scenarios remain"):
        application.execute_suite(EXAMPLE, run_id="all-invalid-regression")


def test_cli_returns_an_error_when_no_valid_scenarios_remain(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def empty_suite(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(scenarios=())

    monkeypatch.setattr(cli, "execute_suite", empty_suite)

    assert cli.main(["test", "."]) == 2
    captured = capsys.readouterr()
    assert "No valid scenarios remain" in captured.err


def test_inspect_help_discloses_target_import_execution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["inspect", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    normalized = " ".join(output.split())
    assert "executing its module-level code" in normalized
    assert "without running an agent turn" in normalized
