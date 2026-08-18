from __future__ import annotations

from pathlib import Path

import pytest

import agentcheck.application as application
from agentcheck.adapters import OpenAIAgentsAdapter, UnsupportedTargetError
from agentcheck.cli import main
from agentcheck.config import AgentCheckConfig
from agentcheck.errors import IncompatibleSuiteError, ScenarioValidationError
from agentcheck.generate import (
    ACCOUNT_SUPPORT_SUITE,
    ACCOUNT_TOOLS,
    CaseOrigin,
    DEFAULT_SUITE_FILENAME,
    build_frozen_suite,
    spec_matches_built_in_suite,
)
from agentcheck.initialize import write_initial_config
from agentcheck.inspect import load_target


EXAMPLE = Path(__file__).parents[2] / "examples" / "evaluation" / "account_agent"

ACCOUNT_SUPPORT_IDS = {
    "happy_lookup",
    "happy_email_update",
    "confirmed_cancel",
    "confirmed_delete",
    "delete_without_confirmation",
    "ambiguous_account_selection",
    "destructive_ambiguous_timeout",
    "fabricated_email_success",
    "duplicate_email_update",
    "honest_lookup_timeout",
    "missing_account",
    "ambiguous_delete_clarification",
}

WEATHER_AGENT = """
from agents import Agent, function_tool


@function_tool
def get_weather(city: str) -> str:
    return "sunny"


agent = Agent(
    name="Weather",
    instructions="Look up weather.",
    tools=[get_weather],
    model="gpt-4.1-mini",
)
"""

PARTIAL_ACCOUNT_AGENT = """
from agents import Agent, function_tool


@function_tool
def lookup_account(account_id: str) -> dict:
    return {}


agent = Agent(
    name="Partial account",
    instructions="Look up accounts.",
    tools=[lookup_account],
    model="gpt-4.1-mini",
)
"""

DYNAMIC_INSTRUCTIONS = """
from agents import Agent


def custom_instructions(_context, _agent):
    return "Be helpful."


agent = Agent(
    name="Chat agent",
    instructions=custom_instructions,
    model="gpt-4.1-mini",
)
"""

NO_TOOLS_AGENT = """
from agents import Agent

agent = Agent(
    name="Chat",
    instructions="Say hello.",
    model="gpt-4.1-mini",
)
"""


def _write_target(tmp_path: Path, source: str, *, name: str = "target") -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "agent.py").write_text(source.lstrip(), encoding="utf-8")
    write_initial_config(root)
    return root


def _inspect(root: Path):
    target, source = load_target(root)
    return OpenAIAgentsAdapter().inspect(target, source=source)


def _stop_before_worker(**kwargs: object) -> object:
    raise _StoppedBeforeWorker(kwargs)


class _StoppedBeforeWorker(Exception):
    def __init__(self, captured: dict[str, object]) -> None:
        super().__init__("stop-before-worker")
        self.captured = captured


def test_account_support_agent_matches_built_in_suite() -> None:
    spec = _inspect(EXAMPLE)
    assert spec_matches_built_in_suite(spec, ACCOUNT_SUPPORT_SUITE)
    assert ACCOUNT_TOOLS <= {item.value.name for item in spec.tools.items}


def test_weather_agent_does_not_match_account_support_suite(tmp_path: Path) -> None:
    spec = _inspect(_write_target(tmp_path, WEATHER_AGENT))
    assert not spec_matches_built_in_suite(spec, ACCOUNT_SUPPORT_SUITE)
    assert {item.value.name for item in spec.tools.items} == {"get_weather"}


def test_partial_account_tools_do_not_match_account_support_suite(
    tmp_path: Path,
) -> None:
    spec = _inspect(_write_target(tmp_path, PARTIAL_ACCOUNT_AGENT))
    assert "lookup_account" in {item.value.name for item in spec.tools.items}
    assert not spec_matches_built_in_suite(spec, ACCOUNT_SUPPORT_SUITE)


def test_account_support_agent_selects_built_in_suite_without_frozen_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(application, "_execute_valid_scenarios", _stop_before_worker)

    with pytest.raises(_StoppedBeforeWorker) as exc_info:
        application.execute_suite(EXAMPLE, persist_store=False)

    captured = exc_info.value.captured
    assert captured["frozen"] is None
    valid = captured["valid"]
    assert isinstance(valid, tuple)
    ids = {scenario.scenario_id for scenario in valid}
    assert ids == ACCOUNT_SUPPORT_IDS


def test_weather_test_does_not_select_or_lint_account_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _write_target(tmp_path, WEATHER_AGENT)

    def explode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("account_support_v1 must not be selected")

    monkeypatch.setattr(application, "_suite", explode)
    monkeypatch.setattr(application, "_execute_valid_scenarios", explode)

    with pytest.raises(IncompatibleSuiteError, match="No compatible built-in suite"):
        application.execute_suite(target, persist_store=False)


def test_cli_test_explains_when_no_compatible_built_in_suite_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _write_target(tmp_path, WEATHER_AGENT)

    assert main(["test", str(target), "--no-store"]) == 2
    err = capsys.readouterr().err
    assert "No compatible built-in suite exists for this target" in err
    assert "account_support_v1" in err
    assert "get_weather" in err
    assert "agentcheck generate" in err
    assert "not a passing verdict" in err
    assert "No valid scenarios remain after linting" not in err


def test_generate_omits_incompatible_built_in_cases(tmp_path: Path) -> None:
    spec = _inspect(_write_target(tmp_path, WEATHER_AGENT))
    suite = build_frozen_suite(spec, AgentCheckConfig(), seed=1729)

    assert suite.cases
    # The point is that no incompatible built-in case leaked in; generated
    # origins such as the action path are expected alongside the boundaries.
    assert all(case.lineage.origin is not CaseOrigin.BUILT_IN for case in suite.cases)
    assert any(case.lineage.origin is CaseOrigin.SCHEMA_BOUNDARY for case in suite.cases)
    assert not any(
        rejected.lineage.origin is CaseOrigin.BUILT_IN for rejected in suite.rejected
    )


def test_generated_frozen_schema_boundary_suite_is_used_by_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _write_target(tmp_path, WEATHER_AGENT)
    assert main(["generate", str(target)]) == 0
    output = capsys.readouterr().out
    assert "Frozen suite written." in output
    assert (target / DEFAULT_SUITE_FILENAME).is_file()

    monkeypatch.setattr(application, "_execute_valid_scenarios", _stop_before_worker)
    with pytest.raises(_StoppedBeforeWorker) as exc_info:
        application.execute_suite(target, persist_store=False)

    captured = exc_info.value.captured
    frozen = captured["frozen"]
    valid = captured["valid"]
    assert frozen is not None
    assert all(case.lineage.origin is not CaseOrigin.BUILT_IN for case in frozen.cases)
    assert any(case.lineage.origin is CaseOrigin.SCHEMA_BOUNDARY for case in frozen.cases)
    assert {scenario.scenario_id for scenario in valid} == {
        case.scenario.scenario_id for case in frozen.cases
    }
    assert "happy_lookup" not in {scenario.scenario_id for scenario in valid}


def test_cli_test_with_frozen_suite_does_not_report_suite_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _write_target(tmp_path, WEATHER_AGENT)
    assert main(["generate", str(target)]) == 0
    capsys.readouterr()

    assert main(["test", str(target), "--no-store"]) == 2
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "No compatible built-in suite" not in combined
    assert "No valid scenarios remain after linting" not in captured.err
    assert "INFRA_ERROR" in captured.out
    assert "Using frozen suite" in captured.out


def test_unsupported_preflight_is_reported_before_suite_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _write_target(tmp_path, DYNAMIC_INSTRUCTIONS)

    with pytest.raises(UnsupportedTargetError, match="dynamic_instructions"):
        application.execute_suite(target, persist_store=False)

    assert main(["test", str(target), "--no-store"]) == 2
    err = capsys.readouterr().err
    assert "dynamic_instructions" in err
    assert "No compatible built-in suite" not in err
    assert "No valid scenarios remain after linting" not in err


def test_supported_target_with_no_tools_still_refuses_empty_generate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _write_target(tmp_path, NO_TOOLS_AGENT)

    with pytest.raises(IncompatibleSuiteError, match="No compatible built-in suite"):
        application.execute_suite(target, persist_store=False)
    with pytest.raises(ScenarioValidationError, match="No compatible cases can be generated"):
        application.generate_suite(target)

    assert main(["test", str(target), "--no-store"]) == 2
    test_err = capsys.readouterr().err
    assert "No compatible built-in suite exists for this target" in test_err
    assert "(none)" in test_err

    assert main(["generate", str(target)]) == 2
    generate_err = capsys.readouterr().err
    assert "No compatible cases can be generated" in generate_err
    assert "No valid scenarios remain after linting" not in generate_err
