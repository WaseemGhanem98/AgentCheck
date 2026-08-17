from __future__ import annotations

import json
from pathlib import Path

import pytest

import agentcheck.application as application
from agentcheck.adapters import UnsupportedTargetError, decode_preflight_report
from agentcheck.adapters.base import PreflightReport, SupportIssue, encode_preflight_report
from agentcheck.cli import main
from agentcheck.domain import AgentSpec
from agentcheck.errors import IncompatibleSuiteError, ScenarioValidationError
from agentcheck.generate import DEFAULT_SUITE_FILENAME
from agentcheck.initialize import write_initial_config


def _write_target(tmp_path: Path, source: str, *, name: str = "target") -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "agent.py").write_text(source.lstrip(), encoding="utf-8")
    write_initial_config(root)
    return root


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

STRUCTURED_OUTPUT = """
from pydantic import BaseModel
from agents import Agent


class Plan(BaseModel):
    steps: list[str]


agent = Agent(
    name="Planner",
    instructions="Plan the work.",
    output_type=Plan,
    model="gpt-4.1-mini",
)
"""

SIMPLE_FUNCTION_TOOL = """
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

SUPPORTED_NO_TOOLS = """
from agents import Agent

agent = Agent(
    name="Chat",
    instructions="Say hello.",
    model="gpt-4.1-mini",
)
"""

HANDOFF_AGENT = """
from agents import Agent

faq = Agent(name="FAQ", instructions="Answer FAQ questions.", model="gpt-4.1-mini")
agent = Agent(
    name="Triage",
    instructions="Route the request.",
    handoffs=[faq],
    model="gpt-4.1-mini",
)
"""

UNSUPPORTED_TOOL_TYPE = """
from agents import Agent, WebSearchTool

agent = Agent(
    name="Search",
    instructions="Search the web.",
    tools=[WebSearchTool()],
    model="gpt-4.1-mini",
)
"""


def test_preflight_report_round_trips_and_rejects_unknown_fields() -> None:
    report = PreflightReport(
        framework="openai_agents",
        issues=(
            SupportIssue(
                code="dynamic_instructions",
                message="Dynamic instruction callbacks are unsupported in Phase 1.",
                location="agent.instructions",
            ),
        ),
    )
    payload = encode_preflight_report(report)
    restored = decode_preflight_report(payload)
    assert restored == report
    with pytest.raises(ValueError, match="unknown fields"):
        decode_preflight_report({**payload, "extra": True})


def test_inspect_lists_dynamic_instructions_and_keeps_exit_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _write_target(tmp_path, DYNAMIC_INSTRUCTIONS)

    assert main(["inspect", str(target)]) == 0
    output = capsys.readouterr().out
    assert "Preflight: unsupported" in output
    assert "- dynamic_instructions (agent.instructions):" in output
    assert "Unknown properties: 1" in output

    assert main(["inspect", str(target), "--json"]) == 0
    payload = capsys.readouterr().out
    spec = AgentSpec.model_validate_json(payload)
    assert spec.identity.name.value == "Chat agent"
    assert "preflight" not in json.loads(payload)


def test_inspect_lists_structured_output_handoffs_and_unsupported_tool_type(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    structured = _write_target(tmp_path, STRUCTURED_OUTPUT, name="structured")
    assert main(["inspect", str(structured)]) == 0
    structured_out = capsys.readouterr().out
    assert "Preflight: unsupported" in structured_out
    assert "- structured_output (agent.output_type):" in structured_out

    handoffs = _write_target(tmp_path, HANDOFF_AGENT, name="handoffs")
    assert main(["inspect", str(handoffs)]) == 0
    handoff_out = capsys.readouterr().out
    assert "- handoffs (agent.handoffs):" in handoff_out

    hosted = _write_target(tmp_path, UNSUPPORTED_TOOL_TYPE, name="hosted")
    assert main(["inspect", str(hosted)]) == 0
    hosted_out = capsys.readouterr().out
    assert "- unsupported_tool_type (agent.tools[0]):" in hosted_out


def test_inspect_supported_function_tool_agent_reports_preflight_supported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _write_target(tmp_path, SIMPLE_FUNCTION_TOOL)

    assert main(["inspect", str(target)]) == 0
    output = capsys.readouterr().out
    assert "Preflight: supported" in output
    assert "✓ get_weather" in output
    assert "unsupported" not in output.lower().split("preflight:", 1)[1]


def test_generate_and_test_surface_dynamic_instructions_not_empty_suite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _write_target(tmp_path, DYNAMIC_INSTRUCTIONS)

    with pytest.raises(UnsupportedTargetError, match="dynamic_instructions"):
        application.generate_suite(target)
    assert not (target / DEFAULT_SUITE_FILENAME).exists()

    with pytest.raises(UnsupportedTargetError, match="dynamic_instructions"):
        application.execute_suite(target, persist_store=False)

    assert main(["generate", str(target)]) == 2
    generate_err = capsys.readouterr().err
    assert "dynamic_instructions" in generate_err
    assert "No valid scenarios remain after linting" not in generate_err

    assert main(["test", str(target), "--no-store"]) == 2
    test_err = capsys.readouterr().err
    assert "dynamic_instructions" in test_err
    assert "No valid scenarios remain after linting" not in test_err
    assert "No compatible built-in suite" not in test_err


def test_generate_and_test_surface_structured_output_not_empty_suite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _write_target(tmp_path, STRUCTURED_OUTPUT)

    with pytest.raises(UnsupportedTargetError, match="structured_output"):
        application.generate_suite(target)
    with pytest.raises(UnsupportedTargetError, match="structured_output"):
        application.execute_suite(target, persist_store=False)

    assert main(["generate", str(target)]) == 2
    generate_err = capsys.readouterr().err
    assert "structured_output" in generate_err
    assert "No valid scenarios remain after linting" not in generate_err

    assert main(["test", str(target), "--no-store"]) == 2
    test_err = capsys.readouterr().err
    assert "structured_output" in test_err
    assert "No valid scenarios remain after linting" not in test_err
    assert "No compatible built-in suite" not in test_err


def test_supported_function_tool_agent_can_generate_schema_boundaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _write_target(tmp_path, SIMPLE_FUNCTION_TOOL)

    assert main(["generate", str(target), "--force"]) == 0
    output = capsys.readouterr().out
    assert "Preflight: supported" in output
    assert "Frozen suite written." in output
    suite = json.loads((target / DEFAULT_SUITE_FILENAME).read_text(encoding="utf-8"))
    assert len(suite["cases"]) >= 1
    assert any(
        case["lineage"]["origin"] == "schema_boundary" for case in suite["cases"]
    )


def test_supported_target_with_no_matching_suite_keeps_empty_generate_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _write_target(tmp_path, SUPPORTED_NO_TOOLS)

    with pytest.raises(ScenarioValidationError, match="No valid scenarios remain"):
        application.generate_suite(target)
    with pytest.raises(IncompatibleSuiteError, match="No compatible built-in suite"):
        application.execute_suite(target, persist_store=False)

    assert main(["generate", str(target)]) == 2
    generate_err = capsys.readouterr().err
    assert "No valid scenarios remain after linting" in generate_err
    assert "dynamic_instructions" not in generate_err
    assert "structured_output" not in generate_err

    assert main(["test", str(target), "--no-store"]) == 2
    test_err = capsys.readouterr().err
    assert "No compatible built-in suite exists for this target" in test_err
    assert "No valid scenarios remain after linting" not in test_err

    assert main(["inspect", str(target)]) == 0
    assert "Preflight: supported" in capsys.readouterr().out
