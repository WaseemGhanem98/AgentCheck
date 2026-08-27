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

SUPPORTED_STRUCTURED_OUTPUT = """
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

SUPPORTED_STRUCTURED_OUTPUT_WITH_TOOL = """
from pydantic import BaseModel
from agents import Agent, function_tool


class EmailDraft(BaseModel):
    to: str
    subject: str
    body: str


@function_tool
def lookup_recipient(name: str) -> str:
    return "user@example.com"


agent = Agent(
    name="Email Drafter",
    instructions="Draft an email.",
    tools=[lookup_recipient],
    output_type=EmailDraft,
    model="gpt-4.1-mini",
)
"""

CUSTOM_STRUCTURED_OUTPUT = """
from typing import Any
from agents import Agent
from agents.agent_output import AgentOutputSchemaBase


class CustomOutputSchema(AgentOutputSchemaBase):
    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return "CustomOutputSchema"

    def json_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"steps": {"type": "array"}}}

    def is_strict_json_schema(self) -> bool:
        return False

    def validate_json(self, json_str: str) -> Any:
        import json as _json

        return _json.loads(json_str)


agent = Agent(
    name="Planner",
    instructions="Plan the work.",
    output_type=CustomOutputSchema(),
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

SUPPORTED_CALLBACK_HANDOFF_AGENT = """
import random
from agents import Agent, handoff


async def on_seat_booking_handoff(context):
    flight_number = f"FLT-{random.randint(100, 999)}"
    context.context.flight_number = flight_number


faq = Agent(name="FAQ", instructions="Answer FAQ questions.", model="gpt-4.1-mini")
seat = Agent(name="Seat Booking", instructions="Book seats.", model="gpt-4.1-mini")
agent = Agent(
    name="Triage",
    instructions="Route the request.",
    handoffs=[
        faq,
        handoff(
            seat,
            on_handoff=on_seat_booking_handoff,
            tool_name_override="transfer_to_seat_booking_agent",
        ),
    ],
    model="gpt-4.1-mini",
)
"""

CALLBACK_HANDOFF_AGENT = """
from agents import Agent, handoff

async def on_seat_booking_handoff(context):
    del context

faq = Agent(name="FAQ", instructions="Answer FAQ questions.", model="gpt-4.1-mini")
agent = Agent(
    name="Triage",
    instructions="Route the request.",
    handoffs=[handoff(faq, on_handoff=on_seat_booking_handoff)],
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


def test_wrong_framework_target_names_the_likely_correct_adapter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A target evaluated with the wrong --adapter gets an actionable hint.

    Pure name-based duck-typing (module + class name of the already-imported
    object) -- no cross-adapter import, so this works whichever adapter's
    extra happens to be installed alongside the other in the dev environment.
    """

    root = tmp_path / "wrong_adapter"
    root.mkdir()
    (root / "agent.py").write_text(
        "from agents import Agent\n\n"
        'agent = Agent(name="Support", instructions="Help.", model="gpt-4.1-mini")\n',
        encoding="utf-8",
    )
    write_initial_config(root, adapter="pydantic_ai")

    assert main(["inspect", str(root)]) == 0
    output = capsys.readouterr().out
    assert "unsupported_agent_type" in output
    assert 'set "adapter": \'openai_agents\' in agentcheck.json instead' in output


def test_wrong_framework_target_names_the_likely_correct_adapter_the_other_way(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The openai_agents adapter refuses a wrong-type target outright.

    Unlike pydantic_ai's tolerant, best-effort ``inspect()`` (which still
    produces a spec and defers the whole "is this supported" decision to
    preflight), openai_agents' ``inspect()`` refuses immediately -- so the CLI
    command exits non-zero here rather than 0. What changed is only that the
    refusal now carries the same actionable message preflight() would give,
    instead of a bare, generic ``TypeError``.
    """

    root = tmp_path / "wrong_adapter_reverse"
    root.mkdir()
    (root / "agent.py").write_text(
        "from pydantic_ai import Agent\n\n"
        'agent = Agent("test", instructions="Help.", name="Support")\n',
        encoding="utf-8",
    )
    write_initial_config(root, adapter="openai_agents")

    assert main(["inspect", str(root)]) == 2
    error = capsys.readouterr().err
    assert "unsupported_agent_type" in error
    assert 'set "adapter": \'pydantic_ai\' in agentcheck.json instead' in error


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
    structured = _write_target(tmp_path, CUSTOM_STRUCTURED_OUTPUT, name="structured")
    assert main(["inspect", str(structured)]) == 0
    structured_out = capsys.readouterr().out
    assert "Preflight: unsupported" in structured_out
    assert "- structured_output (agent.output_type):" in structured_out

    handoffs = _write_target(tmp_path, HANDOFF_AGENT, name="handoffs")
    assert main(["inspect", str(handoffs)]) == 0
    handoff_out = capsys.readouterr().out
    assert "Handoff topology (2 reachable agents):" in handoff_out
    assert "Preflight: supported" in handoff_out
    assert "- handoffs (agent.handoffs):" not in handoff_out

    callback = _write_target(tmp_path, CALLBACK_HANDOFF_AGENT, name="callback-handoffs")
    assert main(["inspect", str(callback)]) == 0
    callback_out = capsys.readouterr().out
    assert "Handoff topology (2 reachable agents):" in callback_out
    assert "- handoff_callback (agent.handoffs[0].on_handoff):" in callback_out

    supported_callback = _write_target(
        tmp_path, SUPPORTED_CALLBACK_HANDOFF_AGENT, name="supported-callback"
    )
    assert main(["inspect", str(supported_callback)]) == 0
    supported_out = capsys.readouterr().out
    assert "Handoff topology (3 reachable agents):" in supported_out
    assert "Preflight: supported" in supported_out
    assert "handoff_callback" not in supported_out
    assert "context assignment: flight_number='FLT-100'" in supported_out

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
    target = _write_target(tmp_path, CUSTOM_STRUCTURED_OUTPUT)

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


def test_inspect_supported_structured_output_agent_reports_preflight_supported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _write_target(tmp_path, SUPPORTED_STRUCTURED_OUTPUT)

    assert main(["inspect", str(target)]) == 0
    output = capsys.readouterr().out
    assert "Preflight: supported" in output
    assert "structured_output" not in output

    assert main(["inspect", str(target), "--json"]) == 0
    payload = capsys.readouterr().out
    spec = AgentSpec.model_validate_json(payload)
    assert spec.interface.output_schema.value == {
        "type": "object",
        "properties": {
            "steps": {"type": "array", "items": {"type": "string"}, "title": "Steps"}
        },
        "required": ["steps"],
        "title": "Plan",
    }


def test_supported_structured_output_agent_with_no_tools_generates_output_schema_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tool-less structured-output agent is now covered by its own contract.

    This previously produced "No compatible cases can be generated": boundary
    generation had no FunctionTool to derive from. The declared output schema is
    itself an authoritative contract, so conformance to it is a deterministic
    offline assertion.
    """

    target = _write_target(tmp_path, SUPPORTED_STRUCTURED_OUTPUT)

    assert main(["generate", str(target), "--force"]) == 0
    output = capsys.readouterr().out
    assert "Preflight: supported" in output
    assert "Frozen suite written." in output

    suite = json.loads((target / DEFAULT_SUITE_FILENAME).read_text(encoding="utf-8"))
    origins = {case["lineage"]["origin"] for case in suite["cases"]}
    assert "output_schema" in origins

    case = next(
        item for item in suite["cases"] if item["lineage"]["origin"] == "output_schema"
    )
    criterion = case["scenario"]["output_criteria"][0]
    assert criterion["kind"] == "json_schema"
    assert criterion["parameters"]["schema"]["title"] == "Plan"


def test_supported_structured_output_agent_with_tool_can_generate_schema_boundaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _write_target(tmp_path, SUPPORTED_STRUCTURED_OUTPUT_WITH_TOOL)

    assert main(["generate", str(target), "--force"]) == 0
    output = capsys.readouterr().out
    assert "Preflight: supported" in output
    assert "Frozen suite written." in output
    suite = json.loads((target / DEFAULT_SUITE_FILENAME).read_text(encoding="utf-8"))
    assert len(suite["cases"]) >= 1
    assert any(
        case["lineage"]["origin"] == "schema_boundary" for case in suite["cases"]
    )


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

    with pytest.raises(ScenarioValidationError, match="No compatible cases can be generated"):
        application.generate_suite(target)
    with pytest.raises(IncompatibleSuiteError, match="No compatible built-in suite"):
        application.execute_suite(target, persist_store=False)

    assert main(["generate", str(target)]) == 2
    generate_err = capsys.readouterr().err
    assert "No compatible cases can be generated" in generate_err
    assert "No valid scenarios remain after linting" not in generate_err
    assert "dynamic_instructions" not in generate_err
    assert "structured_output" not in generate_err

    assert main(["test", str(target), "--no-store"]) == 2
    test_err = capsys.readouterr().err
    assert "No compatible built-in suite exists for this target" in test_err
    assert "No valid scenarios remain after linting" not in test_err

    assert main(["inspect", str(target)]) == 0
    assert "Preflight: supported" in capsys.readouterr().out


def test_generate_and_test_surface_handoff_callback_not_empty_suite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _write_target(tmp_path, CALLBACK_HANDOFF_AGENT)

    with pytest.raises(UnsupportedTargetError, match="handoff_callback"):
        application.generate_suite(target)
    with pytest.raises(UnsupportedTargetError, match="handoff_callback"):
        application.execute_suite(target, persist_store=False)

    assert main(["generate", str(target)]) == 2
    generate_err = capsys.readouterr().err
    assert "handoff_callback" in generate_err
    assert "No valid scenarios remain after linting" not in generate_err
    assert "handoffs (" not in generate_err

    assert main(["test", str(target), "--no-store"]) == 2
    test_err = capsys.readouterr().err
    assert "handoff_callback" in test_err
    assert "No valid scenarios remain after linting" not in test_err
    assert "No compatible built-in suite" not in test_err
