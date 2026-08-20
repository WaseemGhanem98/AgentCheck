"""Part B: output-schema-only generation and duplicate-tool-name disambiguation.

Both changes narrow a previously blanket refusal. The tests therefore pin the
*boundary*: the newly supported shape works, and the genuinely ambiguous or
unprovable shape still fails closed.
"""

from __future__ import annotations

from typing import Any

from agents import Agent, FunctionTool, function_tool
from pydantic import BaseModel

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.generate.boundaries import build_output_schema_cases
from agentcheck.domain import OutputCriterionKind


def _spec(agent: Agent) -> Any:
    return OpenAIAgentsAdapter().inspect(agent)


class Plan(BaseModel):
    steps: list[str]
    confidence: float


# --------------------------------------------------------------------------
# Output-schema generation for tool-less structured-output agents
# --------------------------------------------------------------------------


def test_toolless_structured_output_agent_gets_a_deterministic_case() -> None:
    """The real email-workflow triage shape: authoritative output_type, 0 tools."""

    agent = Agent(
        name="Triage",
        instructions="Triage the email.",
        output_type=Plan,
        model="gpt-4.1-mini",
    )
    spec = _spec(agent)
    assert spec.tools.items == ()

    cases = build_output_schema_cases(spec, seed=1729)

    assert len(cases) == 1
    scenario = cases[0]
    criterion = scenario.output_criteria[0]
    assert criterion.kind == OutputCriterionKind.JSON_SCHEMA
    # The oracle carries the statically derived schema, not a paraphrase of it.
    assert criterion.parameters["schema"] == spec.interface.output_schema.value
    assert criterion.parameters["schema"]["properties"]["steps"]["type"] == "array"


def test_output_schema_generation_is_deterministic() -> None:
    """Same spec and seed must reproduce an identical fingerprint."""

    agent = Agent(
        name="Triage", instructions="t", output_type=Plan, model="gpt-4.1-mini"
    )
    first = build_output_schema_cases(_spec(agent), seed=1729)
    second = build_output_schema_cases(_spec(agent), seed=1729)

    assert first[0].fingerprint == second[0].fingerprint
    # Nothing is sampled, so the case is stable rather than seed-shuffled.
    assert first[0].scenario_id == second[0].scenario_id


def test_agent_without_output_type_gets_no_output_schema_case() -> None:
    """Plain text agents keep the previous behaviour: no case invented."""

    agent = Agent(name="Chat", instructions="Say hi.", model="gpt-4.1-mini")

    assert build_output_schema_cases(_spec(agent), seed=1729) == ()


def test_non_authoritative_output_schema_yields_no_case() -> None:
    """A schema AgentCheck could not prove must not become an oracle.

    This is the fail-closed half: an unprovable schema produces no case rather
    than a case asserting something unverified.
    """

    agent = Agent(
        name="Triage", instructions="t", output_type=Plan, model="gpt-4.1-mini"
    )
    spec = _spec(agent)
    degraded = spec.model_copy(
        update={
            "interface": spec.interface.model_copy(
                update={
                    "output_schema": spec.interface.output_schema.model_copy(
                        update={"authoritative": False}
                    )
                }
            )
        }
    )

    assert build_output_schema_cases(degraded, seed=1729) == ()


# --------------------------------------------------------------------------
# duplicate_tool_name
# --------------------------------------------------------------------------


def _codes(agent: Agent) -> set[str]:
    return {issue.code for issue in OpenAIAgentsAdapter().preflight(agent).issues}


def test_one_tool_shared_by_two_agents_is_not_ambiguous() -> None:
    """The real cs-airline shape: one imported tool reused across agents.

    The gateway intercepts by name and validates against the schema, so an
    identical contract means the same call means the same thing wherever it is
    reached. This previously blocked the target as a false positive.
    """

    @function_tool
    def faq_lookup_tool(question: str) -> str:
        """Look up an FAQ answer."""
        return "answer"

    faq = Agent(
        name="FAQ",
        instructions="Answer questions.",
        tools=[faq_lookup_tool],
        model="gpt-4.1-mini",
    )
    triage = Agent(
        name="Triage",
        instructions="Route.",
        tools=[faq_lookup_tool],
        handoffs=[faq],
        model="gpt-4.1-mini",
    )

    assert "duplicate_tool_name" not in _codes(triage)


def test_same_name_with_conflicting_schema_still_fails_closed() -> None:
    """A real contract conflict remains undecidable and must stay rejected."""

    @function_tool(name_override="lookup")
    def lookup_a(question: str) -> str:
        """Look up by question."""
        return "a"

    @function_tool(name_override="lookup")
    def lookup_b(account_id: int, region: str) -> str:
        """Look up by account."""
        return "b"

    other = Agent(
        name="Other", instructions="o", tools=[lookup_b], model="gpt-4.1-mini"
    )
    root = Agent(
        name="Root",
        instructions="r",
        tools=[lookup_a],
        handoffs=[other],
        model="gpt-4.1-mini",
    )

    assert "duplicate_tool_name" in _codes(root)


def test_same_name_and_schema_with_different_description_fails_closed() -> None:
    """Description is part of the contract the model sees, so a conflict counts."""

    shared_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
        "additionalProperties": False,
    }

    async def _never_invoked(_ctx: Any, _args: str) -> str:
        raise AssertionError("original tool handler must never run during preflight")

    first = FunctionTool(
        name="lookup",
        description="Look up an FAQ answer.",
        params_json_schema=shared_schema,
        on_invoke_tool=_never_invoked,
    )
    second = FunctionTool(
        name="lookup",
        description="Completely different behaviour for the same name.",
        params_json_schema=shared_schema,
        on_invoke_tool=_never_invoked,
    )
    other = Agent(name="Other", instructions="o", tools=[second], model="gpt-4.1-mini")
    root = Agent(
        name="Root",
        instructions="r",
        tools=[first],
        handoffs=[other],
        model="gpt-4.1-mini",
    )

    assert "duplicate_tool_name" in _codes(root)


def test_duplicate_tool_check_never_invokes_the_original_handler() -> None:
    """Contract identity is read from declared schemas, never by calling the tool."""

    calls: list[str] = []

    @function_tool
    def shared(question: str) -> str:
        """Shared tool."""
        calls.append(question)  # pragma: no cover - must never execute
        return "x"

    other = Agent(name="Other", instructions="o", tools=[shared], model="gpt-4.1-mini")
    root = Agent(
        name="Root",
        instructions="r",
        tools=[shared],
        handoffs=[other],
        model="gpt-4.1-mini",
    )

    report = OpenAIAgentsAdapter().preflight(root)

    assert "duplicate_tool_name" not in {issue.code for issue in report.issues}
    assert calls == []
