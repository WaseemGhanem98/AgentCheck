"""PydanticAI "agent delegation" / "programmatic hand-off": audit findings.

Reading the pinned SDK's own documentation (docs/multi-agent-applications.md)
establishes that neither pattern is a framework-level primitive comparable to
the OpenAI Agents SDK's `Handoff`:

- "Agent delegation" is an ordinary `@agent.tool` function whose *body*
  happens to call `await other_agent.run(...)`. There is no distinct object
  type, no registration on the parent `Agent`, and no attribute PydanticAI
  itself exposes to say "this tool delegates". The sub-agent reference lives
  entirely inside source code AgentCheck already replaces and never executes.
- "Programmatic hand-off" is application code (a loop, a human-in-the-loop
  prompt, arbitrary branching) calling one `Agent.run()` after another. There
  is no single `Agent` object representing the hand-off at all -- it is two
  or more independent targets driven by orchestration outside any single
  `pydantic_ai.Agent`, which is what AgentCheck's PydanticAI adapter inspects
  and prepares.

Building a "multi-agent topology" abstraction for either pattern would be
exactly the fake abstraction based on names this project's own standards
warn against: there is no real structure to introspect, and the delegating
tool's real behaviour is already unreachable under the existing single-tool
replacement guarantee. This file proves that directly: a delegating tool
that genuinely would drive a second real `pydantic_ai.Agent`, with its own
tripwired tool, never reaches either tripwire.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agentcheck.adapters import PydanticAIAdapter
from agentcheck.domain import SimulatedToolOutcome, SimulatedToolStatus, ToolFixture
from agentcheck.runner import ToolGateway

from tests.agentcheck.test_pydantic_ai_adapter import (
    ORIGINAL_HANDLER_CALLS,
    _prepare,
    _run,
    _script,
)


@dataclass
class Deps:
    token: str


SUB_AGENT_HANDLER_CALLS: list[str] = []


def _sub_agent_tripwire_model(messages: list, info) -> ModelResponse:  # type: ignore[no-untyped-def]
    raise AssertionError("the delegate sub-agent's model must never be invoked")


def test_a_delegating_tool_never_reaches_the_sub_agent_it_would_call() -> None:
    """`joke_factory` here mirrors the SDK's own agent-delegation example
    exactly: its real body calls a second, independent Agent. Both the
    parent tool's own tripwire and the sub-agent's model/tool tripwires must
    stay unreached -- proving the delegation is inert, not merely untested."""

    ORIGINAL_HANDLER_CALLS.clear()
    SUB_AGENT_HANDLER_CALLS.clear()

    joke_generation_agent = Agent(FunctionModel(_sub_agent_tripwire_model), output_type=list[str])

    @joke_generation_agent.tool_plain
    def get_jokes(count: int) -> str:  # pragma: no cover
        SUB_AGENT_HANDLER_CALLS.append("get_jokes")
        raise AssertionError("the delegate sub-agent's own tool must never run")

    joke_selection_agent = Agent(
        _script(_message_response("joke_factory", {"count": 3}), _text_response("done")),
        deps_type=Deps,
    )

    @joke_selection_agent.tool
    async def joke_factory(ctx: RunContext[Deps], count: int) -> list[str]:
        ORIGINAL_HANDLER_CALLS.append("joke_factory")
        result = await joke_generation_agent.run(f"Please generate {count} jokes.")
        return result.output

    report = PydanticAIAdapter().preflight(joke_selection_agent)
    assert report.supported, [(i.code, i.message) for i in report.issues]

    gateway = ToolGateway(
        [
            item.value
            for item in PydanticAIAdapter().inspect(joke_selection_agent).tools.items
        ],
        [
            ToolFixture(
                fixture_id="f",
                tool_name="joke_factory",
                outcome=SimulatedToolOutcome(
                    status=SimulatedToolStatus.SUCCESS, result=["a simulated joke"]
                ),
            )
        ],
    )
    prepared = _prepare(joke_selection_agent, gateway)
    run = _run(prepared, "tell me a joke")

    assert ORIGINAL_HANDLER_CALLS == []
    assert SUB_AGENT_HANDLER_CALLS == []
    assert run.tool_outcomes[0].result == ["a simulated joke"]


def test_the_sub_agent_object_is_invisible_to_preflight_and_inspection() -> None:
    """The parent agent's declared toolset/topology must not be polluted by
    a sub-agent referenced only from inside a tool body -- there is no
    connection PydanticAI itself exposes for AgentCheck to find, and none
    should be invented."""

    joke_generation_agent = Agent(FunctionModel(_sub_agent_tripwire_model), output_type=list[str])
    joke_selection_agent = Agent(_script(_text_response("unused")))

    @joke_selection_agent.tool
    async def joke_factory(ctx: RunContext[None], count: int) -> list[str]:  # pragma: no cover
        result = await joke_generation_agent.run("x")
        return result.output

    spec = PydanticAIAdapter().inspect(joke_selection_agent)

    assert [item.value.name for item in spec.tools.items] == ["joke_factory"]
    # No trace of the sub-agent's own tools, name, or model leaks into the
    # parent's declared surface.
    assert "get_jokes" not in {item.value.name for item in spec.tools.items}


def _message_response(name: str, args: dict) -> ModelResponse:  # type: ignore[type-arg]
    return ModelResponse(parts=[ToolCallPart(name, args)])


def _text_response(value: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(value)])
